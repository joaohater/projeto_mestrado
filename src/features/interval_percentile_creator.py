"""
interval_percentile_creator.py

Gera features de percentis sobre janela deslizante (rolling window).
Extensão do interval_creator.py — mesma lógica, mas usa percentis
no lugar de min/mean/max.

Percentis padrão (pedido do professor): 5, 10, 25, 50, 75, 90, 95
Janelas padrão: w=7, w=14, w=30

Regra anti-leakage (idêntica ao interval_creator):
  - shift(1) ANTES do rolling → valor atual nunca entra na janela
  - Treino: dropna=True  (remove as primeiras W linhas sem contexto)
  - Teste : dropna=False + fill com contexto do treino
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Função principal de geração de percentis
# ---------------------------------------------------------------------------

def generate_percentiles(
    df: pd.DataFrame,
    value_col: str = 'value',
    window_size: int = 7,
    percentiles: list = [5, 10, 25, 50, 75, 90, 95],
    dropna: bool = True
) -> pd.DataFrame:
    """
    Gera colunas de percentis sobre janela deslizante.

    Parâmetros
    ----------
    df          : DataFrame com colunas ['timestamp', value_col]
    value_col   : nome da coluna de valores
    window_size : tamanho da janela W
    percentiles : lista de percentis a calcular (0-100)
    dropna      : True para treino, False para teste

    Retorna
    -------
    DataFrame com colunas originais + p{P}_{W} para cada percentil P
    """
    result = df.copy()

    # shift(1): valor atual nunca entra na janela — evita data leakage
    shifted = result[value_col].shift(1)

    for p in percentiles:
        col_name = f'p{p}_{window_size}'
        result[col_name] = (
            shifted
            .rolling(window=window_size, min_periods=window_size)
            .quantile(p / 100.0)
        )

    if dropna:
        result = result.dropna().reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Preenchimento do teste com contexto do treino (idêntico ao interval_creator)
# ---------------------------------------------------------------------------

def fill_test_percentiles_from_train(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    value_col: str = 'value',
    window_size: int = 7,
    percentiles: list = [5, 10, 25, 50, 75, 90, 95]
) -> pd.DataFrame:
    """
    Preenche os NaN do teste usando contexto do treino.

    As primeiras W linhas do teste não têm janela completa — usa as
    últimas W observações do treino para completar o rolling.

    Parâmetros
    ----------
    df_test  : DataFrame do teste (com NaN nas primeiras W linhas)
    df_train : DataFrame do treino (para extrair contexto)
    value_col, window_size, percentiles : mesmos usados na geração

    Retorna
    -------
    DataFrame do teste sem NaN nas colunas de percentil
    """
    # Contexto: últimas W linhas do treino
    contexto = df_train[[value_col]].tail(window_size).copy()

    # Concatena contexto + teste para recalcular o rolling
    df_combined = pd.concat(
        [contexto, df_test[[value_col]]],
        ignore_index=True
    )

    shifted = df_combined[value_col].shift(1)

    for p in percentiles:
        col_name = f'p{p}_{window_size}'
        valores = (
            shifted
            .rolling(window=window_size, min_periods=window_size)
            .quantile(p / 100.0)
        )
        # Remove as linhas do contexto — fica só o teste
        df_test = df_test.copy()
        df_test[col_name] = valores.iloc[window_size:].values

    return df_test


# ---------------------------------------------------------------------------
# Processamento em lote (espelho do IntervalCreator)
# ---------------------------------------------------------------------------

class PercentileCreator:
    """
    Processa splits train/test gerando features de percentis.

    Uso:
        pc = PercentileCreator(window_size=7)
        pc.process_split_folder(
            split_dir='../data/splits/univariate/',
            output_dir='../data/processed/univariate/percentile_7/'
        )
    """

    def __init__(
        self,
        window_size: int = 7,
        percentiles: list = [5, 10, 25, 50, 75, 90, 95],
        value_col: str = 'value'
    ):
        self.window_size = window_size
        self.percentiles  = percentiles
        self.value_col    = value_col

    def process_split_folder(self, split_dir: str, output_dir: str):
        """
        Lê todos os CSVs de split_dir/train/ e split_dir/test/,
        gera percentis e salva em output_dir/train/ e output_dir/test/.
        """
        train_in  = os.path.join(split_dir,  'train')
        test_in   = os.path.join(split_dir,  'test')
        train_out = os.path.join(output_dir, 'train')
        test_out  = os.path.join(output_dir, 'test')

        os.makedirs(train_out, exist_ok=True)
        os.makedirs(test_out,  exist_ok=True)

        arquivos = sorted(f for f in os.listdir(train_in) if f.endswith('.csv'))

        for arquivo in arquivos:
            print(f"  Processando {arquivo} ...")

            # Nome do arquivo de teste: troca _train.csv por _test.csv
            arquivo_test = arquivo.replace('_train.csv', '_test.csv')

            df_train = pd.read_csv(os.path.join(train_in, arquivo))
            df_test  = pd.read_csv(os.path.join(test_in,  arquivo_test))

            # Treino: dropna=True
            df_train_proc = generate_percentiles(
                df_train,
                value_col=self.value_col,
                window_size=self.window_size,
                percentiles=self.percentiles,
                dropna=True
            )

            # Teste: dropna=False + fill com contexto do treino
            df_test_proc = generate_percentiles(
                df_test,
                value_col=self.value_col,
                window_size=self.window_size,
                percentiles=self.percentiles,
                dropna=False
            )
            df_test_proc = fill_test_percentiles_from_train(
                df_test_proc,
                df_train,
                value_col=self.value_col,
                window_size=self.window_size,
                percentiles=self.percentiles
            )

            # Salva
            nome_base_train = arquivo.replace('.csv', '')
            nome_base_test  = arquivo_test.replace('.csv', '')
            df_train_proc.to_csv(
                os.path.join(train_out, f'{nome_base_train}_percentile_{self.window_size}.csv'),
                index=False
            )
            df_test_proc.to_csv(
                os.path.join(test_out, f'{nome_base_test}_percentile_{self.window_size}.csv'),
                index=False
            )

        print(f"  Concluido: {len(arquivos)} arquivos → {output_dir}")
