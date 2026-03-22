"""
interval_creator.py

Modulo responsavel pela geracao de features de aritmetica intervalar a partir
dos CSVs de treino e teste produzidos pelo modulo splitter.py.

Fundamentacao teorica:
    A Aritmetica Intervalar (IA) representa cada observacao nao como um valor
    pontual, mas como um intervalo [minimo, maximo] com ponto central (media),
    calculado sobre uma janela deslizante de tamanho fixo. 

    Aplicada a series temporais, cada observacao t e representada por:
        min_W  = min(value[t-W], ..., value[t-1])
        mean_W = mean(value[t-W], ..., value[t-1])
        max_W  = max(value[t-W], ..., value[t-1])

    onde W e o tamanho da janela. O valor atual (t) NUNCA entra no calculo
    das suas proprias features — garante ausencia de data leakage.

Diferenca em relacao aos lags (lag_creator.py):
    - Lags (rolling window classico): cada valor da janela vira uma feature
      separada -> W features por observacao
    - Aritmetica Intervalar: a janela inteira e comprimida em 3 features
      (min, mean, max) -> 3 features por observacao, independente de W

    Exemplo com W=7:
        Lags:    lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7  (7 colunas)
        IA:      min_7, mean_7, max_7                              (3 colunas)

Contexto de uso no pipeline do projeto:
    Aplicado apos o split temporal (splitter.py), de forma independente sobre
    os conjuntos de treino e teste, eliminando qualquer risco de data leakage.

Fluxo completo do projeto:
    raw/ (.tsf)
        └─► MonashDataLoader                  [data_loader.py]
                └─► TimeSeriesSplitter        [splitter.py]
                        ├─► splits/train/<dataset>_train.csv
                        └─► splits/test/<dataset>_test.csv
                                └─► IntervalCreator  <- este modulo
                                        ├─► processed/train/
                                        │       <dataset>_train_interval_W.csv
                                        └─► processed/test/
                                                <dataset>_test_interval_W.csv

Tratamento do conjunto de teste:
    O teste NAO usa dropna. As primeiras W linhas de cada serie teriam NaN
    porque nao ha observacoes anteriores suficientes dentro do proprio teste.
    Esses NaN sao preenchidos com os ultimos W valores do treino correspondente,
    exatamente como ocorreria em producao.

    Exemplo com W=3 e serie de teste [101, 102, 103, 104, 105]:
        Treino (ultimas obs): [..., 98, 99, 100]

        min_3 da obs 101 = min(98, 99, 100) = 98
        mean_3 da obs 101 = mean(98, 99, 100) = 99.0
        max_3 da obs 101 = max(98, 99, 100) = 100

        -> Nenhuma linha removida do teste.

Uso basico - arquivo unico:
    from src.features.interval_creator import IntervalCreator

    creator = IntervalCreator(window_size=7)
    result = creator.process_file(
        filepath='../data/splits/univariate/train/saugeenday_dataset_train.csv',
        output_dir='../data/processed/univariate/train/',
    )
    result.print_summary()

Uso basico - lote completo (treino e teste):
    from src.features.interval_creator import IntervalCreator

    creator = IntervalCreator(window_size=7)
    report = creator.process_split_folder(
        splits_dir='../data/splits/univariate/',
        output_dir='../data/processed/univariate/',
    )
    report.print_summary()

Experimentando multiplos tamanhos de janela:
    for w in [7, 14, 30]:
        creator = IntervalCreator(window_size=w)
        creator.process_split_folder(
            splits_dir='../data/splits/univariate/',
            output_dir=f'../data/processed/univariate/interval_{w}/',
        )

Autor: [Seu Nome]
Dissertacao: [Titulo da Dissertacao]
Data: 2024
"""

import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Logging - mesmo formato padronizado dos modulos anteriores
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ---------------------------------------------------------------------------
# Constantes do modulo
# ---------------------------------------------------------------------------
DEFAULT_VALUE_COL: str = "value"
DEFAULT_DATE_COL: str = "timestamp"
DEFAULT_SERIES_COL: str = "series_name"
INTERVAL_COL_SUFFIXES: tuple[str, ...] = ("min", "mean", "max")


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------
@dataclass
class IntervalResult:
    """
    Encapsula os metadados e estatisticas de um arquivo processado.

    Atributos:
        input_name (str): Nome base do arquivo de entrada (sem extensao).
        output_path (str): Caminho absoluto do CSV gerado.
        window_size (int): Tamanho da janela deslizante utilizada.
        n_series (int): Numero de series presentes no arquivo.
        shape_before (tuple[int, int]): Shape do DataFrame antes da transformacao.
        shape_after (tuple[int, int]): Shape do DataFrame apos a transformacao.
        dropna (bool): Indica se linhas com NaN foram removidas (treino).
    """

    input_name: str
    output_path: str
    window_size: int
    n_series: int
    shape_before: tuple[int, int]
    shape_after: tuple[int, int]
    dropna: bool

    @property
    def rows_removed(self) -> int:
        """Numero de linhas removidas pelo dropna."""
        if not self.dropna:
            return 0
        return self.shape_before[0] - self.shape_after[0]

    @property
    def cols_added(self) -> int:
        """Numero de colunas de intervalo adicionadas (sempre 3)."""
        return self.shape_after[1] - self.shape_before[1]

    def print_summary(self) -> None:
        """Exibe um resumo detalhado do arquivo processado."""
        sep = "-" * 68
        print(f"\n{sep}")
        print(f"  INTERVALO: {self.input_name}")
        print(sep)
        print(f"  Janela (W)       : {self.window_size}")
        print(f"  Num. de series   : {self.n_series}")
        print(
            f"  Shape antes      : {self.shape_before[0]} linhas x "
            f"{self.shape_before[1]} colunas"
        )
        print(
            f"  Shape apos       : {self.shape_after[0]} linhas x "
            f"{self.shape_after[1]} colunas"
        )
        print(
            f"  Colunas geradas  : {self.cols_added}  "
            f"(min_{self.window_size}, mean_{self.window_size}, "
            f"max_{self.window_size})"
        )
        if self.dropna:
            print(
                f"  Linhas removidas : {self.rows_removed}  "
                f"(dropna=True, {self.window_size} obs x "
                f"{self.n_series} series)"
            )
        else:
            print(
                f"  Linhas removidas : 0  "
                f"(dropna=False, NaN preenchidos com contexto do treino)"
            )
        print(f"\n  Saida -> {self.output_path}")
        print(f"{sep}\n")


@dataclass
class BatchIntervalReport:
    """
    Consolida os resultados do processamento em lote.

    Mantem o mesmo padrao do BatchLagReport do lag_creator.py para
    consistencia de interface no projeto.

    Atributos:
        window_size (int): Tamanho da janela aplicado em todos os arquivos.
        processed (list[IntervalResult]): Resultados dos arquivos processados.
        failed (list[tuple[str, str]]): Pares (arquivo, mensagem de erro).
        skipped (list[str]): Arquivos ignorados (extensao diferente de .csv).
    """

    window_size: int
    processed: list[IntervalResult] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def n_processed(self) -> int:
        return len(self.processed)

    @property
    def n_failed(self) -> int:
        return len(self.failed)

    def print_summary(self) -> None:
        """Exibe o relatorio consolidado do lote."""
        sep = "=" * 68
        print(f"\n{sep}")
        print("  RELATORIO DE GERACAO DE INTERVALOS EM LOTE")
        print(sep)
        print(f"  Janela aplicada : {self.window_size}")
        print(f"  Processados     : {self.n_processed}")
        print(f"  Falhas          : {self.n_failed}")
        print(f"  Ignorados       : {len(self.skipped)}")

        if self.processed:
            print(f"\n  Arquivos gerados:")
            header = (
                f"  {'Arquivo':<50} {'Antes':>12}  {'Apos':>12}"
                f"  {'Removidas':>9}"
            )
            print(header)
            print(f"  {'-'*50} {'-'*12}  {'-'*12}  {'-'*9}")
            for r in self.processed:
                before = f"{r.shape_before[0]}x{r.shape_before[1]}"
                after = f"{r.shape_after[0]}x{r.shape_after[1]}"
                print(
                    f"  {r.input_name:<50} {before:>12}  {after:>12}"
                    f"  {r.rows_removed:>9}"
                )

        if self.failed:
            print(f"\n  Falhas:")
            for fname, err in self.failed:
                print(f"    [ERRO] {fname}: {err}")

        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Contrato abstrato
# ---------------------------------------------------------------------------
class IntervalTransformer(ABC):
    """
    Classe-base abstrata que define o contrato de um transformador intervalar.

    Qualquer transformador de intervalos deve implementar os metodos
    `transform_dataframe` e `process_file`.

    Parametros:
        window_size (int): Tamanho da janela deslizante. Deve ser >= 2.
        dropna (bool): Se True, remove as window_size primeiras linhas com
            NaN apos o rolling. Padrao: True.
    """

    def __init__(self, window_size: int, dropna: bool = True) -> None:
        _validate_window_size(window_size)
        self.window_size = window_size
        self.dropna = dropna

    @abstractmethod
    def transform_dataframe(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: Optional[str],
        series_col: Optional[str],
    ) -> pd.DataFrame:
        """Aplica a transformacao intervalar em um DataFrame completo."""

    @abstractmethod
    def process_file(
        self,
        filepath: str,
        output_dir: str,
        value_col: Optional[str],
        date_col: Optional[str],
    ) -> IntervalResult:
        """Processa um arquivo CSV e salva o resultado com features intervalares."""


# ---------------------------------------------------------------------------
# Funcoes puras
# ---------------------------------------------------------------------------
def generate_intervals(
    df: pd.DataFrame,
    value_col: str,
    window_size: int,
    date_col: Optional[str] = None,
    dropna: bool = True,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gera 3 colunas de aritmetica intervalar para a coluna value_col.

    Para cada observacao t, calcula sobre a janela [t-W, t-1]:
        min_W  = minimo dos W valores anteriores
        mean_W = media dos W valores anteriores
        max_W  = maximo dos W valores anteriores

    O valor atual (t) nunca entra no calculo das suas proprias features.
    Isso e garantido pelo uso de .shift(1) antes do .rolling(W), que
    desloca a serie um passo para frente antes de calcular a janela.

    Parametros:
        df (pd.DataFrame): DataFrame de entrada. Deve conter value_col.
        value_col (str): Nome da coluna com os valores da serie.
        window_size (int): Tamanho da janela deslizante (>= 2).
        date_col (str | None): Coluna de data - preservada intacta.
            Padrao: None.
        dropna (bool): Remove as primeiras window_size linhas com NaN.
            Padrao: True.
        group_col (str | None): Coluna de agrupamento (ex: 'series_name').
            Quando informado, o rolling e aplicado dentro de cada grupo,
            evitando contaminacao entre series distintas. Padrao: None.

    Returns:
        pd.DataFrame: Copia do DataFrame com colunas min_W, mean_W, max_W
            adicionadas a direita.

    Raises:
        ValueError: Se value_col nao existe no DataFrame.
        ValueError: Se window_size < 2.

    Exemplos:
        >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        >>> generate_intervals(df, value_col='value', window_size=3)
           value  min_3  mean_3  max_3
        3      4    1.0     2.0    3.0
        4      5    2.0     3.0    4.0
        5      6    3.0     4.0    5.0
        ...
    """
    _validate_window_size(window_size)
    _validate_column_exists(df, value_col, "value_col")

    if date_col is not None:
        _validate_column_exists(df, date_col, "date_col")

    result = df.copy()
    w = window_size
    min_col = f"min_{w}"
    mean_col = f"mean_{w}"
    max_col = f"max_{w}"

    def _rolling_stats(series: pd.Series) -> pd.DataFrame:
        """Calcula min, mean e max sobre janela deslizante sem o valor atual."""
        # .shift(1) garante que o valor t nao entra na janela de t
        shifted = series.shift(1)
        rolled = shifted.rolling(window=w)
        return pd.DataFrame({
            min_col:  rolled.min(),
            mean_col: rolled.mean(),
            max_col:  rolled.max(),
        }, index=series.index)

    if group_col is not None and group_col in result.columns:
        stats = (
            result.groupby(group_col, sort=False)[value_col]
            .apply(lambda s: _rolling_stats(s))
            .reset_index(level=0, drop=True)
        )
    else:
        stats = _rolling_stats(result[value_col])

    result[min_col] = stats[min_col]
    result[mean_col] = stats[mean_col]
    result[max_col] = stats[max_col]

    if dropna:
        result = result.dropna().reset_index(drop=True)

    return result


def fill_test_intervals_from_train(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    value_col: str,
    window_size: int,
    series_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Preenche os NaN iniciais das colunas intervalares do teste com o
    contexto do treino correspondente, por serie.

    Contexto:
        Apos aplicar generate_intervals(dropna=False) no teste, as primeiras
        window_size linhas de cada serie ficam com NaN porque nao ha
        observacoes anteriores suficientes dentro do proprio teste.

        Esta funcao reconstroi a janela correta para cada uma dessas linhas
        usando a combinacao dos ultimos valores do treino com os valores
        iniciais do teste, exatamente como ocorreria em producao.

        Exemplo com W=3, treino=[..., 98, 99, 100], teste=[101, 102, 103]:
            obs 101: janela = [98, 99, 100]  -> todos do treino
            obs 102: janela = [99, 100, 101] -> 2 do treino + 1 do teste
            obs 103: janela = [100, 101, 102]-> 1 do treino + 2 do teste
            obs 104: janela = [101, 102, 103]-> todos do teste (sem NaN)

    Parametros:
        df_test (pd.DataFrame): DataFrame do teste com colunas min_W, mean_W,
            max_W ja geradas (com NaN nas primeiras window_size linhas).
        df_train (pd.DataFrame): DataFrame do treino com a coluna value_col
            original.
        value_col (str): Nome da coluna de valores em ambos os DataFrames.
        window_size (int): Tamanho da janela (determina quantas linhas
            de cada serie precisam ser preenchidas).
        series_col (str | None): Coluna de identificacao das series.
            Quando None, trata todo o DataFrame como uma unica serie.
            Padrao: None.

    Returns:
        pd.DataFrame: Copia do df_test com os NaN iniciais preenchidos.
            Nenhuma linha e removida.

    Raises:
        ValueError: Se value_col nao existir em df_train ou df_test.
        ValueError: Se o treino de alguma serie tiver menos de window_size obs.
    """
    _validate_column_exists(df_train, value_col, "value_col (df_train)")
    _validate_column_exists(df_test, value_col, "value_col (df_test)")

    w = window_size
    min_col = f"min_{w}"
    mean_col = f"mean_{w}"
    max_col = f"max_{w}"

    result = df_test.copy()

    if series_col is not None and series_col in df_train.columns:
        series_list = df_train[series_col].unique().tolist()
    else:
        series_list = [None]

    for serie in series_list:
        if serie is not None:
            train_mask = df_train[series_col] == serie
            test_mask = result[series_col] == serie
            train_values = df_train.loc[train_mask, value_col].values
            test_values = result.loc[test_mask, value_col].values
            test_idx = result.index[test_mask].tolist()
        else:
            train_values = df_train[value_col].values
            test_values = result[value_col].values
            test_idx = result.index.tolist()

        if len(train_values) < w:
            raise ValueError(
                f"Serie '{serie}': treino tem {len(train_values)} obs, "
                f"mas window_size={w}. O treino precisa ter ao menos "
                f"window_size observacoes para preencher o contexto do teste."
            )

        # Concatena os ultimos W valores do treino com o inicio do teste
        # para reconstruir as janelas corretas das primeiras linhas do teste
        context = list(train_values[-w:]) + list(test_values)

        # Preenche as primeiras W linhas de cada serie onde ha NaN
        n_nan_rows = min(w, len(test_idx))
        for i in range(n_nan_rows):
            idx = test_idx[i]

            # Verifica se essa linha realmente tem NaN a preencher
            if pd.isna(result.at[idx, min_col]):
                # Janela para a observacao i e context[i : i+w]
                window_vals = context[i: i + w]
                result.at[idx, min_col] = float(min(window_vals))
                result.at[idx, mean_col] = float(
                    sum(window_vals) / len(window_vals)
                )
                result.at[idx, max_col] = float(max(window_vals))

    return result


def build_output_filename(input_path: str, window_size: int) -> str:
    """
    Constroi o nome do arquivo de saida com sufixo indicando o tamanho
    da janela aplicada.

    Padrao adotado: <stem>_interval_<W>.csv

    Parametros:
        input_path (str): Caminho do arquivo de entrada.
        window_size (int): Tamanho da janela aplicada.

    Returns:
        str: Nome do arquivo de saida (apenas o nome, sem diretorio).

    Exemplos:
        >>> build_output_filename('saugeenday_dataset_train.csv', 7)
        'saugeenday_dataset_train_interval_7.csv'
    """
    stem = Path(input_path).stem
    return f"{stem}_interval_{window_size}.csv"


# ---------------------------------------------------------------------------
# Validadores internos
# ---------------------------------------------------------------------------
def _validate_window_size(window_size: int) -> None:
    """
    Garante que window_size e um inteiro >= 2.

    Parametros:
        window_size (int): Valor a validar.

    Raises:
        TypeError: Se nao for inteiro.
        ValueError: Se for menor que 2.
    """
    if not isinstance(window_size, int):
        raise TypeError(
            f"'window_size' deve ser um inteiro, mas recebeu "
            f"{type(window_size).__name__}."
        )
    if window_size < 2:
        raise ValueError(
            f"'window_size' deve ser >= 2, mas recebeu {window_size}."
        )


def _validate_column_exists(
    df: pd.DataFrame,
    col: str,
    param_name: str,
) -> None:
    if col not in df.columns:
        raise ValueError(
            f"Parametro '{param_name}': coluna '{col}' nao encontrada. "
            f"Colunas disponiveis: {list(df.columns)}"
        )


def _ensure_output_dir(output_dir: str) -> str:
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    return abs_dir


def _detect_columns(
    df: pd.DataFrame,
    value_col: Optional[str],
    date_col: Optional[str],
) -> tuple[str, Optional[str]]:
    """
    Detecta automaticamente as colunas de valor e data quando nao informadas.

    Parametros:
        df (pd.DataFrame): DataFrame a inspecionar.
        value_col (str | None): Coluna de valor informada pelo usuario.
        date_col (str | None): Coluna de data informada pelo usuario.

    Returns:
        tuple[str, Optional[str]]: (value_col resolvida, date_col resolvida).

    Raises:
        ValueError: Se value_col nao puder ser determinada.
    """
    if value_col is None:
        if DEFAULT_VALUE_COL in df.columns:
            value_col = DEFAULT_VALUE_COL
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise ValueError(
                    "Nao foi possivel detectar automaticamente a coluna de "
                    "valor. Informe 'value_col' explicitamente."
                )
            value_col = numeric_cols[0]
            logger.warning(
                "value_col nao informada. Usando a primeira coluna "
                "numerica detectada: '%s'.",
                value_col,
            )

    if date_col is None and DEFAULT_DATE_COL in df.columns:
        date_col = DEFAULT_DATE_COL

    return value_col, date_col


# ---------------------------------------------------------------------------
# Implementacao concreta
# ---------------------------------------------------------------------------
class IntervalCreator(IntervalTransformer):
    """
    Gera features de aritmetica intervalar (min, mean, max) por janela
    deslizante a partir dos CSVs produzidos pelo splitter.py.

    Para series de treino: dropna=True — remove as window_size primeiras
    linhas de cada serie (sem contexto suficiente para preencher a janela).

    Para series de teste: dropna=False + preenchimento com contexto do treino
    — garante que todos os pontos de avaliacao sejam preservados e que as
    primeiras janelas do teste sejam calculadas corretamente usando o final
    do treino.

    Parametros:
        window_size (int): Tamanho da janela deslizante (>= 2).
        value_col (str | None): Nome da coluna de valores. Quando None,
            detecta automaticamente. Padrao: None.
        date_col (str | None): Nome da coluna de data. Quando None,
            detecta automaticamente. Padrao: None.
        series_col (str): Nome da coluna de identificacao das series.
            Padrao: 'series_name'.

    Exemplos:
        >>> creator = IntervalCreator(window_size=7)
        >>> result = creator.process_file(
        ...     filepath='../data/splits/univariate/train/saugeenday_train.csv',
        ...     output_dir='../data/processed/univariate/train/',
        ... )
        >>> result.print_summary()
    """

    def __init__(
        self,
        window_size: int,
        value_col: Optional[str] = None,
        date_col: Optional[str] = None,
        series_col: str = DEFAULT_SERIES_COL,
    ) -> None:
        super().__init__(window_size=window_size, dropna=True)
        self._value_col_override = value_col
        self._date_col_override = date_col
        self.series_col = series_col

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: Optional[str] = None,
        series_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Aplica a transformacao intervalar em um DataFrame completo.

        Parametros:
            df (pd.DataFrame): DataFrame de entrada.
            value_col (str): Nome da coluna de valores.
            date_col (str | None): Coluna de data (preservada intacta).
            series_col (str | None): Coluna de agrupamento por serie.

        Returns:
            pd.DataFrame: DataFrame com colunas min_W, mean_W, max_W
                adicionadas. Linhas com NaN removidas (dropna=True).
        """
        return generate_intervals(
            df=df,
            value_col=value_col,
            window_size=self.window_size,
            date_col=date_col,
            dropna=self.dropna,
            group_col=series_col,
        )

    def process_file(
        self,
        filepath: str,
        output_dir: str,
        value_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> IntervalResult:
        """
        Carrega um CSV de treino, gera as features intervalares e salva.

        Fluxo interno:
            1. Valida existencia do arquivo.
            2. Carrega o CSV.
            3. Detecta colunas de valor e data.
            4. Aplica generate_intervals(dropna=True) — remove primeiras linhas.
            5. Salva o CSV resultante.
            6. Retorna IntervalResult com metadados.

        Parametros:
            filepath (str): Caminho do CSV de entrada.
            output_dir (str): Diretorio de saida para o CSV gerado.
            value_col (str | None): Coluna de valores (opcional).
            date_col (str | None): Coluna de data (opcional).

        Returns:
            IntervalResult: Objeto com metadados do processamento.

        Raises:
            FileNotFoundError: Se filepath nao existir.
        """
        abs_path = os.path.abspath(filepath)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(
                f"Arquivo nao encontrado: '{abs_path}'."
            )

        logger.info(
            "Processando treino: %s (window_size=%d)",
            os.path.basename(abs_path),
            self.window_size,
        )

        df = pd.read_csv(abs_path)
        shape_before = df.shape

        eff_value_col = value_col or self._value_col_override
        eff_date_col = date_col or self._date_col_override
        eff_value_col, eff_date_col = _detect_columns(
            df, eff_value_col, eff_date_col
        )

        n_series = (
            df[self.series_col].nunique()
            if self.series_col in df.columns
            else 1
        )

        df_result = generate_intervals(
            df=df,
            value_col=eff_value_col,
            window_size=self.window_size,
            date_col=eff_date_col,
            dropna=True,
            group_col=(
                self.series_col if self.series_col in df.columns else None
            ),
        )

        abs_output_dir = _ensure_output_dir(output_dir)
        out_filename = build_output_filename(abs_path, self.window_size)
        out_path = os.path.join(abs_output_dir, out_filename)
        df_result.to_csv(out_path, index=False, encoding="utf-8")

        result = IntervalResult(
            input_name=Path(abs_path).stem,
            output_path=out_path,
            window_size=self.window_size,
            n_series=n_series,
            shape_before=shape_before,
            shape_after=df_result.shape,
            dropna=True,
        )

        logger.info(
            "Salvo: %s | %dx%d -> %dx%d | removidas: %d",
            out_filename,
            *shape_before,
            *df_result.shape,
            result.rows_removed,
        )
        return result

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        value_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> BatchIntervalReport:
        """
        Processa em lote todos os CSVs de uma pasta (apenas treino).

        Parametros:
            input_dir (str): Pasta com os CSVs de entrada.
            output_dir (str): Pasta de saida.
            value_col (str | None): Coluna de valores (opcional).
            date_col (str | None): Coluna de data (opcional).

        Returns:
            BatchIntervalReport: Relatorio consolidado do lote.

        Raises:
            FileNotFoundError: Se input_dir nao existir.
        """
        abs_input = os.path.abspath(input_dir)
        if not os.path.isdir(abs_input):
            raise FileNotFoundError(
                f"Pasta nao encontrada: '{abs_input}'."
            )

        report = BatchIntervalReport(window_size=self.window_size)
        all_files = sorted(os.listdir(abs_input))

        # Carrega DATASETS_ATIVOS do config para filtrar arquivos
        datasets_ativos = None
        try:
            from config import DATASETS_ATIVOS as _da
            datasets_ativos = _da
        except ImportError:
            pass

        for fname in all_files:
            fpath = os.path.join(abs_input, fname)

            if not fname.lower().endswith(".csv"):
                report.skipped.append(fname)
                continue

            # Verifica DATASETS_ATIVOS — compara pelo nome base do dataset
            if datasets_ativos is not None:
                dataset_tsf = fname.replace("_train.csv", ".tsf") \
                                   .replace("_test.csv", ".tsf")
                if not datasets_ativos.get(dataset_tsf, True):
                    report.skipped.append(fname)
                    logger.info(
                        "Dataset ignorado (DATASETS_ATIVOS=False): %s", fname
                    )
                    continue

            try:
                result = self.process_file(
                    filepath=fpath,
                    output_dir=output_dir,
                    value_col=value_col,
                    date_col=date_col,
                )
                report.processed.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha em '%s': %s", fname, exc)
                report.failed.append((fname, str(exc)))

        return report

    def process_split_folder(
        self,
        splits_dir: str,
        output_dir: str,
        value_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> BatchIntervalReport:
        """
        Processa em lote as pastas train/ e test/ produzidas pelo splitter.

        O treino e processado com dropna=True. O teste e processado com
        dropna=False e preenchimento do contexto com os ultimos valores
        do treino correspondente.

        Estrutura de entrada esperada:
            <splits_dir>/train/<dataset>_train.csv
            <splits_dir>/test/<dataset>_test.csv

        Estrutura de saida gerada:
            <output_dir>/train/<dataset>_train_interval_W.csv
            <output_dir>/test/<dataset>_test_interval_W.csv

        Correspondencia treino-teste:
            O modulo emparelha automaticamente cada arquivo de teste com
            seu treino correspondente pelo nome base do dataset. Por exemplo,
            'saugeenday_dataset_test.csv' sera emparelhado com
            'saugeenday_dataset_train.csv'.

        Parametros:
            splits_dir (str): Diretorio raiz com subpastas train/ e test/.
            output_dir (str): Diretorio raiz de saida.
            value_col (str | None): Coluna de valores (opcional).
            date_col (str | None): Coluna de data (opcional).

        Returns:
            BatchIntervalReport: Relatorio consolidado de ambas as particoes.

        Raises:
            FileNotFoundError: Se splits_dir nao existir.
        """
        abs_splits = os.path.abspath(splits_dir)
        abs_output = os.path.abspath(output_dir)

        if not os.path.isdir(abs_splits):
            raise FileNotFoundError(
                f"Pasta nao encontrada: '{abs_splits}'."
            )

        combined = BatchIntervalReport(window_size=self.window_size)

        # ── Passo 1: processa o treino com dropna=True ────────────────────
        train_input = os.path.join(abs_splits, "train")
        train_output = os.path.join(abs_output, "train")

        if not os.path.isdir(train_input):
            logger.warning(
                "Subpasta 'train/' nao encontrada em '%s'.", abs_splits
            )
        else:
            logger.info("Processando particao: train/")
            train_report = self.process_folder(
                input_dir=train_input,
                output_dir=train_output,
                value_col=value_col,
                date_col=date_col,
            )
            combined.processed.extend(train_report.processed)
            combined.failed.extend(train_report.failed)
            combined.skipped.extend(train_report.skipped)

        # ── Passo 2: processa o teste com contexto do treino ──────────────
        test_input = os.path.join(abs_splits, "test")
        test_output = os.path.join(abs_output, "test")

        if not os.path.isdir(test_input):
            logger.warning(
                "Subpasta 'test/' nao encontrada em '%s'.", abs_splits
            )
            return combined

        logger.info("Processando particao: test/")
        test_files = sorted(
            f for f in os.listdir(test_input)
            if f.lower().endswith(".csv")
        )

        for fname in test_files:
            test_path = os.path.join(test_input, fname)
            train_fname = fname.replace("_test.csv", "_train.csv")
            train_path = os.path.join(train_input, train_fname)

            try:
                self._process_test_file_with_context(
                    test_path=test_path,
                    train_path=train_path,
                    output_dir=test_output,
                    value_col=value_col,
                    date_col=date_col,
                    report=combined,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha em '%s': %s", fname, exc)
                combined.failed.append((fname, str(exc)))

        return combined

    def _process_test_file_with_context(
        self,
        test_path: str,
        train_path: str,
        output_dir: str,
        value_col: Optional[str],
        date_col: Optional[str],
        report: BatchIntervalReport,
    ) -> None:
        """
        Processa um unico arquivo de teste preenchendo os NaN iniciais
        com o contexto do treino correspondente.

        Parametros:
            test_path (str): Caminho absoluto do CSV de teste.
            train_path (str): Caminho absoluto do CSV de treino correspondente.
            output_dir (str): Diretorio de saida.
            value_col (str | None): Coluna de valores.
            date_col (str | None): Coluna de data.
            report (BatchIntervalReport): Objeto de relatorio a atualizar.
        """
        fname = os.path.basename(test_path)

        if not os.path.isfile(test_path):
            raise FileNotFoundError(
                f"Arquivo de teste nao encontrado: '{test_path}'."
            )
        if not os.path.isfile(train_path):
            raise FileNotFoundError(
                f"Arquivo de treino correspondente nao encontrado: "
                f"'{train_path}'. Verifique se o splitter.py foi executado."
            )

        logger.info(
            "Processando teste: %s (window_size=%d)",
            fname,
            self.window_size,
        )

        df_test = pd.read_csv(test_path)
        shape_before = df_test.shape

        eff_value_col = value_col or self._value_col_override
        eff_date_col = date_col or self._date_col_override
        eff_value_col, eff_date_col = _detect_columns(
            df_test, eff_value_col, eff_date_col
        )

        n_series = (
            df_test[self.series_col].nunique()
            if self.series_col in df_test.columns
            else 1
        )

        # Gera intervalos sem remover NaN — preserva todos os pontos
        df_intervaled = generate_intervals(
            df=df_test,
            value_col=eff_value_col,
            window_size=self.window_size,
            date_col=eff_date_col,
            dropna=False,
            group_col=(
                self.series_col if self.series_col in df_test.columns
                else None
            ),
        )

        # Preenche os NaN iniciais com o contexto do treino
        df_train = pd.read_csv(train_path)
        df_filled = fill_test_intervals_from_train(
            df_test=df_intervaled,
            df_train=df_train,
            value_col=eff_value_col,
            window_size=self.window_size,
            series_col=(
                self.series_col if self.series_col in df_test.columns
                else None
            ),
        )

        abs_output_dir = _ensure_output_dir(output_dir)
        out_filename = build_output_filename(test_path, self.window_size)
        out_path = os.path.join(abs_output_dir, out_filename)
        df_filled.to_csv(out_path, index=False, encoding="utf-8")

        result = IntervalResult(
            input_name=Path(test_path).stem,
            output_path=out_path,
            window_size=self.window_size,
            n_series=n_series,
            shape_before=shape_before,
            shape_after=df_filled.shape,
            dropna=False,
        )

        logger.info(
            "Salvo: %s | %dx%d -> %dx%d | NaN preenchidos com contexto",
            out_filename,
            *shape_before,
            *df_filled.shape,
        )
        report.processed.append(result)

    def __repr__(self) -> str:
        return (
            f"IntervalCreator("
            f"window_size={self.window_size}, "
            f"value_col='{self._value_col_override or 'auto'}', "
            f"series_col='{self.series_col}')"
        )