"""
lag_creator.py

Modulo responsavel pela geracao de variaveis defasadas (lags) a partir dos
CSVs de treino e teste produzidos pelo modulo splitter.py.

Contexto de uso no pipeline do projeto:
    Os lags sao aplicados APOS o split temporal, de forma independente sobre
    os conjuntos de treino e teste, eliminando qualquer risco de data leakage.

Fluxo completo do projeto:
    raw/ (.tsf)
        └─► MonashDataLoader              [data_loader.py]
                └─► TimeSeriesSplitter    [splitter.py]
                        ├─► splits/train/<dataset>_train.csv
                        └─► splits/test/<dataset>_test.csv
                                └─► LagCreator  [lag_creator.py]  <- este modulo
                                        ├─► processed/train/<dataset>_train_lags_N.csv
                                        └─► processed/test/<dataset>_test_lags_N.csv

Entrada esperada:
    CSVs gerados pelo splitter.py, localizados em:
        data/splits/univariate/train/
        data/splits/univariate/test/

Saida gerada:
    CSVs com colunas de lag adicionadas, salvos em:
        data/processed/univariate/train/
        data/processed/univariate/test/

    Convencao de nome: <stem>_lags_<N>.csv
    Exemplo: saugeenday_dataset_train_lags_5.csv

Uso basico - arquivo unico:
    from src.features.lag_creator import LagCreator

    creator = LagCreator(n_lags=5)
    result = creator.process_file(
        filepath='../data/splits/univariate/train/saugeenday_dataset_train.csv',
        output_dir='../data/processed/univariate/train/',
    )
    result.print_summary()

Uso basico - lote completo (treino e teste):
    from src.features.lag_creator import LagCreator

    creator = LagCreator(n_lags=5)
    report = creator.process_split_folder(
        splits_dir='../data/splits/univariate/',
        output_dir='../data/processed/univariate/',
    )
    report.print_summary()

Tratamento correto dos lags no conjunto de teste:
    O conjunto de teste NAO usa dropna. Em vez disso, os n_lags primeiros
    NaN de cada serie sao preenchidos com os ultimos n_lags valores do
    treino correspondente. Isso garante que:
        1. Todos os pontos de avaliacao sejam preservados (sem perda de obs).
        2. Os lags do inicio do teste sejam contextualizados corretamente
           com o final do treino, exatamente como ocorreria em producao.

    Exemplo com n_lags=3 e horizonte=5:
        Treino (ultimas obs): [..., 98, 99, 100]
        Teste  (todas  obs):  [101, 102, 103, 104, 105]

        lag_1 da obs 101 = 100  (ultimo valor do treino)
        lag_2 da obs 101 = 99
        lag_3 da obs 101 = 98
        -> Nenhuma linha removida do teste.

Arquitetura e escalabilidade:
    - LagTransformer (ABC): contrato generico de transformacao. Permite criar
      MultivariateLagCreator sem alterar o codigo existente.
    - LagCreator: implementacao concreta para series univariadas.
    - generate_lags(): funcao pura reutilizavel em notebooks e pipelines.
    - fill_test_lags_from_train(): funcao pura que preenche os NaN iniciais
      do teste com o contexto do treino, sem dependencia de classes.
    - LagResult / BatchLagReport: dataclasses de resultado com o mesmo
      padrao do splitter.py, facilitando integracao entre modulos.

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

# Garante que src/ esta no path para importar data_loader
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
LAG_COL_PREFIX: str = "lag_"


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------
@dataclass
class LagResult:
    """
    Encapsula os metadados e estatisticas de um arquivo processado com lags.

    Atributos:
        input_name (str): Nome base do arquivo de entrada (sem extensao).
        output_path (str): Caminho absoluto do CSV gerado.
        n_lags (int): Numero de lags aplicados.
        n_series (int): Numero de series presentes no arquivo.
        shape_before (tuple[int, int]): Shape do DataFrame antes dos lags.
        shape_after (tuple[int, int]): Shape do DataFrame apos os lags.
        dropna (bool): Indica se linhas com NaN foram removidas.
    """

    input_name: str
    output_path: str
    n_lags: int
    n_series: int
    shape_before: tuple[int, int]
    shape_after: tuple[int, int]
    dropna: bool

    @property
    def rows_removed(self) -> int:
        """Numero de linhas removidas pelo dropna (por serie)."""
        if not self.dropna:
            return 0
        return self.shape_before[0] - self.shape_after[0]

    @property
    def cols_added(self) -> int:
        """Numero de colunas de lag adicionadas."""
        return self.shape_after[1] - self.shape_before[1]

    def print_summary(self) -> None:
        """
        Exibe um resumo detalhado do arquivo processado, incluindo shape
        antes e apos a transformacao, linhas removidas e colunas geradas.
        """
        sep = "-" * 68
        print(f"\n{sep}")
        print(f"  LAGS: {self.input_name}")
        print(sep)
        print(f"  Lags gerados     : {self.n_lags}")
        print(f"  Num. de series   : {self.n_series}")
        print(f"  Shape antes      : {self.shape_before[0]} linhas x "
              f"{self.shape_before[1]} colunas")
        print(f"  Shape apos       : {self.shape_after[0]} linhas x "
              f"{self.shape_after[1]} colunas")
        print(f"  Colunas geradas  : {self.cols_added}  "
              f"(lag_1 ... lag_{self.n_lags})")
        if self.dropna:
            print(
                f"  Linhas removidas : {self.rows_removed}  "
                f"(dropna=True, {self.n_lags} obs x {self.n_series} series)"
            )
        else:
            print(f"  Linhas removidas : 0  (dropna=False, NaN mantidos)")
        print(f"\n  Saida -> {self.output_path}")
        print(f"{sep}\n")


@dataclass
class BatchLagReport:
    """
    Consolida os resultados do processamento em lote de uma pasta.

    Mantem o mesmo padrao do BatchSplitReport do splitter.py para
    consistencia de interface no projeto.

    Atributos:
        n_lags (int): Numero de lags aplicados em todos os arquivos.
        processed (list[LagResult]): Resultados dos arquivos processados.
        failed (list[tuple[str, str]]): Pares (arquivo, mensagem de erro).
        skipped (list[str]): Arquivos ignorados (extensao diferente de .csv).
    """

    n_lags: int
    processed: list[LagResult] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def n_processed(self) -> int:
        """Numero de arquivos processados com sucesso."""
        return len(self.processed)

    @property
    def n_failed(self) -> int:
        """Numero de arquivos que falharam."""
        return len(self.failed)

    def print_summary(self) -> None:
        """
        Exibe o relatorio consolidado do lote, com uma linha por arquivo
        mostrando shape antes/apos e colunas geradas.
        """
        sep = "=" * 68
        print(f"\n{sep}")
        print("  RELATORIO DE GERACAO DE LAGS EM LOTE")
        print(sep)
        print(f"  Lags aplicados : {self.n_lags}")
        print(f"  Processados    : {self.n_processed}")
        print(f"  Falhas         : {self.n_failed}")
        print(f"  Ignorados      : {len(self.skipped)}")

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
# Contrato abstrato - ponto de extensao para dados multivariados
# ---------------------------------------------------------------------------
class LagTransformer(ABC):
    """
    Classe-base abstrata que define o contrato de um transformador de lags.

    Qualquer transformador de lags - univariado ou multivariado - deve
    implementar os metodos `transform_dataframe` e `process_file`.

    Parametros:
        n_lags (int): Numero de defasagens a gerar. Deve ser >= 1.
        dropna (bool): Se True, remove as n_lags primeiras linhas que
            conterao NaN apos o shift. Padrao: True.

    Exemplo de extensao futura:
        class MultivariateLagCreator(LagTransformer):
            def transform_dataframe(self, df, value_cols, ...): ...
            def process_file(self, filepath, ...): ...
    """

    def __init__(self, n_lags: int, dropna: bool = True) -> None:
        _validate_n_lags(n_lags)
        self.n_lags = n_lags
        self.dropna = dropna

    @abstractmethod
    def transform_dataframe(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: Optional[str],
        series_col: Optional[str],
    ) -> pd.DataFrame:
        """
        Aplica a transformacao de lags em um DataFrame completo.

        Parametros:
            df (pd.DataFrame): DataFrame de entrada.
            value_col (str): Nome da coluna de valores.
            date_col (str | None): Coluna de data (preservada, nao defasada).
            series_col (str | None): Coluna de agrupamento por serie.

        Returns:
            pd.DataFrame: DataFrame com colunas de lag adicionadas.
        """

    @abstractmethod
    def process_file(
        self,
        filepath: str,
        output_dir: str,
        value_col: Optional[str],
        date_col: Optional[str],
    ) -> LagResult:
        """
        Processa um arquivo CSV e salva o resultado com lags.

        Parametros:
            filepath (str): Caminho do arquivo CSV de entrada.
            output_dir (str): Diretorio de saida.
            value_col (str | None): Nome da coluna de valores.
            date_col (str | None): Nome da coluna de data.

        Returns:
            LagResult: Objeto com metadados e estatisticas do processamento.
        """


# ---------------------------------------------------------------------------
# Funcoes utilitarias puras
# ---------------------------------------------------------------------------
def generate_lags(
    df: pd.DataFrame,
    value_col: str,
    n_lags: int,
    date_col: Optional[str] = None,
    dropna: bool = True,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Gera N colunas de lag para a coluna value_col de um DataFrame.

    Funcao pura e stateless - nao depende de nenhuma classe e pode ser
    usada diretamente em notebooks ou pipelines sklearn. Suporta tanto
    DataFrames de serie unica quanto DataFrames com multiplas series
    (via group_col), sem vazamento de dados entre grupos.

    Logica:
        Para cada k em [1, n_lags], cria a coluna lag_k como
        df[value_col].shift(k) dentro de cada grupo (se group_col
        for informado). Colunas de data sao preservadas intactas.

    Parametros:
        df (pd.DataFrame): DataFrame de entrada. Deve conter value_col.
        value_col (str): Nome da coluna com os valores da serie.
        n_lags (int): Numero de defasagens a gerar (>= 1).
        date_col (str | None): Coluna de data - nao sera defasada.
            Padrao: None.
        dropna (bool): Remove as primeiras n_lags linhas com NaN gerados
            pelo shift. Padrao: True.
        group_col (str | None): Coluna de agrupamento (ex: 'series_name').
            Quando informado, o shift e aplicado dentro de cada grupo,
            evitando contaminacao entre series distintas. Padrao: None.

    Returns:
        pd.DataFrame: Copia do DataFrame com colunas lag_1 ... lag_N
            adicionadas a direita.

    Raises:
        ValueError: Se value_col nao existe no DataFrame.
        ValueError: Se n_lags < 1.

    Exemplos:
        >>> df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        >>> generate_lags(df, value_col='value', n_lags=2)
           value  lag_1  lag_2
        2     30   20.0   10.0
        3     40   30.0   20.0
        4     50   40.0   30.0
    """
    _validate_n_lags(n_lags)
    _validate_column_exists(df, value_col, "value_col")

    if date_col is not None:
        _validate_column_exists(df, date_col, "date_col")

    result = df.copy()

    for k in range(1, n_lags + 1):
        lag_col_name = f"{LAG_COL_PREFIX}{k}"

        if group_col is not None:
            result[lag_col_name] = (
                result.groupby(group_col)[value_col].shift(k)
            )
        else:
            result[lag_col_name] = result[value_col].shift(k)

    if dropna:
        result = result.dropna().reset_index(drop=True)

    return result


def fill_test_lags_from_train(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    value_col: str,
    n_lags: int,
    series_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Preenche os NaN iniciais das colunas de lag do teste com os ultimos
    valores do treino correspondente, por serie.

    Contexto:
        Apos aplicar generate_lags(dropna=False) no teste, as primeiras
        n_lags linhas de cada serie ficam com NaN nas colunas de lag porque
        nao ha observacoes anteriores dentro do proprio teste. Esses NaN
        representam exatamente o contexto que vem do treino.

        Esta funcao preenche esses NaN usando as ultimas n_lags observacoes
        do valor original de cada serie no treino, de forma que:
            lag_1[0] = ultimo valor do treino da mesma serie
            lag_2[0] = penultimo valor do treino da mesma serie
            ...
            lag_k[0] = k-esimo valor a partir do final do treino

        O resultado e um teste completo, sem NaN e sem perda de nenhuma
        observacao de avaliacao.

    Parametros:
        df_test (pd.DataFrame): DataFrame do teste com colunas lag_1...lag_N
            ja geradas (com NaN nas primeiras linhas de cada serie).
        df_train (pd.DataFrame): DataFrame do treino com a coluna value_col
            original (sem necessidade de ter colunas de lag).
        value_col (str): Nome da coluna de valores em ambos os DataFrames.
        n_lags (int): Numero de lags gerados (determina quantas linhas
            de cada serie precisam ser preenchidas).
        series_col (str | None): Coluna de identificacao das series.
            Quando None, trata todo o DataFrame como uma unica serie.
            Padrao: None.

    Returns:
        pd.DataFrame: Copia do df_test com os NaN iniciais preenchidos
            pelo contexto do treino. Nenhuma linha e removida.

    Raises:
        ValueError: Se value_col nao existir em df_train ou df_test.
        ValueError: Se o treino de alguma serie tiver menos de n_lags obs.

    Exemplos:
        >>> df_train, df_test = split_dataframe(df, horizon=5)
        >>> df_test_lagged  = generate_lags(df_test, 'value', n_lags=3,
        ...                                 dropna=False)
        >>> df_test_filled  = fill_test_lags_from_train(
        ...     df_test_lagged, df_train, 'value', n_lags=3)
        >>> assert df_test_filled.isna().sum().sum() == 0
    """
    _validate_column_exists(df_train, value_col, "value_col (df_train)")
    _validate_column_exists(df_test, value_col, "value_col (df_test)")

    result = df_test.copy()

    # Determina os grupos de series a processar
    if series_col is not None and series_col in df_train.columns:
        series_list = df_train[series_col].unique().tolist()
    else:
        series_list = [None]

    for serie in series_list:
        # Filtra treino e teste pela serie atual
        if serie is not None:
            train_mask = df_train[series_col] == serie
            test_mask = result[series_col] == serie
            train_values = (
                df_train.loc[train_mask, value_col].values
            )
        else:
            train_values = df_train[value_col].values
            test_mask = slice(None)

        if len(train_values) < n_lags:
            raise ValueError(
                f"Serie '{serie}': treino tem {len(train_values)} obs, "
                f"mas n_lags={n_lags}. O treino precisa ter ao menos "
                f"n_lags observacoes para preencher o contexto do teste."
            )

        # Extrai os ultimos n_lags valores do treino
        # train_context[0] = ultimo valor, train_context[1] = penultimo, ...
        train_context = train_values[-n_lags:][::-1]

        # Preenche lag_k com o k-esimo valor a partir do final do treino
        # apenas nas linhas onde o lag ainda e NaN (primeiras n_lags linhas)
        for k in range(1, n_lags + 1):
            lag_col = f"{LAG_COL_PREFIX}{k}"
            if lag_col not in result.columns:
                continue

            # Identifica linhas desta serie onde lag_k e NaN
            if serie is not None:
                nan_mask = test_mask & result[lag_col].isna()
            else:
                nan_mask = result[lag_col].isna()

            nan_idx = result.index[nan_mask].tolist()

            # Mapeamento correto:
            # A linha i do teste tem lag_k = NaN quando i < k.
            # O valor correto e train_context[k - 1 - i], onde:
            #   train_context[0] = ultimo valor do treino
            #   train_context[1] = penultimo valor do treino, etc.
            # Para a linha i=0 e lag_k=1: train_context[0] = ultimo treino
            # Para a linha i=0 e lag_k=2: train_context[1] = penultimo treino
            # Para a linha i=1 e lag_k=2: train_context[0] = ultimo treino
            for row_offset, idx in enumerate(nan_idx):
                context_idx = k - 1 - row_offset
                if 0 <= context_idx < len(train_context):
                    result.at[idx, lag_col] = float(
                        train_context[context_idx]
                    )

    return result


def build_output_filename(input_path: str, n_lags: int) -> str:
    """
    Constroi o nome do arquivo de saida com sufixo indicando o numero
    de lags aplicados.

    Padrao adotado: <stem>_lags_<N>.csv

    Parametros:
        input_path (str): Caminho do arquivo de entrada (qualquer extensao).
        n_lags (int): Numero de lags aplicados.

    Returns:
        str: Nome do arquivo de saida (apenas o nome, sem diretorio).

    Exemplos:
        >>> build_output_filename('saugeenday_dataset_train.csv', 5)
        'saugeenday_dataset_train_lags_5.csv'
    """
    stem = Path(input_path).stem
    return f"{stem}_lags_{n_lags}.csv"


# ---------------------------------------------------------------------------
# Validadores internos
# ---------------------------------------------------------------------------
def _validate_n_lags(n_lags: int) -> None:
    """
    Garante que n_lags e um inteiro positivo.

    Parametros:
        n_lags (int): Valor a validar.

    Raises:
        TypeError: Se nao for inteiro.
        ValueError: Se for menor que 1.
    """
    if not isinstance(n_lags, int):
        raise TypeError(
            f"'n_lags' deve ser um inteiro, mas recebeu "
            f"{type(n_lags).__name__}."
        )
    if n_lags < 1:
        raise ValueError(
            f"'n_lags' deve ser >= 1, mas recebeu {n_lags}."
        )


def _validate_column_exists(
    df: pd.DataFrame,
    col: str,
    param_name: str,
) -> None:
    """
    Verifica se uma coluna existe no DataFrame.

    Parametros:
        df (pd.DataFrame): DataFrame a inspecionar.
        col (str): Nome da coluna esperada.
        param_name (str): Nome do parametro (para mensagem de erro clara).

    Raises:
        ValueError: Se a coluna nao existir no DataFrame.
    """
    if col not in df.columns:
        raise ValueError(
            f"Parametro '{param_name}': coluna '{col}' nao encontrada. "
            f"Colunas disponiveis: {list(df.columns)}"
        )


def _ensure_output_dir(output_dir: str) -> str:
    """
    Cria o diretorio de saida se ele nao existir.

    Parametros:
        output_dir (str): Caminho do diretorio.

    Returns:
        str: Caminho absoluto do diretorio criado/confirmado.
    """
    abs_dir = os.path.abspath(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    return abs_dir


def _detect_columns(
    df: pd.DataFrame,
    value_col: Optional[str],
    date_col: Optional[str],
) -> tuple[str, Optional[str]]:
    """
    Detecta automaticamente as colunas de valor e data no DataFrame quando
    os parametros nao sao informados explicitamente.

    Logica de deteccao:
        - value_col: usa DEFAULT_VALUE_COL; se ausente, usa a ultima
          coluna numerica do DataFrame.
        - date_col: usa DEFAULT_DATE_COL; se ausente, procura por colunas
          com dtype datetime64.

    Parametros:
        df (pd.DataFrame): DataFrame a inspecionar.
        value_col (str | None): Nome explícito da coluna de valores.
        date_col (str | None): Nome explícito da coluna de data.

    Returns:
        tuple[str, str | None]: Par (value_col resolvido, date_col resolvido).

    Raises:
        ValueError: Se nenhuma coluna numerica for encontrada.
    """
    if value_col is not None:
        resolved_value = value_col
    elif DEFAULT_VALUE_COL in df.columns:
        resolved_value = DEFAULT_VALUE_COL
    else:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError(
                "Nenhuma coluna numerica encontrada. "
                "Informe 'value_col' explicitamente."
            )
        resolved_value = numeric_cols[-1]
        logger.warning(
            "value_col nao informado. Usando '%s' (ultima coluna numerica).",
            resolved_value,
        )

    if date_col is not None:
        resolved_date: Optional[str] = date_col
    elif DEFAULT_DATE_COL in df.columns:
        resolved_date = DEFAULT_DATE_COL
    else:
        datetime_cols = df.select_dtypes(
            include="datetime"
        ).columns.tolist()
        resolved_date = datetime_cols[0] if datetime_cols else None
        if resolved_date:
            logger.info(
                "date_col detectado automaticamente: '%s'.",
                resolved_date,
            )

    return resolved_value, resolved_date


# ---------------------------------------------------------------------------
# Implementacao concreta - dados univariados
# ---------------------------------------------------------------------------
class LagCreator(LagTransformer):
    """
    Transformador concreto de lags para CSVs produzidos pelo splitter.py.

    Recebe os arquivos CSV de treino e teste gerados pelo TimeSeriesSplitter
    e aplica as defasagens de forma independente em cada particao, sem
    qualquer cruzamento de informacao entre elas.

    Cada serie dentro do CSV e processada de forma independente - os lags
    nao vazam entre series distintas.

    Parametros:
        n_lags (int): Numero de defasagens. Deve ser >= 1.
        dropna (bool): Remove linhas com NaN gerados pelo shift.
            Padrao: True.
        value_col (str | None): Nome da coluna de valores. Quando None,
            e detectado automaticamente. Padrao: None.
        date_col (str | None): Nome da coluna de data. Quando None, e
            detectado automaticamente. Padrao: None.
        series_col (str): Nome da coluna de identificacao das series.
            Padrao: 'series_name'.

    Exemplos:
        >>> creator = LagCreator(n_lags=5)
        >>> result = creator.process_file(
        ...     filepath='../data/splits/univariate/train/saugeenday_dataset_train.csv',
        ...     output_dir='../data/processed/univariate/train/',
        ... )
        >>> result.print_summary()
    """

    def __init__(
        self,
        n_lags: int,
        dropna: bool = True,
        value_col: Optional[str] = None,
        date_col: Optional[str] = None,
        series_col: str = DEFAULT_SERIES_COL,
    ) -> None:
        super().__init__(n_lags=n_lags, dropna=dropna)
        self._value_col_override = value_col
        self._date_col_override = date_col
        self.series_col = series_col

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: Optional[str],
        series_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Aplica generate_lags() em um DataFrame completo, respeitando grupos
        de series quando series_col estiver presente.

        Quando series_col e informado e existe no DataFrame, o shift e
        aplicado dentro de cada grupo separadamente, sem vazamento entre
        series. Quando ausente, o shift e aplicado sobre o DataFrame inteiro.

        Parametros:
            df (pd.DataFrame): DataFrame de entrada.
            value_col (str): Coluna de valores.
            date_col (str | None): Coluna de data (preservada, nao defasada).
            series_col (str | None): Coluna de agrupamento por serie.

        Returns:
            pd.DataFrame: DataFrame com colunas lag_1 ... lag_N adicionadas.
        """
        effective_group = (
            series_col
            if series_col is not None and series_col in df.columns
            else None
        )
        return generate_lags(
            df=df,
            value_col=value_col,
            n_lags=self.n_lags,
            date_col=date_col,
            dropna=self.dropna,
            group_col=effective_group,
        )

    def process_file(
        self,
        filepath: str,
        output_dir: str = "../data/processed/univariate/",
        value_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> LagResult:
        """
        Carrega um CSV de treino ou teste, gera os lags e salva o resultado.

        Fluxo interno:
            1. Valida existencia do arquivo CSV.
            2. Carrega o CSV com pandas.
            3. Detecta automaticamente as colunas (se nao informadas).
            4. Aplica transform_dataframe() com agrupamento por serie.
            5. Salva o CSV resultante com sufixo _lags_N.
            6. Retorna LagResult com shape antes/apos e estatisticas.

        Parametros:
            filepath (str): Caminho relativo ou absoluto do CSV de entrada.
                Deve ser um CSV gerado pelo splitter.py.
            output_dir (str): Diretorio onde o CSV com lags sera salvo.
                Padrao: '../data/processed/univariate/'.
            value_col (str | None): Sobrescreve o valor padrao da instancia.
            date_col (str | None): Sobrescreve o valor padrao da instancia.

        Returns:
            LagResult: Objeto com metadados e estatisticas do processamento.

        Raises:
            FileNotFoundError: Se filepath nao existir.
            ValueError: Se a coluna de valores nao for encontrada.
        """
        abs_input = os.path.abspath(filepath)
        if not os.path.isfile(abs_input):
            raise FileNotFoundError(
                f"Arquivo nao encontrado: '{abs_input}'. "
                "Verifique o caminho relativo informado."
            )

        logger.info(
            "Processando: %s (n_lags=%d)",
            os.path.basename(abs_input),
            self.n_lags,
        )

        df = pd.read_csv(abs_input)
        shape_before = df.shape

        eff_value_col = value_col or self._value_col_override
        eff_date_col = date_col or self._date_col_override
        eff_value_col, eff_date_col = _detect_columns(
            df, eff_value_col, eff_date_col
        )

        logger.info(
            "Colunas resolvidas | valor: '%s' | data: '%s'",
            eff_value_col,
            eff_date_col or "N/A",
        )

        n_series = (
            df[self.series_col].nunique()
            if self.series_col in df.columns
            else 1
        )

        df_result = self.transform_dataframe(
            df=df,
            value_col=eff_value_col,
            date_col=eff_date_col,
            series_col=self.series_col,
        )

        abs_output_dir = _ensure_output_dir(output_dir)
        out_filename = build_output_filename(abs_input, self.n_lags)
        out_path = os.path.join(abs_output_dir, out_filename)
        df_result.to_csv(out_path, index=False, encoding="utf-8")

        result = LagResult(
            input_name=Path(abs_input).stem,
            output_path=out_path,
            n_lags=self.n_lags,
            n_series=n_series,
            shape_before=shape_before,
            shape_after=df_result.shape,
            dropna=self.dropna,
        )

        logger.info(
            "Salvo: %s | %dx%d -> %dx%d",
            out_filename,
            *shape_before,
            *df_result.shape,
        )
        return result

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
        value_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> BatchLagReport:
        """
        Processa em lote todos os CSVs de uma pasta (treino ou teste).

        Arquivos com extensao diferente de .csv sao silenciosamente
        ignorados. Falhas em arquivos individuais sao registradas no
        relatorio sem interromper os demais.

        Parametros:
            input_dir (str): Pasta com os CSVs de entrada.
            output_dir (str): Pasta de saida para os CSVs com lags.
            value_col (str | None): Coluna de valores (opcional).
            date_col (str | None): Coluna de data (opcional).

        Returns:
            BatchLagReport: Relatorio consolidado do lote.

        Raises:
            FileNotFoundError: Se input_dir nao existir.
        """
        abs_input = os.path.abspath(input_dir)
        if not os.path.isdir(abs_input):
            raise FileNotFoundError(
                f"Pasta nao encontrada: '{abs_input}'."
            )

        report = BatchLagReport(n_lags=self.n_lags)
        all_files = sorted(os.listdir(abs_input))

        # Carrega DATASETS_ATIVOS do config para filtrar arquivos
        datasets_ativos = None
        try:
            from config import DATASETS_ATIVOS as _da
            datasets_ativos = _da
        except ImportError:
            pass

        logger.info(
            "Iniciando lote | pasta: %s | arquivos: %d | n_lags: %d",
            abs_input,
            len(all_files),
            self.n_lags,
        )

        for fname in all_files:
            fpath = os.path.join(abs_input, fname)

            if not fname.lower().endswith(".csv"):
                report.skipped.append(fname)
                logger.debug("Ignorado (nao e .csv): %s", fname)
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

        logger.info(
            "Lote concluido | processados: %d | falhas: %d | ignorados: %d",
            report.n_processed,
            report.n_failed,
            len(report.skipped),
        )
        return report

    def process_split_folder(
        self,
        splits_dir: str = "../data/splits/univariate/",
        output_dir: str = "../data/processed/univariate/",
        value_col: Optional[str] = None,
        date_col: Optional[str] = None,
    ) -> BatchLagReport:
        """
        Processa em lote as pastas train/ e test/ geradas pelo splitter.py,
        com tratamento correto e independente de cada particao.

        Comportamento por particao:
            - train/: dropna=True — remove as n_lags primeiras linhas de
              cada serie (que nao possuem contexto historico).
            - test/ : dropna=False + fill_test_lags_from_train() — preserva
              todos os pontos de avaliacao e preenche os NaN iniciais com
              os ultimos n_lags valores do treino correspondente.

        Estrutura de entrada esperada:
            <splits_dir>/train/<dataset>_train.csv
            <splits_dir>/test/<dataset>_test.csv

        Estrutura de saida gerada:
            <output_dir>/train/<dataset>_train_lags_N.csv
            <output_dir>/test/<dataset>_test_lags_N.csv

        Correspondencia treino-teste:
            O modulo emparelha automaticamente cada arquivo de teste com
            seu treino correspondente pelo nome base do dataset. Por exemplo,
            'saugeenday_dataset_test.csv' sera emparelhado com
            'saugeenday_dataset_train.csv'.

        Parametros:
            splits_dir (str): Diretorio raiz com as subpastas train/ e test/.
                Padrao: '../data/splits/univariate/'.
            output_dir (str): Diretorio raiz de saida (train/ e test/ serao
                criados automaticamente). Padrao: '../data/processed/univariate/'.
            value_col (str | None): Coluna de valores (opcional).
            date_col (str | None): Coluna de data (opcional).

        Returns:
            BatchLagReport: Relatorio consolidado de ambas as particoes.

        Raises:
            FileNotFoundError: Se splits_dir nao existir.
        """
        abs_splits = os.path.abspath(splits_dir)
        abs_output = os.path.abspath(output_dir)

        if not os.path.isdir(abs_splits):
            raise FileNotFoundError(
                f"Pasta nao encontrada: '{abs_splits}'."
            )

        combined_report = BatchLagReport(n_lags=self.n_lags)

        # ── Passo 1: processa o treino normalmente (dropna=True) ──────────
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
            combined_report.processed.extend(train_report.processed)
            combined_report.failed.extend(train_report.failed)
            combined_report.skipped.extend(train_report.skipped)

        # ── Passo 2: processa o teste com preenchimento do contexto ───────
        test_input = os.path.join(abs_splits, "test")
        test_output = os.path.join(abs_output, "test")

        if not os.path.isdir(test_input):
            logger.warning(
                "Subpasta 'test/' nao encontrada em '%s'.", abs_splits
            )
            return combined_report

        logger.info("Processando particao: test/")
        test_files = sorted(
            f for f in os.listdir(test_input)
            if f.lower().endswith(".csv")
        )

        for fname in test_files:
            test_path = os.path.join(test_input, fname)

            # Localiza o arquivo de treino correspondente pelo nome base
            # Convencao: <dataset>_test.csv <-> <dataset>_train.csv
            train_fname = fname.replace("_test.csv", "_train.csv")
            train_path = os.path.join(train_input, train_fname)

            try:
                self._process_test_file_with_context(
                    test_path=test_path,
                    train_path=train_path,
                    output_dir=test_output,
                    value_col=value_col,
                    date_col=date_col,
                    report=combined_report,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Falha em '%s': %s", fname, exc)
                combined_report.failed.append((fname, str(exc)))

        return combined_report

    def _process_test_file_with_context(
        self,
        test_path: str,
        train_path: str,
        output_dir: str,
        value_col: Optional[str],
        date_col: Optional[str],
        report: BatchLagReport,
    ) -> None:
        """
        Processa um unico arquivo de teste preenchendo os NaN iniciais
        com o contexto do treino correspondente.

        Fluxo interno:
            1. Carrega o CSV de teste.
            2. Detecta colunas de valor e data.
            3. Aplica generate_lags(dropna=False) — mantem todos os pontos.
            4. Carrega o CSV de treino correspondente.
            5. Chama fill_test_lags_from_train() para preencher os NaN.
            6. Salva o CSV resultante e registra no relatorio.

        Parametros:
            test_path (str): Caminho absoluto do CSV de teste.
            train_path (str): Caminho absoluto do CSV de treino correspondente.
            output_dir (str): Diretorio de saida para o CSV processado.
            value_col (str | None): Coluna de valores.
            date_col (str | None): Coluna de data.
            report (BatchLagReport): Objeto de relatorio a ser atualizado.
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

        logger.info("Processando teste: %s (n_lags=%d)", fname, self.n_lags)

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

        # Gera lags sem remover NaN — preserva todos os pontos de avaliacao
        df_lagged = generate_lags(
            df=df_test,
            value_col=eff_value_col,
            n_lags=self.n_lags,
            date_col=eff_date_col,
            dropna=False,
            group_col=(
                self.series_col
                if self.series_col in df_test.columns
                else None
            ),
        )

        # Preenche os NaN iniciais com o contexto do treino
        df_train = pd.read_csv(train_path)
        df_filled = fill_test_lags_from_train(
            df_test=df_lagged,
            df_train=df_train,
            value_col=eff_value_col,
            n_lags=self.n_lags,
            series_col=(
                self.series_col
                if self.series_col in df_test.columns
                else None
            ),
        )

        abs_output_dir = _ensure_output_dir(output_dir)
        out_filename = build_output_filename(test_path, self.n_lags)
        out_path = os.path.join(abs_output_dir, out_filename)
        df_filled.to_csv(out_path, index=False, encoding="utf-8")

        result = LagResult(
            input_name=Path(test_path).stem,
            output_path=out_path,
            n_lags=self.n_lags,
            n_series=n_series,
            shape_before=shape_before,
            shape_after=df_filled.shape,
            dropna=False,
        )

        logger.info(
            "Salvo: %s | %dx%d -> %dx%d | NaN preenchidos com contexto do treino",
            out_filename,
            *shape_before,
            *df_filled.shape,
        )
        report.processed.append(result)

    def __repr__(self) -> str:
        return (
            f"LagCreator("
            f"n_lags={self.n_lags}, "
            f"dropna={self.dropna}, "
            f"value_col='{self._value_col_override or 'auto'}', "
            f"date_col='{self._date_col_override or 'auto'}', "
            f"series_col='{self.series_col}')"
        )