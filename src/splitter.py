"""
splitter.py

Modulo responsavel pela divisao temporal (train/test split) de datasets
de series temporais univariadas carregados a partir do Monash Archive.

Contexto de uso no pipeline do projeto:
    O split temporal SEMPRE ocorre ANTES da geracao de lags (lag_creator.py),
    garantindo que nao haja vazamento de dados (data leakage) entre as
    particoes de treino e teste.

Fluxo completo do projeto:
    raw/ (.tsf)
        └─► MonashDataLoader            [data_loader.py]
                └─► TimeSeriesSplitter  [splitter.py]  <- este modulo
                        ├─► data/splits/univariate/train/<dataset>_train.csv
                        └─► data/splits/univariate/test/<dataset>_test.csv
                                └─► LagCreator          [lag_creator.py]
                                        └─► data/processed/univariate/

Estrategia de horizonte:
    Como nenhum dos datasets selecionados define @horizon no arquivo .tsf,
    este modulo adota uma tabela de horizontes padrao por frequencia,
    alinhada a convencao do Monash Forecasting Archive (Godahewa et al.,
    NeurIPS 2021). O usuario pode sobrescrever o horizonte a qualquer momento
    via parametro `horizon`.

    Horizontes padrao adotados:
        half_hourly -> 336  (1 semana: 48 obs/dia x 7 dias)
        hourly     -> 168  (1 semana)
        daily      ->  30  (1 mes)
        weekly     ->   8
        monthly    ->  12  (1 ano)
        quarterly  ->   4  (1 ano)
        yearly     ->   4

Datasets do projeto e seus horizontes resolvidos:
    australian_electricity_demand  -> half_hourly -> h=336
    bitcoin_without_missing        -> daily      -> h=30
    saugeenday                     -> daily      -> h=30
    sunspot_without_missing        -> daily      -> h=30
    us_births                      -> daily      -> h=30

Uso basico - arquivo unico:
    from src.splitter import TimeSeriesSplitter

    splitter = TimeSeriesSplitter()
    result = splitter.split_file(
        filepath='../data/raw/univariate/saugeenday_dataset.tsf',
        output_dir='../data/splits/univariate/',
    )
    result.print_summary()

Uso basico - lote completo:
    from src.splitter import TimeSeriesSplitter

    splitter = TimeSeriesSplitter()
    report = splitter.split_folder(
        input_dir='../data/raw/univariate/',
        output_dir='../data/splits/univariate/',
    )
    report.print_summary()

Uso com horizonte manual:
    splitter = TimeSeriesSplitter(horizon=60)
    splitter.split_file(...)

Arquitetura e escalabilidade:
    - DEFAULT_HORIZONS: tabela centralizada de horizontes por frequencia.
      Adicionar uma nova frequencia exige apenas uma nova entrada no dict.
    - FREQUENCY_LABELS: rotulos legiveis por frequencia, usados no output
      do relatorio (ex: 'daily' -> 'Diaria').
    - SplitResult: dataclass que encapsula caminhos e estatisticas de um
      split individual.
    - BatchSplitReport: consolida os resultados do processamento em lote,
      com o mesmo padrao do BatchReport do lag_creator.py.
    - TimeSeriesSplitter: classe principal com separacao clara entre logica
      de split (split_dataframe) e logica de I/O (split_file, split_folder).

Autor: [Seu Nome]
Dissertacao: [Titulo da Dissertacao]
Data: 2024
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Garante que src/ esta no path para importar data_loader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from data_loader import MonashDataLoader

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
# Horizontes padrao por frequencia (convencao Monash Archive)
# Referencia: Godahewa et al., NeurIPS 2021
# ---------------------------------------------------------------------------
DEFAULT_HORIZONS: dict[str, int] = {
    "half_hourly": 336,
    "hourly": 168,
    "daily": 30,
    "weekly": 8,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 4,
}

# ---------------------------------------------------------------------------
# Rotulos legiveis por frequencia - usados apenas no output do relatorio
# ---------------------------------------------------------------------------
FREQUENCY_LABELS: dict[str, str] = {
    "half_hourly": "A cada 30 minutos",
    "hourly": "Horaria",
    "daily": "Diaria",
    "weekly": "Semanal",
    "monthly": "Mensal",
    "quarterly": "Trimestral",
    "yearly": "Anual",
}


# ---------------------------------------------------------------------------
# Mapeamento de frequencias Monash para aliases pandas
# ---------------------------------------------------------------------------
FREQUENCY_TO_PANDAS: dict[str, str] = {
    "half_hourly": "30min",
    "half_hourly": "30min",
    "hourly":      "h",
    "daily":       "D",
    "weekly":      "W",
    "monthly":     "MS",
    "quarterly":   "QS",
    "yearly":      "YS",
}


def generate_timestamps(
    df: pd.DataFrame,
    frequency: str,
    start_col: str = "start_timestamp",
    series_col: str = "series_name",
) -> pd.Series:
    """
    Gera o timestamp real de cada observacao a partir do start_timestamp
    e da frequencia da serie.

    O start_timestamp no formato Monash e replicado em todas as linhas e
    representa o inicio da serie. O timestamp real de cada observacao e:
        timestamp_i = start_timestamp + (posicao_na_serie * frequencia)

    Parametros:
        df (pd.DataFrame): DataFrame com start_timestamp e series_name.
        frequency (str): Frequencia Monash (ex: 'daily', 'half_hourly').
        start_col (str): Coluna de timestamp inicial. Padrao: 'start_timestamp'.
        series_col (str): Coluna de identificacao das series.

    Returns:
        pd.Series: Timestamps reais alinhados ao indice do df.

    Raises:
        ValueError: Se frequency nao estiver mapeada.
    """
    if frequency not in FREQUENCY_TO_PANDAS:
        raise ValueError(
            f"Frequencia '{frequency}' nao mapeada. "
            f"Opcoes: {list(FREQUENCY_TO_PANDAS.keys())}"
        )

    freq_alias = FREQUENCY_TO_PANDAS[frequency]
    timestamps = pd.Series(index=df.index, dtype="datetime64[ns]")

    if series_col in df.columns:
        groups = df.groupby(series_col, sort=False)
    else:
        groups = [(None, df)]

    for _, group in groups:
        start = pd.Timestamp(group[start_col].iloc[0])
        ts = pd.date_range(start=start, periods=len(group), freq=freq_alias)
        timestamps.iloc[group.index] = ts.values

    return timestamps



@dataclass
class SplitResult:
    """
    Encapsula os metadados, estatisticas e caminhos de um split individual.

    Atributos:
        dataset_name (str): Nome base do arquivo processado (sem extensao).
        horizon (int): Numero de observacoes reservadas para o teste.
        train_path (str): Caminho absoluto do CSV de treino salvo.
        test_path (str): Caminho absoluto do CSV de teste salvo.
        n_series (int): Numero de series presentes no dataset.
        train_sizes (dict[str, int]): Mapeamento serie -> n. obs. no treino.
        test_sizes (dict[str, int]): Mapeamento serie -> n. obs. no teste.
        frequency (str | None): Frequencia bruta extraida dos metadados .tsf
            (ex: 'daily', 'half_hourly').
    """

    dataset_name: str
    horizon: int
    train_path: str
    test_path: str
    n_series: int
    train_sizes: dict[str, int] = field(default_factory=dict)
    test_sizes: dict[str, int] = field(default_factory=dict)
    frequency: Optional[str] = None

    @property
    def total_train(self) -> int:
        """Total de observacoes no conjunto de treino (todas as series)."""
        return sum(self.train_sizes.values())

    @property
    def total_test(self) -> int:
        """Total de observacoes no conjunto de teste (todas as series)."""
        return sum(self.test_sizes.values())

    @property
    def frequency_label(self) -> str:
        """Rotulo legivel da frequencia (ex: 'daily' -> 'Diaria')."""
        if self.frequency is None:
            return "N/A"
        return FREQUENCY_LABELS.get(self.frequency.lower(), self.frequency)

    def print_summary(self) -> None:
        """
        Exibe um resumo detalhado do split individual, incluindo tipo de
        serie, tamanhos de treino e teste por serie, e totais agregados.
        """
        sep = "-" * 68
        total = self.total_train + self.total_test
        pct_train = self.total_train / total * 100 if total > 0 else 0
        pct_test = self.total_test / total * 100 if total > 0 else 0

        print(f"\n{sep}")
        print(f"  SPLIT: {self.dataset_name}")
        print(sep)
        print(f"  Tipo de serie  : {self.frequency_label}")
        print(f"  Frequencia     : {self.frequency or 'N/A'}")
        print(f"  Horizonte (h)  : {self.horizon}")
        print(f"  Num. de series : {self.n_series}")
        print(f"  Total treino   : {self.total_train:>8} obs  ({pct_train:.1f}%)")
        print(f"  Total teste    : {self.total_test:>8} obs  ({pct_test:.1f}%)")
        print(f"  Total geral    : {total:>8} obs")

        if self.n_series > 1:
            print(f"\n  Detalhamento por serie:")
            header = (
                f"    {'Serie':<22} {'Treino':>8}  {'Teste':>6}"
                f"  {'Total':>8}  {'% Treino':>8}"
            )
            print(header)
            print(f"    {'-'*22} {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}")
            for name in self.train_sizes:
                tr = self.train_sizes[name]
                te = self.test_sizes.get(name, 0)
                tot = tr + te
                pct = tr / tot * 100 if tot > 0 else 0
                print(
                    f"    {name:<22} {tr:>8}  {te:>6}"
                    f"  {tot:>8}  {pct:>7.1f}%"
                )

        print(f"\n  Treino -> {self.train_path}")
        print(f"  Teste  -> {self.test_path}")
        print(f"{sep}\n")


@dataclass
class BatchSplitReport:
    """
    Consolida os resultados do processamento em lote de uma pasta inteira.

    Mantem o mesmo padrao do BatchReport do lag_creator.py para
    consistencia de interface no projeto.

    Atributos:
        processed (list[SplitResult]): Splits realizados com sucesso.
        failed (list[tuple[str, str]]): Pares (arquivo, mensagem de erro).
        skipped (list[str]): Arquivos ignorados (extensao diferente de .tsf).
    """

    processed: list[SplitResult] = field(default_factory=list)
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

    @property
    def total_train(self) -> int:
        """Soma de todas as observacoes de treino do lote."""
        return sum(r.total_train for r in self.processed)

    @property
    def total_test(self) -> int:
        """Soma de todas as observacoes de teste do lote."""
        return sum(r.total_test for r in self.processed)

    def print_summary(self) -> None:
        """
        Exibe o relatorio consolidado do lote, com uma linha por dataset
        mostrando tipo de serie, horizonte e volumes de treino e teste.
        """
        sep = "=" * 68
        print(f"\n{sep}")
        print("  RELATORIO DE SPLIT EM LOTE")
        print(sep)
        print(f"  Processados : {self.n_processed}")
        print(f"  Falhas      : {self.n_failed}")
        print(f"  Ignorados   : {len(self.skipped)}")

        if self.processed:
            print(f"\n  Splits gerados:")
            header = (
                f"  {'Dataset':<46} {'Tipo':<20}"
                f" {'h':>4}  {'Treino':>8}  {'Teste':>6}"
            )
            print(header)
            print(
                f"  {'-'*46} {'-'*20} {'-'*4}  {'-'*8}  {'-'*6}"
            )
            for r in self.processed:
                print(
                    f"  {r.dataset_name:<46} {r.frequency_label:<20}"
                    f" {r.horizon:>4}  {r.total_train:>8}  {r.total_test:>6}"
                )
            total = self.total_train + self.total_test
            print(f"\n  Total treino (lote) : {self.total_train:>10} obs")
            print(f"  Total teste  (lote) : {self.total_test:>10} obs")
            print(f"  Total geral  (lote) : {total:>10} obs")

        if self.failed:
            print(f"\n  Falhas:")
            for fname, err in self.failed:
                print(f"    [ERRO] {fname}: {err}")

        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Funcoes utilitarias puras
# ---------------------------------------------------------------------------
def resolve_horizon(
    frequency: Optional[str],
    horizon_override: Optional[int],
    metadata_horizon: Optional[int],
) -> int:
    """
    Determina o horizonte de previsao a usar no split, seguindo esta
    ordem de prioridade:

        1. horizon_override  - parametro explícito do usuario
        2. metadata_horizon  - valor de @horizon extraido do arquivo .tsf
        3. DEFAULT_HORIZONS  - convencao Monash por frequencia

    Parametros:
        frequency (str | None): Frequencia do dataset (ex: 'daily').
        horizon_override (int | None): Valor explícito passado pelo usuario.
        metadata_horizon (int | None): Valor de @horizon extraido do .tsf.

    Returns:
        int: Horizonte resolvido.

    Raises:
        ValueError: Se nenhuma das tres fontes produzir um horizonte valido.
    """
    if horizon_override is not None:
        logger.info("Horizonte definido pelo usuario: %d", horizon_override)
        return horizon_override

    if metadata_horizon is not None:
        logger.info(
            "Horizonte extraido dos metadados .tsf: %d", metadata_horizon
        )
        return metadata_horizon

    if frequency is not None:
        freq_lower = frequency.lower()
        if freq_lower in DEFAULT_HORIZONS:
            h = DEFAULT_HORIZONS[freq_lower]
            logger.info(
                "Horizonte padrao para frequencia '%s': %d", frequency, h
            )
            return h

    raise ValueError(
        "Nao foi possivel determinar o horizonte automaticamente. "
        f"Frequencia recebida: '{frequency}'. "
        "Passe 'horizon' explicitamente ou adicione a frequencia "
        "ao dicionario DEFAULT_HORIZONS."
    )


def split_dataframe(
    df: pd.DataFrame,
    horizon: int,
    series_col: str = "series_name",
    value_col: str = "value",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide um DataFrame de series temporais em treino e teste.

    A divisao e feita por serie: as ultimas `horizon` observacoes de
    cada serie vao para o teste; o restante vai para o treino. Cada
    serie tem exatamente `horizon` pontos de teste, independente do
    comprimento total. A ordem cronologica e sempre preservada.

    Parametros:
        df (pd.DataFrame): DataFrame no formato longo com coluna de
            identificacao de serie e coluna de valores.
        horizon (int): Numero de observacoes finais de cada serie
            destinadas ao conjunto de teste.
        series_col (str): Nome da coluna que identifica cada serie.
            Padrao: 'series_name'.
        value_col (str): Nome da coluna de valores (usado na validacao).
            Padrao: 'value'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Par (df_train, df_test).

    Raises:
        ValueError: Se series_col nao existir no DataFrame.
        ValueError: Se horizon >= comprimento de qualquer serie.

    Exemplos:
        >>> df_train, df_test = split_dataframe(df, horizon=30)
        >>> assert len(df_test) == 30 * df['series_name'].nunique()
    """
    if series_col not in df.columns:
        raise ValueError(
            f"Coluna '{series_col}' nao encontrada. "
            f"Colunas disponiveis: {list(df.columns)}"
        )

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for name, group in df.groupby(series_col, sort=False):
        group = group.reset_index(drop=True)
        n = len(group)

        if horizon >= n:
            raise ValueError(
                f"Horizonte ({horizon}) >= comprimento da serie "
                f"'{name}' ({n} obs). Reduza o horizonte."
            )

        train_parts.append(group.iloc[:-horizon].copy())
        test_parts.append(group.iloc[-horizon:].copy())

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test = pd.concat(test_parts, ignore_index=True)

    return df_train, df_test


def build_split_paths(
    input_path: str,
    output_dir: str,
) -> tuple[str, str]:
    """
    Constroi os caminhos de saida para os CSVs de treino e teste.

    Estrutura gerada:
        <output_dir>/train/<stem>_train.csv
        <output_dir>/test/<stem>_test.csv

    Os subdiretorios train/ e test/ sao criados automaticamente
    se nao existirem.

    Parametros:
        input_path (str): Caminho do arquivo de entrada.
        output_dir (str): Diretorio raiz de saida dos splits.

    Returns:
        tuple[str, str]: Par (train_path, test_path) com caminhos absolutos.
    """
    stem = Path(input_path).stem
    abs_out = os.path.abspath(output_dir)

    train_dir = os.path.join(abs_out, "train")
    test_dir = os.path.join(abs_out, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_path = os.path.join(train_dir, f"{stem}_train.csv")
    test_path = os.path.join(test_dir, f"{stem}_test.csv")

    return train_path, test_path


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------
class TimeSeriesSplitter:
    """
    Realiza o split temporal (train/test) de datasets do Monash Archive.

    O split e sempre feito por serie e cronologicamente: as ultimas
    `horizon` observacoes de cada serie formam o teste. Nao ha
    embaralhamento em nenhuma etapa.

    O horizonte e resolvido automaticamente pela seguinte prioridade:
        1. Parametro `horizon` da instancia (override do usuario)
        2. Tag @horizon do arquivo .tsf
        3. Tabela DEFAULT_HORIZONS indexada pela frequencia do dataset

    Parametros:
        horizon (int | None): Horizonte fixo para todos os arquivos.
            Quando None, o horizonte e resolvido automaticamente por
            arquivo. Padrao: None.
        series_col (str): Nome da coluna de identificacao das series.
            Padrao: 'series_name'.
        value_col (str): Nome da coluna de valores.
            Padrao: 'value'.

    Exemplos:
        >>> splitter = TimeSeriesSplitter()
        >>> result = splitter.split_file(
        ...     filepath='../data/raw/univariate/saugeenday_dataset.tsf',
        ...     output_dir='../data/splits/univariate/',
        ... )
        >>> result.print_summary()
    """

    def __init__(
        self,
        horizon: Optional[int] = None,
        series_col: str = "series_name",
        value_col: str = "value",
    ) -> None:
        if horizon is not None and horizon < 1:
            raise ValueError(
                f"'horizon' deve ser >= 1, mas recebeu {horizon}."
            )
        self.horizon = horizon
        self.series_col = series_col
        self.value_col = value_col

    def split_file(
        self,
        filepath: str,
        output_dir: str = "../data/splits/univariate/",
        horizon: Optional[int] = None,
        series_filter: Optional[str] = None,
    ) -> SplitResult:
        """
        Carrega um arquivo .tsf, divide em treino/teste e salva os CSVs.

        Fluxo interno:
            1. Valida existencia do arquivo.
            2. Carrega via MonashDataLoader.
            3. Filtra para a serie selecionada (se series_filter informado).
            4. Resolve o horizonte (override > @horizon > DEFAULT_HORIZONS).
            5. Aplica split_dataframe() por serie.
            6. Salva train.csv e test.csv em subpastas separadas.
            7. Retorna SplitResult com metadados e estatisticas do split.

        Parametros:
            filepath (str): Caminho relativo ou absoluto do arquivo .tsf.
            output_dir (str): Diretorio raiz onde train/ e test/ serao
                criados. Padrao: '../data/splits/univariate/'.
            horizon (int | None): Sobrescreve o horizonte da instancia
                apenas para este arquivo. Padrao: None.
            series_filter (str | None): Nome exato da serie a manter.
                Quando informado, apenas essa serie e incluida no split.
                Quando None, todas as series do arquivo sao processadas.
                Padrao: None.

        Returns:
            SplitResult: Objeto com caminhos dos CSVs e estatisticas.

        Raises:
            FileNotFoundError: Se o arquivo nao existir.
            ValueError: Se series_filter nao existir no dataset.
            ValueError: Se horizonte nao puder ser determinado.
            ValueError: Se horizonte >= comprimento de alguma serie.
        """
        abs_path = os.path.abspath(filepath)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(
                f"Arquivo nao encontrado: '{abs_path}'. "
                "Verifique o caminho relativo informado."
            )

        logger.info("Processando split: %s", os.path.basename(abs_path))

        # Carregamento via MonashDataLoader
        loader = MonashDataLoader(filepath=abs_path, value_column="value")
        loader.load()
        df = loader.dataframe
        meta = loader.metadata

        # Filtragem opcional por serie
        if series_filter is not None:
            series_disponiveis = df[self.series_col].unique().tolist()
            if series_filter not in series_disponiveis:
                raise ValueError(
                    f"Serie '{series_filter}' nao encontrada em "
                    f"'{os.path.basename(abs_path)}'. "
                    f"Series disponiveis: {series_disponiveis}"
                )
            df = df[df[self.series_col] == series_filter].reset_index(
                drop=True
            )
            logger.info("Serie filtrada: '%s'", series_filter)

        # Resolucao do horizonte com prioridade explicita
        effective_horizon = resolve_horizon(
            frequency=meta.frequency,
            horizon_override=horizon or self.horizon,
            metadata_horizon=meta.horizon,
        )

        # Split cronologico por serie
        df_train, df_test = split_dataframe(
            df=df,
            horizon=effective_horizon,
            series_col=self.series_col,
            value_col=self.value_col,
        )

        # Coleta tamanhos do treino por serie — necessario para calcular
        # o start correto do teste na geracao de timestamps
        train_sizes = (
            df_train.groupby(self.series_col)[self.value_col]
            .count()
            .to_dict()
        )
        test_sizes = (
            df_test.groupby(self.series_col)[self.value_col]
            .count()
            .to_dict()
        )

        # Gera timestamp real por observacao e substitui start_timestamp
        if meta.frequency and meta.frequency in FREQUENCY_TO_PANDAS:
            freq_alias = FREQUENCY_TO_PANDAS[meta.frequency]
            processed = []
            for is_test, df_part in ((False, df_train), (True, df_test)):
                df_part = df_part.copy().reset_index(drop=True)
                ts_list = []
                groups = (
                    df_part.groupby(self.series_col, sort=False)
                    if self.series_col in df_part.columns
                    else [(None, df_part)]
                )
                for serie_name, group in groups:
                    raw_start = pd.Timestamp(
                        group["start_timestamp"].iloc[0]
                    )
                    if is_test:
                        n_train = train_sizes.get(serie_name, len(df_train))
                        offset = pd.tseries.frequencies.to_offset(freq_alias)
                        start = raw_start + n_train * offset
                    else:
                        start = raw_start
                    ts = pd.date_range(
                        start=start, periods=len(group), freq=freq_alias
                    )
                    ts_list.append(pd.Series(ts, index=group.index))
                df_part["timestamp"] = pd.concat(ts_list)
                if "start_timestamp" in df_part.columns:
                    df_part = df_part.drop(columns=["start_timestamp"])
                cols = list(df_part.columns)
                priority = [c for c in [self.series_col, "timestamp"]
                            if c in cols]
                other = [c for c in cols if c not in priority]
                df_part = df_part[priority + other]
                processed.append(df_part)
            df_train, df_test = processed
            logger.info(
                "Timestamps reais gerados | freq=%s", meta.frequency
            )
        else:
            logger.warning(
                "Frequencia '%s' nao mapeada — "
                "start_timestamp mantido sem alteracao.",
                meta.frequency,
            )

        # Salva CSVs em subpastas train/ e test/
        train_path, test_path = build_split_paths(abs_path, output_dir)
        df_train.to_csv(train_path, index=False, encoding="utf-8")
        df_test.to_csv(test_path, index=False, encoding="utf-8")

        result = SplitResult(
            dataset_name=Path(abs_path).stem,
            horizon=effective_horizon,
            train_path=train_path,
            test_path=test_path,
            n_series=loader.n_series,
            train_sizes=train_sizes,
            test_sizes=test_sizes,
            frequency=meta.frequency,
        )

        logger.info(
            "Split salvo | treino: %d obs | teste: %d obs | h=%d",
            len(df_train),
            len(df_test),
            effective_horizon,
        )
        return result

    def split_folder(
        self,
        input_dir: str = "../data/raw/univariate/",
        output_dir: str = "../data/splits/univariate/",
        horizon: Optional[int] = None,
        use_config: bool = True,
    ) -> BatchSplitReport:
        """
        Processa em lote todos os arquivos .tsf de uma pasta.

        Arquivos com outras extensoes sao silenciosamente ignorados.
        Falhas em arquivos individuais sao registradas no relatorio sem
        interromper o processamento dos demais.

        Parametros:
            input_dir (str): Pasta contendo os arquivos .tsf.
                Padrao: '../data/raw/univariate/'.
            output_dir (str): Pasta raiz de saida (train/ e test/ sao
                criadas automaticamente dentro dela).
                Padrao: '../data/splits/univariate/'.
            horizon (int | None): Horizonte fixo para todos os arquivos
                do lote. Quando None, cada arquivo resolve o seu proprio.
                Padrao: None.
            use_config (bool): Quando True, consulta src/config.py para
                determinar qual serie usar em cada dataset. Quando False,
                processa todas as series de todos os arquivos.
                Padrao: True.

        Returns:
            BatchSplitReport: Relatorio consolidado do lote.

        Raises:
            FileNotFoundError: Se input_dir nao existir.
        """
        abs_input = os.path.abspath(input_dir)
        if not os.path.isdir(abs_input):
            raise FileNotFoundError(
                f"Pasta nao encontrada: '{abs_input}'."
            )

        # Importa config apenas se necessario, evitando dependencia circular
        series_config = {}
        datasets_ativos = None
        if use_config:
            try:
                from config import resolver_serie as _resolver_serie
                series_config["resolver"] = _resolver_serie
            except ImportError:
                logger.warning(
                    "config.py nao encontrado. "
                    "Processando todas as series sem filtro."
                )
            try:
                from config import DATASETS_ATIVOS as _datasets_ativos
                datasets_ativos = _datasets_ativos
            except ImportError:
                pass

        report = BatchSplitReport()
        all_files = sorted(os.listdir(abs_input))

        logger.info(
            "Iniciando lote de splits | pasta: %s | arquivos: %d",
            abs_input,
            len(all_files),
        )

        for fname in all_files:
            fpath = os.path.join(abs_input, fname)

            if not fname.lower().endswith(".tsf"):
                report.skipped.append(fname)
                logger.debug("Ignorado (nao e .tsf): %s", fname)
                continue

            # Verifica se o dataset esta ativo em DATASETS_ATIVOS
            if datasets_ativos is not None and not datasets_ativos.get(fname, True):
                report.skipped.append(fname)
                logger.info("Dataset ignorado (DATASETS_ATIVOS=False): %s", fname)
                continue

            # Resolve serie via config.py (se disponivel)
            series_filter = None
            if "resolver" in series_config:
                try:
                    _loader = MonashDataLoader(
                        filepath=fpath, value_column="value"
                    )
                    _loader.load()
                    series_filter = series_config["resolver"](
                        fname, _loader.list_series_names()
                    )
                except Exception:
                    series_filter = None

            try:
                result = self.split_file(
                    filepath=fpath,
                    output_dir=output_dir,
                    horizon=horizon,
                    series_filter=series_filter,
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

    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplitter("
            f"horizon={self.horizon or 'auto'}, "
            f"series_col='{self.series_col}', "
            f"value_col='{self.value_col}')"
        )