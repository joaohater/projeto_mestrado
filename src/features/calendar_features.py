"""
calendar_features.py

Modulo responsavel pela geracao do timestamp real de cada observacao e,
opcionalmente, pela extracao de features de calendario a partir dele.

Contexto de uso no pipeline do projeto:
    Aplicado apos o lag_creator.py, enriquecendo os CSVs processados com
    informacao temporal explícita para uso nos modelos de forecasting.

Fluxo completo do projeto:
    raw/ (.tsf)
        └─► MonashDataLoader                [data_loader.py]
                └─► TimeSeriesSplitter      [splitter.py]
                        ├─► splits/train/
                        └─► splits/test/
                                └─► LagCreator          [lag_creator.py]
                                        ├─► processed/train/
                                        └─► processed/test/
                                                └─► CalendarFeatures  <- este modulo
                                                        ├─► processed/train/
                                                        └─► processed/test/

Por que o timestamp do Monash nao serve diretamente:
    O campo start_timestamp nos CSVs do pipeline representa o inicio da
    serie inteira — ele e replicado em todas as linhas pelo loader. O
    timestamp real de cada observacao precisa ser recalculado incrementalmente:
        obs_i_timestamp = start_timestamp + (i * frequencia)

    Exemplo para half_hourly:
        obs 0: 2002-01-01 00:00:00
        obs 1: 2002-01-01 00:30:00
        obs 2: 2002-01-01 01:00:00

Comportamento controlado pelo config.py:
    USE_CALENDAR_FEATURES = False -> gera apenas o timestamp real por obs
    USE_CALENDAR_FEATURES = True  -> gera timestamp + colunas de calendario
        As colunas geradas dependem de CALENDAR_FEATURES (controle granular)

Uso basico - arquivo unico:
    from src.features.calendar_features import CalendarFeatures

    cf = CalendarFeatures(frequency='daily')
    result = cf.process_file(
        filepath='../data/processed/univariate/train/saugeenday_dataset_train_lags_5.csv',
        output_dir='../data/processed/univariate/train/',
    )
    result.print_summary()

Uso basico - lote completo:
    from src.features.calendar_features import CalendarFeatures

    cf = CalendarFeatures(frequency='half_hourly')
    report = cf.process_split_folder(
        splits_dir='../data/processed/univariate/',
        output_dir='../data/processed/univariate/',
    )
    report.print_summary()

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
# Mapeamento de frequencias Monash para aliases pandas
# ---------------------------------------------------------------------------
FREQUENCY_TO_PANDAS: dict[str, str] = {
    "30_minutes": "30min",
    "half_hourly": "30min",
    "hourly":      "h",
    "daily":       "D",
    "weekly":      "W",
    "monthly":     "MS",
    "quarterly":   "QS",
    "yearly":      "YS",
}


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------
@dataclass
class CalendarResult:
    """
    Encapsula os metadados e estatisticas de um arquivo processado.

    Atributos:
        input_name (str): Nome base do arquivo de entrada (sem extensao).
        output_path (str): Caminho absoluto do CSV gerado.
        frequency (str): Frequencia usada para gerar os timestamps.
        n_rows (int): Numero de linhas processadas.
        cols_added (list[str]): Colunas adicionadas ao CSV.
        use_calendar (bool): Se features de calendario foram geradas.
    """

    input_name: str
    output_path: str
    frequency: str
    n_rows: int
    cols_added: list[str]
    use_calendar: bool

    def print_summary(self) -> None:
        """Exibe um resumo do arquivo processado."""
        sep = "-" * 68
        print(f"\n{sep}")
        print(f"  CALENDAR: {self.input_name}")
        print(sep)
        print(f"  Frequencia       : {self.frequency}")
        print(f"  Linhas           : {self.n_rows}")
        print(f"  Colunas geradas  : {len(self.cols_added)}")
        for col in self.cols_added:
            print(f"    + {col}")
        print(f"  Features cal.    : {'sim' if self.use_calendar else 'nao'}")
        print(f"\n  Saida -> {self.output_path}")
        print(f"{sep}\n")


@dataclass
class BatchCalendarReport:
    """
    Consolida os resultados do processamento em lote.

    Atributos:
        processed (list[CalendarResult]): Resultados dos arquivos processados.
        failed (list[tuple[str, str]]): Pares (arquivo, mensagem de erro).
        skipped (list[str]): Arquivos ignorados.
    """

    processed: list[CalendarResult] = field(default_factory=list)
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
        print("  RELATORIO DE FEATURES DE CALENDARIO EM LOTE")
        print(sep)
        print(f"  Processados : {self.n_processed}")
        print(f"  Falhas      : {self.n_failed}")
        print(f"  Ignorados   : {len(self.skipped)}")

        if self.processed:
            print(f"\n  Arquivos gerados:")
            header = (
                f"  {'Arquivo':<50} {'Linhas':>8}  {'Cols+':>6}"
            )
            print(header)
            print(f"  {'-'*50} {'-'*8}  {'-'*6}")
            for r in self.processed:
                print(
                    f"  {r.input_name:<50} {r.n_rows:>8}  "
                    f"{len(r.cols_added):>6}"
                )

        if self.failed:
            print(f"\n  Falhas:")
            for fname, err in self.failed:
                print(f"    [ERRO] {fname}: {err}")

        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Funcoes puras
# ---------------------------------------------------------------------------
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
    representa o inicio da serie. O timestamp real de cada observacao e
    calculado como:
        timestamp_i = start_timestamp + (posicao_dentro_da_serie * frequencia)

    Parametros:
        df (pd.DataFrame): DataFrame com as colunas start_timestamp e
            series_name.
        frequency (str): Frequencia da serie no formato Monash
            (ex: 'daily', 'half_hourly').
        start_col (str): Nome da coluna de timestamp inicial.
            Padrao: 'start_timestamp'.
        series_col (str): Nome da coluna de identificacao das series.
            Padrao: 'series_name'.

    Returns:
        pd.Series: Serie com os timestamps reais, alinhada ao indice do df.

    Raises:
        ValueError: Se a frequencia nao estiver mapeada em FREQUENCY_TO_PANDAS.
        ValueError: Se start_col nao existir no DataFrame.
    """
    if frequency not in FREQUENCY_TO_PANDAS:
        raise ValueError(
            f"Frequencia '{frequency}' nao mapeada. "
            f"Opcoes: {list(FREQUENCY_TO_PANDAS.keys())}"
        )
    if start_col not in df.columns:
        raise ValueError(
            f"Coluna '{start_col}' nao encontrada. "
            f"Colunas disponiveis: {list(df.columns)}"
        )

    freq_alias = FREQUENCY_TO_PANDAS[frequency]
    timestamps = pd.Series(index=df.index, dtype="datetime64[ns]")

    # Agrupa por serie para calcular a posicao de cada obs dentro da serie
    if series_col in df.columns:
        groups = df.groupby(series_col, sort=False)
    else:
        groups = [(None, df)]

    for _, group in groups:
        start = pd.Timestamp(group[start_col].iloc[0])
        n = len(group)
        ts = pd.date_range(start=start, periods=n, freq=freq_alias)
        timestamps.iloc[group.index] = ts.values

    return timestamps


def extract_calendar_features(
    timestamps: pd.Series,
    features: dict[str, bool],
) -> pd.DataFrame:
    """
    Extrai features de calendario a partir de uma serie de timestamps.

    Apenas as features com valor True em `features` sao geradas.

    Parametros:
        timestamps (pd.Series): Serie de timestamps (datetime64).
        features (dict[str, bool]): Mapa de feature -> ativar/desativar.
            Chaves suportadas: hour, day_of_week, day_of_month, month,
            is_weekend, quarter, week_of_year.

    Returns:
        pd.DataFrame: DataFrame com as colunas de calendario geradas,
            alinhado ao indice de timestamps.
    """
    dt = pd.to_datetime(timestamps)
    result = pd.DataFrame(index=timestamps.index)

    extractors = {
        'hour':         lambda d: d.dt.hour,
        'day_of_week':  lambda d: d.dt.dayofweek,
        'day_of_month': lambda d: d.dt.day,
        'month':        lambda d: d.dt.month,
        'is_weekend':   lambda d: (d.dt.dayofweek >= 5).astype(int),
        'quarter':      lambda d: d.dt.quarter,
        'week_of_year': lambda d: d.dt.isocalendar().week.astype(int),
    }

    for feat, enabled in features.items():
        if enabled and feat in extractors:
            result[feat] = extractors[feat](dt)

    return result


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------
class CalendarFeatures:
    """
    Gera o timestamp real de cada observacao e, opcionalmente, features
    de calendario, a partir dos CSVs produzidos pelo lag_creator.py.

    O comportamento e controlado pelo config.py:
        USE_CALENDAR_FEATURES = False -> apenas timestamp por observacao
        USE_CALENDAR_FEATURES = True  -> timestamp + colunas de calendario

    Parametros:
        frequency (str): Frequencia da serie no formato Monash.
            Ex: 'daily', 'half_hourly', 'hourly'.
        use_calendar (bool | None): Sobrescreve USE_CALENDAR_FEATURES do
            config.py. Quando None, usa o valor do config. Padrao: None.
        calendar_features (dict | None): Sobrescreve CALENDAR_FEATURES do
            config.py. Quando None, usa o valor do config. Padrao: None.
        start_col (str): Coluna de timestamp inicial. Padrao: 'start_timestamp'.
        series_col (str): Coluna de identificacao das series.
            Padrao: 'series_name'.

    Exemplos:
        >>> cf = CalendarFeatures(frequency='daily')
        >>> result = cf.process_file(
        ...     filepath='../data/processed/univariate/train/saugeenday_dataset_train_lags_5.csv',
        ...     output_dir='../data/processed/univariate/train/',
        ... )
        >>> result.print_summary()
    """

    def __init__(
        self,
        frequency: str,
        use_calendar: Optional[bool] = None,
        calendar_features: Optional[dict[str, bool]] = None,
        start_col: str = "start_timestamp",
        series_col: str = "series_name",
    ) -> None:
        if frequency not in FREQUENCY_TO_PANDAS:
            raise ValueError(
                f"Frequencia '{frequency}' nao reconhecida. "
                f"Opcoes: {list(FREQUENCY_TO_PANDAS.keys())}"
            )
        self.frequency = frequency
        self.start_col = start_col
        self.series_col = series_col

        # Carrega config.py como fonte de verdade, com override opcional
        try:
            from config import (
                USE_CALENDAR_FEATURES,
                CALENDAR_FEATURES,
            )
            self.use_calendar = (
                use_calendar
                if use_calendar is not None
                else USE_CALENDAR_FEATURES
            )
            self.calendar_features = calendar_features or CALENDAR_FEATURES
        except ImportError:
            logger.warning(
                "config.py nao encontrado. "
                "Usando use_calendar=%s.",
                use_calendar,
            )
            self.use_calendar = use_calendar if use_calendar is not None else False
            self.calendar_features = calendar_features or {}

    def process_file(
        self,
        filepath: str,
        output_dir: str,
    ) -> CalendarResult:
        """
        Carrega um CSV processado, gera os timestamps e salva o resultado.

        Fluxo interno:
            1. Valida existencia do arquivo.
            2. Carrega o CSV.
            3. Gera o timestamp real de cada observacao via generate_timestamps().
            4. Se use_calendar=True, extrai features via extract_calendar_features().
            5. Insere as novas colunas apos start_timestamp no DataFrame.
            6. Salva o CSV resultante.
            7. Retorna CalendarResult com metadados.

        Parametros:
            filepath (str): Caminho do CSV de entrada (saida do lag_creator).
            output_dir (str): Diretorio de saida para o CSV enriquecido.

        Returns:
            CalendarResult: Objeto com metadados do processamento.

        Raises:
            FileNotFoundError: Se filepath nao existir.
        """
        abs_input = os.path.abspath(filepath)
        if not os.path.isfile(abs_input):
            raise FileNotFoundError(
                f"Arquivo nao encontrado: '{abs_input}'."
            )

        logger.info(
            "Processando: %s | freq=%s | calendar=%s",
            os.path.basename(abs_input),
            self.frequency,
            self.use_calendar,
        )

        df = pd.read_csv(abs_input)
        cols_added = []

        # Gera timestamp real por observacao
        timestamps = generate_timestamps(
            df=df,
            frequency=self.frequency,
            start_col=self.start_col,
            series_col=self.series_col,
        )
        df["timestamp"] = timestamps
        cols_added.append("timestamp")

        # Extrai features de calendario se habilitado
        if self.use_calendar:
            cal_df = extract_calendar_features(timestamps, self.calendar_features)
            for col in cal_df.columns:
                df[col] = cal_df[col]
                cols_added.append(col)

        # Remove start_timestamp — substituido pelo timestamp real
        if self.start_col in df.columns:
            df = df.drop(columns=[self.start_col])

        # Reordena: series_name, timestamp, [calendar], value, lags
        priority = [self.series_col, "timestamp"] + [
            c for c in cols_added if c not in (self.series_col, "timestamp")
        ]
        other_cols = [c for c in df.columns if c not in priority]
        df = df[[c for c in priority if c in df.columns] + other_cols]

        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        out_filename = Path(abs_input).name
        out_path = os.path.join(abs_output_dir, out_filename)
        df.to_csv(out_path, index=False, encoding="utf-8")

        logger.info(
            "Salvo: %s | %d linhas | +%d colunas",
            out_filename,
            len(df),
            len(cols_added),
        )

        return CalendarResult(
            input_name=Path(abs_input).stem,
            output_path=out_path,
            frequency=self.frequency,
            n_rows=len(df),
            cols_added=cols_added,
            use_calendar=self.use_calendar,
        )

    def process_folder(
        self,
        input_dir: str,
        output_dir: str,
    ) -> BatchCalendarReport:
        """
        Processa em lote todos os CSVs de uma pasta.

        Parametros:
            input_dir (str): Pasta com os CSVs de entrada.
            output_dir (str): Pasta de saida para os CSVs enriquecidos.

        Returns:
            BatchCalendarReport: Relatorio consolidado do lote.

        Raises:
            FileNotFoundError: Se input_dir nao existir.
        """
        abs_input = os.path.abspath(input_dir)
        if not os.path.isdir(abs_input):
            raise FileNotFoundError(
                f"Pasta nao encontrada: '{abs_input}'."
            )

        report = BatchCalendarReport()
        all_files = sorted(os.listdir(abs_input))

        for fname in all_files:
            fpath = os.path.join(abs_input, fname)

            if not fname.lower().endswith(".csv"):
                report.skipped.append(fname)
                continue

            try:
                result = self.process_file(
                    filepath=fpath,
                    output_dir=output_dir,
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
    ) -> BatchCalendarReport:
        """
        Processa em lote as pastas train/ e test/ produzidas pelo lag_creator.

        Estrutura de entrada esperada:
            <splits_dir>/train/*.csv
            <splits_dir>/test/*.csv

        Estrutura de saida gerada:
            <output_dir>/train/*.csv  (sobrescreve com timestamp e calendario)
            <output_dir>/test/*.csv

        Parametros:
            splits_dir (str): Diretorio raiz com subpastas train/ e test/.
            output_dir (str): Diretorio raiz de saida.

        Returns:
            BatchCalendarReport: Relatorio consolidado de ambas as particoes.
        """
        combined = BatchCalendarReport()

        for partition in ("train", "test"):
            part_input  = os.path.join(os.path.abspath(splits_dir), partition)
            part_output = os.path.join(os.path.abspath(output_dir),  partition)

            if not os.path.isdir(part_input):
                logger.warning(
                    "Subpasta '%s/' nao encontrada em '%s'. Ignorando.",
                    partition,
                    splits_dir,
                )
                continue

            logger.info("Processando particao: %s/", partition)
            part_report = self.process_folder(part_input, part_output)
            combined.processed.extend(part_report.processed)
            combined.failed.extend(part_report.failed)
            combined.skipped.extend(part_report.skipped)

        return combined

    def __repr__(self) -> str:
        active = [k for k, v in self.calendar_features.items() if v]
        return (
            f"CalendarFeatures("
            f"frequency='{self.frequency}', "
            f"use_calendar={self.use_calendar}, "
            f"features={active})"
        )
