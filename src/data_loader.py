"""
data_loader.py

Módulo responsável pelo carregamento, parsing e visualização inicial dos
datasets no formato .tsf do Monash Time Series Forecasting Archive.

Formato .tsf (baseado no ARFF do Weka):
    - Seção de cabeçalho: linhas iniciadas com '@' contendo metadados
      do dataset (@frequency, @horizon, @missing, @equallength, @attribute).
    - Seção de dados: iniciada pela tag '@data', onde cada linha representa
      uma série temporal com atributos separados por ':'.

Funcionalidades principais:
    - Leitura e parsing robusto de arquivos .tsf (univariados e multivariados).
    - Extração de metadados (frequência, horizonte, valores ausentes, etc.).
    - Geração de índice temporal para cada série.
    - Visualização das séries temporais contidas no arquivo.
    - Arquitetura orientada a objetos preparada para extensão futura
      (dados multivariados, múltiplas fontes de dados).

Uso básico:
    from src.data_loader import MonashDataLoader

    loader = MonashDataLoader('../data/raw/univariate/m4_daily.tsf')
    loader.load()
    loader.print_summary()
    loader.plot_series(n_series=3)

Autor: [Seu Nome]
Dissertação: [Título da Dissertação]
Data: 2024
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapeamento de frequência Monash → alias Pandas
# ---------------------------------------------------------------------------
FREQUENCY_MAP: dict[str, str] = {
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
    "yearly": "YS",
    "quarterly": "QS",
    "hourly": "h",
    "minutely": "min",
    "secondly": "s",
    "4_seconds": "4s",
    "10_minutes": "10min",
    "half_hourly": "30min",
}


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------
@dataclass
class TSFMetadata:
    """
    Armazena os metadados extraídos do cabeçalho de um arquivo .tsf.

    Atributos:
        frequency (str | None): Frequência declarada no arquivo
            (ex: 'daily', 'monthly').
        horizon (int | None): Horizonte de previsão recomendado.
        has_missing (bool): Indica se o dataset contém valores ausentes.
        equal_length (bool): Indica se todas as séries têm o mesmo
            comprimento.
        attributes (list[tuple[str, str]]): Lista de pares (nome, tipo)
            dos atributos por série declarados via '@attribute'.
        pandas_freq (str | None): Alias Pandas correspondente à frequência
            declarada; None quando a frequência não está mapeada.
    """

    frequency: Optional[str] = None
    horizon: Optional[int] = None
    has_missing: bool = False
    equal_length: bool = False
    attributes: list[tuple[str, str]] = field(default_factory=list)

    @property
    def pandas_freq(self) -> Optional[str]:
        """Converte a frequência Monash para o alias usado pelo Pandas."""
        if self.frequency is None:
            return None
        return FREQUENCY_MAP.get(self.frequency.lower())


# ---------------------------------------------------------------------------
# Loader principal
# ---------------------------------------------------------------------------
class MonashDataLoader:
    """
    Carrega e expõe os dados de um arquivo .tsf do Monash Archive.

    O parsing segue a especificação original do repositório Monash
    (Godahewa et al., NeurIPS 2021):
      1. Linhas de cabeçalho (@tag valor) são lidas até '@data'.
      2. Cada linha de dados corresponde a uma série; os campos são
         separados por ':'. O campo de valores é a última coluna e
         contém os pontos separados por ','.
      3. Valores ausentes são representados pelo token '?'.

    Parâmetros:
        filepath (str): Caminho relativo ou absoluto para o arquivo .tsf.
        missing_token (str): Token que representa valor ausente dentro
            do arquivo. Padrão: '?'.
        value_column (str): Nome da coluna de valores na saída.
            Padrão: 'series_value'.

    Exemplos:
        >>> loader = MonashDataLoader('../data/raw/univariate/m4_daily.tsf')
        >>> loader.load()
        >>> loader.print_summary()
        >>> df = loader.dataframe
    """

    def __init__(
        self,
        filepath: str,
        missing_token: str = "?",
        value_column: str = "series_value",
    ) -> None:
        self.filepath = filepath
        self.missing_token = missing_token
        self.value_column = value_column

        self.metadata: TSFMetadata = TSFMetadata()
        self._series_list: list[dict] = []  # lista interna de séries brutas
        self._dataframe: Optional[pd.DataFrame] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Propriedades públicas
    # ------------------------------------------------------------------
    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Retorna o DataFrame com todas as séries carregadas.

        Colunas mínimas garantidas:
            - series_name (str)  — quando disponível no arquivo
            - start_timestamp (datetime) — quando disponível no arquivo
            - <value_column> (pd.Series) — valores da série

        Raises:
            RuntimeError: Se `load()` ainda não foi chamado.
        """
        self._assert_loaded()
        return self._dataframe

    @property
    def n_series(self) -> int:
        """Número de séries temporais carregadas."""
        self._assert_loaded()
        return len(self._series_list)

    # ------------------------------------------------------------------
    # Método principal de carregamento
    # ------------------------------------------------------------------
    def load(self) -> "MonashDataLoader":
        """
        Faz o parsing completo do arquivo .tsf e popula `self.dataframe`.

        O método é encadeável: `loader.load().print_summary()`.

        Raises:
            FileNotFoundError: Arquivo não encontrado no caminho informado.
            ValueError: Arquivo malformado (sem seção '@data', etc.).

        Returns:
            MonashDataLoader: A própria instância (fluent interface).
        """
        abs_path = os.path.abspath(self.filepath)
        logger.info("Carregando arquivo: %s", abs_path)

        if not os.path.isfile(abs_path):
            raise FileNotFoundError(
                f"Arquivo não encontrado: '{abs_path}'. "
                "Verifique o caminho relativo."
            )

        with open(abs_path, encoding="utf-8") as fh:
            lines = fh.readlines()

        self._parse(lines)
        self._build_dataframe()
        self._loaded = True
        logger.info(
            "Arquivo carregado: %d série(s) | frequência=%s | horizonte=%s",
            self.n_series,
            self.metadata.frequency,
            self.metadata.horizon,
        )
        return self

    # ------------------------------------------------------------------
    # Parsing interno
    # ------------------------------------------------------------------
    def _parse(self, lines: list[str]) -> None:
        """
        Processa as linhas brutas do arquivo .tsf.

        Separa a seção de metadados (cabeçalho '@tag') da seção de dados
        ('@data') e delega o parsing de cada parte aos métodos auxiliares.

        Parâmetros:
            lines (list[str]): Linhas brutas lidas do arquivo.

        Raises:
            ValueError: Quando a tag '@data' está ausente.
        """
        in_data_section = False
        data_lines: list[str] = []

        for raw_line in lines:
            line = raw_line.strip()

            # Ignora linhas em branco e comentários (#)
            if not line or line.startswith("#"):
                continue

            if line.lower() == "@data":
                in_data_section = True
                continue

            if in_data_section:
                data_lines.append(line)
            else:
                self._parse_header_line(line)

        if not in_data_section:
            raise ValueError(
                "Arquivo .tsf malformado: tag '@data' não encontrada."
            )

        self._parse_data_lines(data_lines)

    def _parse_header_line(self, line: str) -> None:
        """
        Extrai metadados de uma linha de cabeçalho do tipo '@tag valor'.

        Tags reconhecidas:
            @frequency, @horizon, @missing, @equallength, @attribute.

        Parâmetros:
            line (str): Linha de cabeçalho já sem espaços laterais.
        """
        lower = line.lower()

        if lower.startswith("@frequency"):
            self.metadata.frequency = line.split()[1].strip()

        elif lower.startswith("@horizon"):
            try:
                self.metadata.horizon = int(line.split()[1])
            except (IndexError, ValueError):
                logger.warning("Não foi possível converter @horizon para int.")

        elif lower.startswith("@missing"):
            self.metadata.has_missing = line.split()[1].strip().lower() == "true"

        elif lower.startswith("@equallength"):
            self.metadata.equal_length = (
                line.split()[1].strip().lower() == "true"
            )

        elif lower.startswith("@attribute"):
            parts = line.split()
            if len(parts) >= 3:
                self.metadata.attributes.append((parts[1], parts[2]))

    def _parse_data_lines(self, data_lines: list[str]) -> None:
        """
        Converte cada linha da seção '@data' em um dicionário de série.

        Estrutura esperada de cada linha:
            attr1:attr2:...:val1,val2,val3,...

        O último campo sempre contém os valores numéricos da série,
        separados por vírgulas. Os campos anteriores mapeiam 1-a-1
        com os atributos declarados no cabeçalho via '@attribute'.

        Parâmetros:
            data_lines (list[str]): Linhas brutas da seção de dados.
        """
        attr_names = [name for name, _ in self.metadata.attributes]

        for idx, line in enumerate(data_lines):
            if not line:
                continue

            parts = line.split(":")

            # O campo de valores é sempre o último
            raw_values = parts[-1].strip()
            series_values = self._parse_values(raw_values)

            # Demais campos são atributos da série
            series_dict: dict = {}
            for i, attr_name in enumerate(attr_names):
                if i < len(parts) - 1:
                    series_dict[attr_name] = parts[i].strip()

            # Fallback: usa índice como nome caso 'series_name' não exista
            if "series_name" not in series_dict:
                series_dict["series_name"] = f"series_{idx}"

            series_dict[self.value_column] = series_values
            self._series_list.append(series_dict)

    def _parse_values(self, raw: str) -> pd.Series:
        """
        Converte a string de valores brutos em pd.Series numérica.

        Substitui o token de valor ausente ('?') por NaN.

        Parâmetros:
            raw (str): String de valores separados por vírgula.

        Returns:
            pd.Series: Série numérica com dtype float64.
        """
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        values = [
            float("nan") if t == self.missing_token else float(t)
            for t in tokens
        ]
        return pd.Series(values, dtype="float64")

    # ------------------------------------------------------------------
    # Construção do DataFrame de saída
    # ------------------------------------------------------------------
    def _build_dataframe(self) -> None:
        """
        Constrói o DataFrame principal a partir de `self._series_list`.

        Quando 'start_timestamp' e a frequência Pandas estão disponíveis,
        gera um índice temporal para cada série e adiciona a coluna
        'timestamp' ao DataFrame resultante.

        O resultado é armazenado em `self._dataframe`.
        """
        rows = []
        for series in self._series_list:
            values: pd.Series = series[self.value_column]
            timestamps = self._build_timestamps(series, len(values))

            for i, val in enumerate(values):
                row = {k: v for k, v in series.items()
                       if k != self.value_column}
                row[self.value_column] = val
                if timestamps is not None:
                    row["timestamp"] = timestamps[i]
                rows.append(row)

        self._dataframe = pd.DataFrame(rows)

        # Converte colunas numéricas implicitamente string → float
        self._dataframe[self.value_column] = pd.to_numeric(
            self._dataframe[self.value_column], errors="coerce"
        )

    def _build_timestamps(
        self, series: dict, length: int
    ) -> Optional[pd.DatetimeIndex]:
        """
        Gera um DatetimeIndex para uma série, quando possível.

        Requer que o dicionário da série contenha 'start_timestamp'
        e que a frequência do dataset esteja mapeada em FREQUENCY_MAP.

        Parâmetros:
            series (dict): Dicionário com metadados e valores da série.
            length (int): Número de observações na série.

        Returns:
            pd.DatetimeIndex | None: Índice temporal ou None se não for
                possível construí-lo.
        """
        start_raw = series.get("start_timestamp")
        freq = self.metadata.pandas_freq

        if start_raw is None or freq is None:
            return None

        try:
            start_dt = pd.to_datetime(start_raw)
            return pd.date_range(start=start_dt, periods=length, freq=freq)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Não foi possível gerar timestamps para '%s': %s",
                series.get("series_name"),
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Métodos de utilidade pública
    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """
        Exibe um resumo formatado dos metadados e estatísticas básicas
        do dataset carregado.

        Raises:
            RuntimeError: Se `load()` ainda não foi chamado.
        """
        self._assert_loaded()

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  RESUMO DO DATASET MONASH")
        print(f"  Arquivo : {os.path.basename(self.filepath)}")
        print(sep)
        print(f"  Frequência      : {self.metadata.frequency or 'N/A'}")
        print(f"  Alias Pandas    : {self.metadata.pandas_freq or 'N/A'}")
        print(f"  Horizonte (h)   : {self.metadata.horizon or 'N/A'}")
        print(f"  Valores ausentes: {self.metadata.has_missing}")
        print(f"  Séries iguais   : {self.metadata.equal_length}")
        print(f"  Nº de séries    : {self.n_series}")
        print(f"  Atributos       : {self.metadata.attributes}")
        print(sep)

        df = self._dataframe
        lengths = (
            df.groupby("series_name")[self.value_column].count()
        )
        print(f"\n  Comprimento das séries (observações):")
        print(f"    Mínimo  : {lengths.min()}")
        print(f"    Máximo  : {lengths.max()}")
        print(f"    Médio   : {lengths.mean():.1f}")

        if self.metadata.has_missing:
            n_missing = df[self.value_column].isna().sum()
            pct = n_missing / len(df) * 100
            print(f"\n  Valores NaN totais : {n_missing} ({pct:.2f}%)")

        print(f"{sep}\n")

    def get_series(self, series_name: str) -> pd.DataFrame:
        """
        Retorna as observações de uma série específica pelo nome.

        Parâmetros:
            series_name (str): Valor da coluna 'series_name'.

        Returns:
            pd.DataFrame: Subconjunto do dataframe filtrado pelo nome.

        Raises:
            RuntimeError: Se `load()` ainda não foi chamado.
            KeyError: Se o nome da série não existir no dataset.
        """
        self._assert_loaded()
        mask = self._dataframe["series_name"] == series_name
        result = self._dataframe[mask].copy()
        if result.empty:
            raise KeyError(
                f"Série '{series_name}' não encontrada. "
                f"Séries disponíveis: {self.list_series_names()[:5]}..."
            )
        return result

    def list_series_names(self) -> list[str]:
        """
        Retorna a lista de nomes de todas as séries carregadas.

        Returns:
            list[str]: Nomes únicos na coluna 'series_name'.
        """
        self._assert_loaded()
        return self._dataframe["series_name"].unique().tolist()

    def to_wide(self) -> pd.DataFrame:
        """
        Converte o DataFrame do formato longo para o formato largo (wide),
        onde cada coluna representa uma série temporal.

        Útil para modelos que esperam a entrada como matriz (T x N).

        Returns:
            pd.DataFrame: DataFrame no formato largo, indexado por
                'timestamp' (se disponível) ou por posição inteira.

        Raises:
            RuntimeError: Se `load()` ainda não foi chamado.
        """
        self._assert_loaded()
        df = self._dataframe.copy()
        index_col = "timestamp" if "timestamp" in df.columns else None

        if index_col:
            wide = df.pivot_table(
                index=index_col,
                columns="series_name",
                values=self.value_column,
                aggfunc="first",
            )
        else:
            df["_pos"] = df.groupby("series_name").cumcount()
            wide = df.pivot_table(
                index="_pos",
                columns="series_name",
                values=self.value_column,
                aggfunc="first",
            )
        return wide

    # ------------------------------------------------------------------
    # Visualização
    # ------------------------------------------------------------------
    def plot_series(
        self,
        n_series: int = 3,
        figsize: tuple[int, int] = (14, 4),
        series_names: Optional[list[str]] = None,
    ) -> None:
        """
        Plota as observações de uma amostra de séries temporais.

        Parâmetros:
            n_series (int): Número de séries a exibir (ignorado se
                `series_names` for fornecido). Padrão: 3.
            figsize (tuple[int, int]): Tamanho (largura, altura) de cada
                subplot em polegadas. Padrão: (14, 4).
            series_names (list[str] | None): Lista explícita de séries a
                plotar. Quando None, usa as primeiras `n_series`.

        Raises:
            RuntimeError: Se `load()` ainda não foi chamado.
        """
        self._assert_loaded()

        names = series_names or self.list_series_names()[:n_series]
        n = len(names)

        fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n))
        if n == 1:
            axes = [axes]

        for ax, name in zip(axes, names):
            subset = self.get_series(name)
            x_col = "timestamp" if "timestamp" in subset.columns else None

            if x_col:
                ax.plot(subset[x_col], subset[self.value_column], linewidth=1)
                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter("%Y-%m-%d")
                )
                fig.autofmt_xdate(rotation=30)
            else:
                ax.plot(subset[self.value_column].values, linewidth=1)

            ax.set_title(
                f"Série: {name}  |  "
                f"Obs.: {len(subset)}  |  "
                f"Freq.: {self.metadata.frequency or 'N/A'}",
                fontsize=10,
            )
            ax.set_ylabel("Valor")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Monash Archive — {os.path.basename(self.filepath)}",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _assert_loaded(self) -> None:
        """
        Garante que `load()` foi chamado antes de acessar os dados.

        Raises:
            RuntimeError: Se o arquivo ainda não foi carregado.
        """
        if not self._loaded:
            raise RuntimeError(
                "Dados ainda não carregados. Chame `loader.load()` primeiro."
            )

    def __repr__(self) -> str:
        status = "carregado" if self._loaded else "não carregado"
        return (
            f"MonashDataLoader("
            f"filepath='{self.filepath}', "
            f"status={status}, "
            f"n_series={self.n_series if self._loaded else '—'}"
            f")"
        )