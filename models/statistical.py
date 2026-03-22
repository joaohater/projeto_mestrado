"""
statistical.py

Modelos estatisticos de forecasting para o pipeline de dissertacao.

Modelos implementados:
    - Naive
    - Seasonal Naive
    - Simple Exponential Smoothing
    - Holt (Double Exponential Smoothing)
    - Damped Exponential Smoothing
    - ETS (Error, Trend, Seasonality)

Todos os modelos fazem previsao multi-step direta — recebem a serie de
treino completa e retornam h previsoes de uma vez, sem realimentacao.

Estes modelos leem diretamente de data/splits/ (series_name, timestamp,
value) e nao utilizam features de lags ou aritmética intervalar.

Como usar:
    from src.models.statistical import NaiveForecaster, ETSForecaster
    from src.config import MODELOS_ATIVOS

    model = NaiveForecaster()
    model.fit(y_train)
    preds = model.predict(h=30)

Localizacao no projeto:
    src/models/statistical.py

Autor: [Seu Nome]
Dissertacao: [Titulo da Dissertacao]
Data: 2024
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(name)s - %(message)s',
)


# ---------------------------------------------------------------------------
# Classe base
# ---------------------------------------------------------------------------

class BaseStatisticalForecaster(ABC):
    """
    Interface comum para todos os modelos estatisticos.

    Todos os modelos herdam desta classe e implementam obrigatoriamente
    fit() e predict(). Para adicionar um novo modelo estatistico, basta
    criar uma nova classe herdando desta e implementar os dois metodos.
    """

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, y_train: np.ndarray | pd.Series) -> 'BaseStatisticalForecaster':
        """
        Ajusta o modelo a serie de treino.

        Parametros:
            y_train: Serie temporal de treino (valores numericos ordenados
                cronologicamente).

        Returns:
            self — permite encadeamento: model.fit(y).predict(h)
        """

    @abstractmethod
    def predict(self, h: int) -> np.ndarray:
        """
        Gera h previsoes a partir do fim da serie de treino.

        Parametros:
            h (int): Horizonte de previsao (numero de passos a frente).

        Returns:
            np.ndarray de shape (h,) com as previsoes.
        """

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}: chame fit() antes de predict()."
            )


# ---------------------------------------------------------------------------
# Naive
# ---------------------------------------------------------------------------

class NaiveForecaster(BaseStatisticalForecaster):
    """
    Naive: repete o ultimo valor observado para todos os h passos.

    Benchmark minimo — qualquer modelo util deve superar o Naive.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_value: float | None = None

    def fit(self, y_train: np.ndarray | pd.Series) -> 'NaiveForecaster':
        y = np.asarray(y_train, dtype=float)
        self._last_value = float(y[-1])
        self._fitted = True
        logger.info("NaiveForecaster ajustado.")
        return self

    def predict(self, h: int) -> np.ndarray:
        self._check_fitted()
        return np.full(h, self._last_value)


# ---------------------------------------------------------------------------
# Seasonal Naive
# ---------------------------------------------------------------------------

class SeasonalNaiveForecaster(BaseStatisticalForecaster):
    """
    Seasonal Naive: repete o ultimo ciclo sazonal completo.

    Parametros:
        season_length (int): Comprimento do ciclo sazonal.
            Ex: 7 para dados diarios com sazonalidade semanal,
                48 para half-hourly com sazonalidade diaria.
    """

    def __init__(self, season_length: int = 7) -> None:
        super().__init__()
        self.season_length = season_length
        self._last_season: np.ndarray | None = None

    def fit(self, y_train: np.ndarray | pd.Series) -> 'SeasonalNaiveForecaster':
        y = np.asarray(y_train, dtype=float)
        if len(y) < self.season_length:
            raise ValueError(
                f"SeasonalNaive requer ao menos {self.season_length} "
                f"observacoes de treino. Recebeu {len(y)}."
            )
        self._last_season = y[-self.season_length:]
        self._fitted = True
        logger.info(
            f"SeasonalNaiveForecaster ajustado (season_length={self.season_length})."
        )
        return self

    def predict(self, h: int) -> np.ndarray:
        self._check_fitted()
        # Repete o ciclo quantas vezes for necessario e fatia em h
        repeticoes = (h // self.season_length) + 1
        return np.tile(self._last_season, repeticoes)[:h]


# ---------------------------------------------------------------------------
# Simple Exponential Smoothing
# ---------------------------------------------------------------------------

class SimpleESForecaster(BaseStatisticalForecaster):
    """
    Simple Exponential Smoothing (SES).

    Adequado para series sem tendencia e sem sazonalidade.
    O parametro alpha e estimado automaticamente pelo statsmodels.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model_fit = None

    def fit(self, y_train: np.ndarray | pd.Series) -> 'SimpleESForecaster':
        y = np.asarray(y_train, dtype=float)
        model = ExponentialSmoothing(y, trend=None, seasonal=None)
        self._model_fit = model.fit(optimized=True)
        self._fitted = True
        logger.info("SimpleESForecaster ajustado.")
        return self

    def predict(self, h: int) -> np.ndarray:
        self._check_fitted()
        return self._model_fit.forecast(h)


# ---------------------------------------------------------------------------
# Holt (Double Exponential Smoothing)
# ---------------------------------------------------------------------------

class HoltForecaster(BaseStatisticalForecaster):
    """
    Holt Double Exponential Smoothing.

    Captura tendencia linear. Adequado para series com tendencia
    mas sem sazonalidade. Parametros estimados automaticamente.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model_fit = None

    def fit(self, y_train: np.ndarray | pd.Series) -> 'HoltForecaster':
        y = np.asarray(y_train, dtype=float)
        model = ExponentialSmoothing(y, trend='add', seasonal=None)
        self._model_fit = model.fit(optimized=True)
        self._fitted = True
        logger.info("HoltForecaster ajustado.")
        return self

    def predict(self, h: int) -> np.ndarray:
        self._check_fitted()
        return self._model_fit.forecast(h)


# ---------------------------------------------------------------------------
# Damped Exponential Smoothing
# ---------------------------------------------------------------------------

class DampedESForecaster(BaseStatisticalForecaster):
    """
    Damped Exponential Smoothing.

    Variante do Holt com amortecimento da tendencia no longo prazo.
    Tende a ser mais conservador e preciso em horizontes maiores.
    Parametros estimados automaticamente.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model_fit = None

    def fit(self, y_train: np.ndarray | pd.Series) -> 'DampedESForecaster':
        y = np.asarray(y_train, dtype=float)
        model = ExponentialSmoothing(y, trend='add', seasonal=None, damped_trend=True)
        self._model_fit = model.fit(optimized=True)
        self._fitted = True
        logger.info("DampedESForecaster ajustado.")
        return self

    def predict(self, h: int) -> np.ndarray:
        self._check_fitted()
        return self._model_fit.forecast(h)


# ---------------------------------------------------------------------------
# ETS
# ---------------------------------------------------------------------------

class ETSForecaster(BaseStatisticalForecaster):
    """
    ETS (Error, Trend, Seasonality) via Holt-Winters.

    Modelo mais completo entre os estatisticos — captura erro, tendencia
    e sazonalidade. Configuracao automatica via statsmodels quando
    trend e seasonal sao None (selecao por AIC internamente).

    Parametros:
        trend    (str | None): 'add', 'mul' ou None.
        seasonal (str | None): 'add', 'mul' ou None.
        season_length (int | None): Periodo sazonal. Obrigatorio se
            seasonal nao for None.
    """

    def __init__(
        self,
        trend:         str | None = 'add',
        seasonal:      str | None = None,
        season_length: int | None = None,
    ) -> None:
        super().__init__()
        self.trend         = trend
        self.seasonal      = seasonal
        self.season_length = season_length
        self._model_fit    = None

    def fit(self, y_train: np.ndarray | pd.Series) -> 'ETSForecaster':
        y = np.asarray(y_train, dtype=float)

        kwargs: dict = {'trend': self.trend, 'seasonal': self.seasonal}
        if self.seasonal is not None:
            if self.season_length is None:
                raise ValueError(
                    "ETSForecaster: 'season_length' e obrigatorio "
                    "quando 'seasonal' nao e None."
                )
            kwargs['seasonal_periods'] = self.season_length

        model = ExponentialSmoothing(y, **kwargs)
        self._model_fit = model.fit(optimized=True)
        self._fitted = True
        logger.info(
            f"ETSForecaster ajustado "
            f"(trend={self.trend}, seasonal={self.seasonal}, "
            f"season_length={self.season_length})."
        )
        return self

    def predict(self, h: int) -> np.ndarray:
        self._check_fitted()
        return self._model_fit.forecast(h)


# ---------------------------------------------------------------------------
# Fabrica de modelos
# ---------------------------------------------------------------------------

def get_statistical_models(season_length: int = 7) -> dict[str, BaseStatisticalForecaster]:
    """
    Retorna um dicionario com todos os modelos estatisticos ativos,
    filtrados por MODELOS_ATIVOS em config.py.

    Parametros:
        season_length (int): Periodo sazonal usado pelo SeasonalNaive e ETS
            com sazonalidade. Padrao: 7 (semanal para dados diarios).

    Returns:
        dict[str, BaseStatisticalForecaster]: chave = nome do modelo,
            valor = instancia pronta para fit().

    Exemplo:
        modelos = get_statistical_models(season_length=7)
        for nome, model in modelos.items():
            model.fit(y_train)
            preds = model.predict(h=30)
    """
    from config import MODELOS_ATIVOS

    todos = {
        'naive':          NaiveForecaster(),
        'seasonal_naive': SeasonalNaiveForecaster(season_length=season_length),
        'simple_es':      SimpleESForecaster(),
        'holt':           HoltForecaster(),
        'damped_es':      DampedESForecaster(),
        'ets':            ETSForecaster(),
    }

    ativos = {
        nome: modelo
        for nome, modelo in todos.items()
        if MODELOS_ATIVOS.get(nome, False)
    }

    logger.info(
        f"Modelos estatisticos ativos: {list(ativos.keys())}"
    )
    return ativos
