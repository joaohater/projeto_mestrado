"""
ml.py

Modelos de Machine Learning para o pipeline de forecasting da dissertacao.

Modelos implementados:
    - Random Forest
    - XGBoost
    - KNN (K-Nearest Neighbors)
    - SVR (Support Vector Regression)
    - Regressao Linear
    - Prophet

Todos os modelos fazem previsao multi-step direta — recebem X_train/y_train
e retornam h previsoes a partir de X_test, sem realimentacao.

Estes modelos leem de data/processed/ (features de lags ou intervalo),
geradas pelos modulos lag_creator.py e interval_creator.py.

Como usar:
    from src.models.ml import XGBoostForecaster
    from src.config import MODELOS_ATIVOS

    model = XGBoostForecaster()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

Localizacao no projeto:
    src/models/ml.py

Autor: [Seu Nome]
Dissertacao: [Titulo da Dissertacao]
Data: 2024
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(name)s - %(message)s',
)


# ---------------------------------------------------------------------------
# Classe base
# ---------------------------------------------------------------------------

class BaseMLForecaster(ABC):
    """
    Interface comum para todos os modelos de ML.

    Todos os modelos herdam desta classe e implementam obrigatoriamente
    fit() e predict(). Para adicionar um novo modelo de ML, basta criar
    uma nova classe herdando desta e implementar os dois metodos.
    """

    def __init__(self) -> None:
        self._fitted = False
        self.model   = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        params:  dict | None = None,
    ) -> 'BaseMLForecaster':
        """
        Treina o modelo com as features e alvo do conjunto de treino.

        Parametros:
            X_train: Features de treino (lags ou intervalo).
            y_train: Alvo de treino (valor real a prever).
            params : Hiperparametros a usar. Se None, usa HYPERPARAMS
                     do config.py.

        Returns:
            self — permite encadeamento: model.fit(X, y).predict(X_test)
        """

    @abstractmethod
    def predict(self, X_test: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Gera previsoes para o conjunto de teste.

        Parametros:
            X_test: Features de teste (mesma estrutura do X_train).

        Returns:
            np.ndarray de shape (n_test,) com as previsoes.
        """

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}: chame fit() antes de predict()."
            )


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestForecaster(BaseMLForecaster):
    """Random Forest Regressor via scikit-learn."""

    def fit(self, X_train, y_train, params=None) -> 'RandomForestForecaster':
        from sklearn.ensemble import RandomForestRegressor
        from config import HYPERPARAMS

        p = params or HYPERPARAMS['random_forest']
        self.model = RandomForestRegressor(**p)
        self.model.fit(X_train, y_train)
        self._fitted = True
        logger.info(f"RandomForestForecaster ajustado (params={p}).")
        return self

    def predict(self, X_test) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X_test)

    def optimize(self, X_train, y_train, trial) -> dict:
        """Espaço de busca para Optuna."""
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state':      42,
        }
        self.fit(X_train, y_train, params=params)
        return params


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostForecaster(BaseMLForecaster):
    """XGBoost Regressor."""

    def fit(self, X_train, y_train, params=None) -> 'XGBoostForecaster':
        from xgboost import XGBRegressor
        from config import HYPERPARAMS

        p = params or HYPERPARAMS['xgboost']
        self.model = XGBRegressor(**p, verbosity=0)
        self.model.fit(X_train, y_train)
        self._fitted = True
        logger.info(f"XGBoostForecaster ajustado (params={p}).")
        return self

    def predict(self, X_test) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X_test)

    def optimize(self, X_train, y_train, trial) -> dict:
        """Espaço de busca para Optuna."""
        params = {
            'n_estimators':  trial.suggest_int('n_estimators', 50, 500),
            'max_depth':     trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'random_state':  42,
        }
        self.fit(X_train, y_train, params=params)
        return params


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

class KNNForecaster(BaseMLForecaster):
    """K-Nearest Neighbors Regressor via scikit-learn."""

    def fit(self, X_train, y_train, params=None) -> 'KNNForecaster':
        from sklearn.neighbors import KNeighborsRegressor
        from config import HYPERPARAMS

        p = params or HYPERPARAMS['knn']
        self.model = KNeighborsRegressor(**p)
        self.model.fit(X_train, y_train)
        self._fitted = True
        logger.info(f"KNNForecaster ajustado (params={p}).")
        return self

    def predict(self, X_test) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X_test)

    def optimize(self, X_train, y_train, trial) -> dict:
        """Espaço de busca para Optuna."""
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
            'weights':     trial.suggest_categorical('weights', ['uniform', 'distance']),
        }
        self.fit(X_train, y_train, params=params)
        return params


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------

class SVRForecaster(BaseMLForecaster):
    """Support Vector Regression via scikit-learn."""

    def fit(self, X_train, y_train, params=None) -> 'SVRForecaster':
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from config import HYPERPARAMS

        p = params or HYPERPARAMS['svr']

        # SVR e sensivelmente afetado pela escala dos dados
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        self.model = SVR(**p)
        self.model.fit(X_scaled, y_train)
        self._fitted = True
        logger.info(f"SVRForecaster ajustado (params={p}).")
        return self

    def predict(self, X_test) -> np.ndarray:
        self._check_fitted()
        X_scaled = self._scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def optimize(self, X_train, y_train, trial) -> dict:
        """Espaço de busca para Optuna."""
        params = {
            'C':       trial.suggest_float('C', 1e-2, 1e3, log=True),
            'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
            'kernel':  trial.suggest_categorical('kernel', ['rbf', 'linear']),
        }
        self.fit(X_train, y_train, params=params)
        return params


# ---------------------------------------------------------------------------
# Regressao Linear
# ---------------------------------------------------------------------------

class LinearRegressionForecaster(BaseMLForecaster):
    """Regressao Linear via scikit-learn. Sem hiperparametros livres."""

    def fit(self, X_train, y_train, params=None) -> 'LinearRegressionForecaster':
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self._fitted = True
        logger.info("LinearRegressionForecaster ajustado.")
        return self

    def predict(self, X_test) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X_test)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

class ProphetForecaster(BaseMLForecaster):
    """
    Prophet (Meta).

    Classificado como modelo aditivo/probabilistico. Requer que X_train
    seja um DataFrame com colunas 'ds' (timestamp) e 'y' (valor), e
    X_test seja um DataFrame com coluna 'ds' para os periodos futuros.

    Nota: Prophet nao usa as features de lags ou intervalo diretamente.
    Ele opera na serie temporal completa com seu proprio mecanismo de
    decomposicao de tendencia e sazonalidade.
    """

    def fit(self, X_train, y_train=None, params=None) -> 'ProphetForecaster':
        from prophet import Prophet
        from config import HYPERPARAMS

        p = params or HYPERPARAMS['prophet']

        self.model = Prophet(**p)

        # X_train deve ser DataFrame com colunas 'ds' e 'y'
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError(
                "ProphetForecaster: X_train deve ser um DataFrame "
                "com colunas 'ds' (timestamp) e 'y' (valor)."
            )

        self.model.fit(X_train)
        self._fitted = True
        logger.info(f"ProphetForecaster ajustado (params={p}).")
        return self

    def predict(self, X_test) -> np.ndarray:
        """
        Parametros:
            X_test: DataFrame com coluna 'ds' contendo os timestamps futuros.
        """
        self._check_fitted()

        if not isinstance(X_test, pd.DataFrame):
            raise ValueError(
                "ProphetForecaster: X_test deve ser um DataFrame "
                "com coluna 'ds' (timestamps futuros)."
            )

        forecast = self.model.predict(X_test)
        return forecast['yhat'].values

    def optimize(self, X_train, y_train, trial) -> dict:
        """Espaço de busca para Optuna."""
        params = {
            'changepoint_prior_scale': trial.suggest_float(
                'changepoint_prior_scale', 0.001, 0.5, log=True
            ),
            'seasonality_prior_scale': trial.suggest_float(
                'seasonality_prior_scale', 0.01, 10.0, log=True
            ),
        }
        self.fit(X_train, params=params)
        return params


# ---------------------------------------------------------------------------
# Fabrica de modelos
# ---------------------------------------------------------------------------

def get_ml_models() -> dict[str, BaseMLForecaster]:
    """
    Retorna um dicionario com todos os modelos de ML ativos,
    filtrados por MODELOS_ATIVOS em config.py.

    Returns:
        dict[str, BaseMLForecaster]: chave = nome do modelo,
            valor = instancia pronta para fit().

    Exemplo:
        modelos = get_ml_models()
        for nome, model in modelos.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
    """
    from config import MODELOS_ATIVOS

    todos = {
        'random_forest':     RandomForestForecaster(),
        'xgboost':           XGBoostForecaster(),
        'knn':               KNNForecaster(),
        'svr':               SVRForecaster(),
        'linear_regression': LinearRegressionForecaster(),
        'prophet':           ProphetForecaster(),
    }

    ativos = {
        nome: modelo
        for nome, modelo in todos.items()
        if MODELOS_ATIVOS.get(nome, False)
    }

    logger.info(f"Modelos de ML ativos: {list(ativos.keys())}")
    return ativos
