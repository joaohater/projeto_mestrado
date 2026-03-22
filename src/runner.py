"""
runner.py

Modulo de orquestracao do pipeline de forecasting com paralelizacao.

Responsabilidades:
    - Ler datasets ativos do config.py
    - Rodar modelos em paralelo por dataset usando joblib
    - Modelos estatisticos + Prophet rodam em fit_estatisticos()
    - Modelos de ML (sem Prophet) rodam em fit_ml()
    - Chamar save_result() para cada combinacao dataset x modelo x config
    - Nao exibir nada — exibicao fica nos blocos do notebook

Como usar:
    from runner import fit_estatisticos, fit_ml

    fit_estatisticos()   # estatisticos + Prophet (uma vez por dataset)
    fit_ml()             # RF, XGBoost, KNN, SVR, LinearRegression (lags + intervalo)

Paralelizacao:
    n_jobs controla quantos modelos rodam simultaneamente.
    n_jobs=6 usa todos os 6 cores do Ryzen 5600G.
    n_jobs=-1 usa todos os cores disponiveis automaticamente.

Localizacao no projeto:
    src/runner.py
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from config import DATASETS_ATIVOS, HORIZONTES, SEASON_LENGTHS
from evaluation import save_result

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(name)s - %(message)s',
)

# ---------------------------------------------------------------------------
# Caminhos base — ajuste se necessario
# ---------------------------------------------------------------------------
SPLITS_DIR      = '../data/splits/univariate'
PROCESSED_DIR   = '../data/processed/univariate'
RESULTS_PATH    = '../results/resultados.csv'
PREDICTIONS_DIR = '../results/predictions/'

N_LAGS       = [7, 14, 30]
WINDOW_SIZES = [7, 14, 30]

# Numero de workers paralelos.
# 6 = todos os cores fisicos do Ryzen 5600G
# -1 = todos os cores logicos (incluindo threads)
N_JOBS = os.cpu_count() - 2


# ---------------------------------------------------------------------------
# Funcoes auxiliares — chamadas pelo joblib
# ---------------------------------------------------------------------------

def _rodar_modelo_estatistico(
    nome:    str,
    model,
    y_train: object,
    y_test:  object,
    dataset: str,
    h:       int,
) -> None:
    """Ajusta um modelo estatistico e salva o resultado."""
    try:
        model.fit(y_train)
        preds = model.predict(h=h)
        save_result(
            dataset=dataset, modelo=nome,
            tipo_feature='estatistico', config='-',
            y_true=y_test, y_pred=preds,
            output_path=RESULTS_PATH,
            predictions_dir=PREDICTIONS_DIR,
            print_result=False,
        )
    except Exception as exc:
        logger.error("Falha: %s | %s | %s", dataset, nome, exc)


def _rodar_prophet(
    y_train_df: object,
    y_test_df:  object,
    y_test:     object,
    dataset:    str,
) -> None:
    """
    Ajusta o Prophet usando a serie original de treino (splits/) e salva
    o resultado. Chamado uma unica vez por dataset, sem depender de
    configuracao de features.
    """
    try:
        from ml import ProphetForecaster

        model = ProphetForecaster()
        model.fit(y_train_df)
        preds = model.predict(y_test_df)
        save_result(
            dataset=dataset, modelo='prophet',
            tipo_feature='estatistico', config='-',
            y_true=y_test, y_pred=preds,
            output_path=RESULTS_PATH,
            predictions_dir=PREDICTIONS_DIR,
            print_result=False,
        )
    except Exception as exc:
        logger.error("Falha Prophet: %s | %s", dataset, exc)


def _rodar_modelo_ml(
    nome:         str,
    model,
    X_train:      object,
    y_train:      object,
    X_test:       object,
    y_test:       object,
    dataset:      str,
    tipo_feature: str,
    config:       str,
) -> None:
    """Ajusta um modelo de ML (sem Prophet) e salva o resultado."""
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        save_result(
            dataset=dataset, modelo=nome,
            tipo_feature=tipo_feature, config=config,
            y_true=y_test, y_pred=preds,
            output_path=RESULTS_PATH,
            predictions_dir=PREDICTIONS_DIR,
            print_result=False,
        )
    except Exception as exc:
        logger.error("Falha: %s | %s | %s | %s", dataset, nome, config, exc)


# ---------------------------------------------------------------------------
# Estatisticos + Prophet
# ---------------------------------------------------------------------------

def fit_estatisticos() -> None:
    """
    Roda todos os modelos estatisticos ativos + Prophet para cada dataset ativo.

    - Modelos estatisticos: rodam em paralelo (n_jobs workers).
    - Prophet: roda em sequencia apos os estatisticos, usando a serie
      original de treino (splits/), sem depender de configuracao de features.
    """
    from statistical import get_statistical_models
    from config import MODELOS_ATIVOS

    logger.info("Iniciando rodada de modelos estatisticos + Prophet.")

    for arquivo_tsf, ativo in DATASETS_ATIVOS.items():
        if not ativo:
            continue

        stem          = arquivo_tsf.replace('.tsf', '')
        dataset       = stem
        h             = HORIZONTES[arquivo_tsf]
        season_length = SEASON_LENGTHS[arquivo_tsf]

        train_path = f'{SPLITS_DIR}/train/{stem}_train.csv'
        test_path  = f'{SPLITS_DIR}/test/{stem}_test.csv'

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logger.warning(
                "Arquivos de split nao encontrados para '%s'. "
                "Execute o splitter primeiro.", stem
            )
            continue

        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)
        y_train  = df_train['value'].values
        y_test   = df_test['value'].values

        logger.info(
            "Dataset: %s | h=%d | season_length=%d",
            dataset, h, season_length
        )

        # -- Modelos estatisticos em paralelo (com season_length correto) ----
        modelos = get_statistical_models(season_length=season_length)

        Parallel(n_jobs=N_JOBS)(
            delayed(_rodar_modelo_estatistico)(
                nome, model, y_train, y_test, dataset, h
            )
            for nome, model in modelos.items()
        )

        # -- Prophet: uma unica vez por dataset, serie original --------------
        if MODELOS_ATIVOS.get('prophet', False):
            prophet_train = df_train[['timestamp', 'value']].rename(
                columns={'timestamp': 'ds', 'value': 'y'}
            )
            prophet_test = df_test[['timestamp']].rename(
                columns={'timestamp': 'ds'}
            )
            # Remove timezone se presente (ex: australian tem +00:00)
            prophet_train['ds'] = pd.to_datetime(
                prophet_train['ds']
            ).dt.tz_localize(None)
            prophet_test['ds'] = pd.to_datetime(
                prophet_test['ds']
            ).dt.tz_localize(None)

            logger.info("Rodando Prophet para %s...", dataset)
            _rodar_prophet(prophet_train, prophet_test, y_test, dataset)

    logger.info("Rodada estatisticos + Prophet concluida.")


# ---------------------------------------------------------------------------
# Machine Learning (sem Prophet)
# ---------------------------------------------------------------------------

def fit_ml() -> None:
    """
    Roda todos os modelos de ML ativos (exceto Prophet) para cada dataset
    ativo, nas configuracoes de lags e intervalo definidas em N_LAGS e
    WINDOW_SIZES.

    Prophet foi movido para fit_estatisticos() — ele nao usa features de
    lags ou intervalo, portanto nao faz sentido repeti-lo por configuracao.
    """
    from ml import get_ml_models

    logger.info("Iniciando rodada de modelos de ML (sem Prophet).")

    for arquivo_tsf, ativo in DATASETS_ATIVOS.items():
        if not ativo:
            continue

        stem    = arquivo_tsf.replace('.tsf', '')
        dataset = stem

        logger.info("Dataset: %s", dataset)

        # -- Lags ----------------------------------------------------------
        for n in N_LAGS:
            train_path = f'{PROCESSED_DIR}/lags_{n}/train/{stem}_train_lags_{n}.csv'
            test_path  = f'{PROCESSED_DIR}/lags_{n}/test/{stem}_test_lags_{n}.csv'

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logger.warning(
                    "Arquivos de lags_%d nao encontrados para '%s'. "
                    "Execute o lag_creator primeiro.", n, stem
                )
                continue

            df_train = pd.read_csv(train_path)
            df_test  = pd.read_csv(test_path)

            feat    = [c for c in df_train.columns if c.startswith('lag_')]
            X_train = df_train[feat].values
            y_train = df_train['value'].values
            X_test  = df_test[feat].values
            y_test  = df_test['value'].values

            # get_ml_models() ja exclui o Prophet se MODELOS_ATIVOS['prophet']
            # for False; aqui forcamos a exclusao independentemente
            modelos = {
                nome: model
                for nome, model in get_ml_models().items()
                if nome != 'prophet'
            }

            Parallel(n_jobs=N_JOBS)(
                delayed(_rodar_modelo_ml)(
                    nome, model,
                    X_train, y_train, X_test, y_test,
                    dataset, 'lags', f'n={n}',
                )
                for nome, model in modelos.items()
            )

        # -- Intervalo -----------------------------------------------------
        for w in WINDOW_SIZES:
            train_path = f'{PROCESSED_DIR}/interval_{w}/train/{stem}_train_interval_{w}.csv'
            test_path  = f'{PROCESSED_DIR}/interval_{w}/test/{stem}_test_interval_{w}.csv'

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logger.warning(
                    "Arquivos de interval_%d nao encontrados para '%s'. "
                    "Execute o interval_creator primeiro.", w, stem
                )
                continue

            df_train = pd.read_csv(train_path)
            df_test  = pd.read_csv(test_path)

            feat    = [c for c in df_train.columns
                       if any(c.startswith(p) for p in ['min_', 'mean_', 'max_'])]
            X_train = df_train[feat].values
            y_train = df_train['value'].values
            X_test  = df_test[feat].values
            y_test  = df_test['value'].values

            modelos = {
                nome: model
                for nome, model in get_ml_models().items()
                if nome != 'prophet'
            }

            Parallel(n_jobs=N_JOBS)(
                delayed(_rodar_modelo_ml)(
                    nome, model,
                    X_train, y_train, X_test, y_test,
                    dataset, 'intervalo', f'w={w}',
                )
                for nome, model in modelos.items()
            )

    logger.info("Rodada ML concluida.")


# ---------------------------------------------------------------------------
# Percentis (interval_percentile_creator)
# ---------------------------------------------------------------------------

PERCENTILE_SIZES = [7, 14, 30]

def fit_percentile() -> None:
    """
    Roda todos os modelos de ML ativos para cada dataset ativo,
    nas configuracoes de percentil definidas em PERCENTILE_SIZES.

    Features usadas: p5_W, p10_W, p15_W, p25_W, p50_W, p75_W, p95_W
    tipo_feature = 'percentile', config = 'w=W'
    """
    from ml import get_ml_models

    logger.info("Iniciando rodada de percentis.")

    for arquivo_tsf, ativo in DATASETS_ATIVOS.items():
        if not ativo:
            continue

        stem    = arquivo_tsf.replace('.tsf', '')
        dataset = stem

        logger.info("Dataset: %s", dataset)

        for w in PERCENTILE_SIZES:
            train_path = f'{PROCESSED_DIR}/percentile_{w}/train/{stem}_train_percentile_{w}.csv'
            test_path  = f'{PROCESSED_DIR}/percentile_{w}/test/{stem}_test_percentile_{w}.csv'

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logger.warning(
                    "Arquivos de percentile_%d nao encontrados para '%s'. "
                    "Execute o interval_percentile_creator primeiro.", w, stem
                )
                continue

            df_train = pd.read_csv(train_path)
            df_test  = pd.read_csv(test_path)

            feat    = [c for c in df_train.columns if c.startswith('p') and '_' in c]
            X_train = df_train[feat].values
            y_train = df_train['value'].values
            X_test  = df_test[feat].values
            y_test  = df_test['value'].values

            modelos = {
                nome: model
                for nome, model in get_ml_models().items()
                if nome != 'prophet'
            }

            Parallel(n_jobs=N_JOBS)(
                delayed(_rodar_modelo_ml)(
                    nome, model,
                    X_train, y_train, X_test, y_test,
                    dataset, 'percentile', f'w={w}',
                )
                for nome, model in modelos.items()
            )

    logger.info("Rodada percentile concluida.")


# ---------------------------------------------------------------------------
# Lags + Intervalo combinados
# ---------------------------------------------------------------------------

def fit_lags_intervalo() -> None:
    """
    Roda todos os modelos de ML ativos combinando features de lags e intervalo.

    Configs pareadas: n=7+w=7, n=14+w=14, n=30+w=30
    tipo_feature = 'lags+intervalo', config = 'n=N+w=W'
    """
    from ml import get_ml_models
    from feature_combiner import combine_features, get_feature_columns

    logger.info("Iniciando rodada lags+intervalo.")

    for arquivo_tsf, ativo in DATASETS_ATIVOS.items():
        if not ativo:
            continue

        stem    = arquivo_tsf.replace('.tsf', '')
        dataset = stem

        logger.info("Dataset: %s", dataset)

        for n in N_LAGS:
            lags_train = f'{PROCESSED_DIR}/lags_{n}/train/{stem}_train_lags_{n}.csv'
            lags_test  = f'{PROCESSED_DIR}/lags_{n}/test/{stem}_test_lags_{n}.csv'
            int_train  = f'{PROCESSED_DIR}/interval_{n}/train/{stem}_train_interval_{n}.csv'
            int_test   = f'{PROCESSED_DIR}/interval_{n}/test/{stem}_test_interval_{n}.csv'

            paths = [lags_train, lags_test, int_train, int_test]
            if not all(os.path.exists(p) for p in paths):
                logger.warning(
                    "Arquivos nao encontrados para lags+intervalo n=%d | '%s'.", n, stem
                )
                continue

            df_lags_train = pd.read_csv(lags_train)
            df_lags_test  = pd.read_csv(lags_test)
            df_int_train  = pd.read_csv(int_train)
            df_int_test   = pd.read_csv(int_test)

            df_train = combine_features(df_lags_train, df_int_train)
            df_test  = combine_features(df_lags_test,  df_int_test)

            feat    = get_feature_columns(df_train)
            X_train = df_train[feat].values
            y_train = df_train['value'].values
            X_test  = df_test[feat].values
            y_test  = df_test['value'].values

            modelos = {
                nome: model
                for nome, model in get_ml_models().items()
                if nome != 'prophet'
            }

            Parallel(n_jobs=N_JOBS)(
                delayed(_rodar_modelo_ml)(
                    nome, model,
                    X_train, y_train, X_test, y_test,
                    dataset, 'lags+intervalo', f'n={n}+w={n}',
                )
                for nome, model in modelos.items()
            )

    logger.info("Rodada lags+intervalo concluida.")


# ---------------------------------------------------------------------------
# Lags + Percentil combinados
# ---------------------------------------------------------------------------

def fit_lags_percentile() -> None:
    """
    Roda todos os modelos de ML ativos combinando features de lags e percentil.

    Configs pareadas: n=7+w=7, n=14+w=14, n=30+w=30
    tipo_feature = 'lags+percentile', config = 'n=N+w=W'
    """
    from ml import get_ml_models
    from feature_combiner import combine_features, get_feature_columns

    logger.info("Iniciando rodada lags+percentile.")

    for arquivo_tsf, ativo in DATASETS_ATIVOS.items():
        if not ativo:
            continue

        stem    = arquivo_tsf.replace('.tsf', '')
        dataset = stem

        logger.info("Dataset: %s", dataset)

        for n in N_LAGS:
            lags_train = f'{PROCESSED_DIR}/lags_{n}/train/{stem}_train_lags_{n}.csv'
            lags_test  = f'{PROCESSED_DIR}/lags_{n}/test/{stem}_test_lags_{n}.csv'
            pct_train  = f'{PROCESSED_DIR}/percentile_{n}/train/{stem}_train_percentile_{n}.csv'
            pct_test   = f'{PROCESSED_DIR}/percentile_{n}/test/{stem}_test_percentile_{n}.csv'

            paths = [lags_train, lags_test, pct_train, pct_test]
            if not all(os.path.exists(p) for p in paths):
                logger.warning(
                    "Arquivos nao encontrados para lags+percentile n=%d | '%s'.", n, stem
                )
                continue

            df_lags_train = pd.read_csv(lags_train)
            df_lags_test  = pd.read_csv(lags_test)
            df_pct_train  = pd.read_csv(pct_train)
            df_pct_test   = pd.read_csv(pct_test)

            df_train = combine_features(df_lags_train, df_pct_train)
            df_test  = combine_features(df_lags_test,  df_pct_test)

            feat    = get_feature_columns(df_train)
            X_train = df_train[feat].values
            y_train = df_train['value'].values
            X_test  = df_test[feat].values
            y_test  = df_test['value'].values

            modelos = {
                nome: model
                for nome, model in get_ml_models().items()
                if nome != 'prophet'
            }

            Parallel(n_jobs=N_JOBS)(
                delayed(_rodar_modelo_ml)(
                    nome, model,
                    X_train, y_train, X_test, y_test,
                    dataset, 'lags+percentile', f'n={n}+w={n}',
                )
                for nome, model in modelos.items()
            )

    logger.info("Rodada lags+percentile concluida.")