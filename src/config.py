"""
config.py

Configuracoes centralizadas do pipeline de forecasting.

Este modulo e a unica fonte de verdade para todas as decisoes metodologicas
do projeto — selecao de series, horizontes, features, modelos e otimizacao.
Nenhum modulo deve conter valores fixos que pertencam a este arquivo.

Localizacao no projeto:
    src/config.py

Autor: [Seu Nome]
Dissertacao: [Titulo da Dissertacao]
Data: 2024
"""

# ---------------------------------------------------------------------------
# Selecao de series por dataset
# ---------------------------------------------------------------------------
# Mapeia o nome do arquivo .tsf para o nome exato da serie a utilizar.
# - Valor str  : nome exato da serie (via loader.list_series_names())
# - Valor None : usa a primeira serie retornada pelo loader
#
# Criterio de escolha:
#   bitcoin  -> 'price': serie de preco agregado, mais usada em benchmarks
#   australian_electricity_demand -> None: primeira serie (Victoria)
#   saugeenday, sunspot, us_births -> serie unica
# ---------------------------------------------------------------------------
SERIES_SELECIONADA: dict[str, str | None] = {
    'bitcoin_dataset_without_missing_values.tsf': 'price',
    'australian_electricity_demand_dataset.tsf':  None,
    'saugeenday_dataset.tsf':                     None,
    'sunspot_dataset_without_missing_values.tsf': None,
    'us_births_dataset.tsf':                      None,
    'dataset_teste.tsf':                          None,
}

# ---------------------------------------------------------------------------
# Horizontes de previsao por dataset
# ---------------------------------------------------------------------------
# Alinhados a convencao do Monash Forecasting Archive (Godahewa et al.,
# NeurIPS 2021). Nenhum dos datasets selecionados define @horizon no .tsf.
# ---------------------------------------------------------------------------
HORIZONTES: dict[str, int] = {
    'australian_electricity_demand_dataset.tsf':  336,
    'bitcoin_dataset_without_missing_values.tsf':  30,
    'saugeenday_dataset.tsf':                      30,
    'sunspot_dataset_without_missing_values.tsf':  30,
    'us_births_dataset.tsf':                       30,
    'dataset_teste.tsf':                           10,
}

# ---------------------------------------------------------------------------
# Comprimentos sazonais por dataset
# ---------------------------------------------------------------------------
# Usado pelo Seasonal Naive e pelos modelos estatisticos que requerem
# conhecimento do periodo sazonal (ex: ETS com sazonalidade).
#
# Criterio de escolha:
#   australian (half_hourly) -> 336: sazonalidade semanal (48 obs/dia * 7)
#                                    equivalente a repetir a mesma semana
#   bitcoin    (daily)       ->   7: sazonalidade semanal
#   saugeenday (daily)       ->   7: sazonalidade semanal (vazao do rio)
#   sunspot    (daily)       -> 365: sazonalidade anual (ciclo solar)
#   us_births  (daily)       ->   7: sazonalidade semanal (nascimentos)
# ---------------------------------------------------------------------------
SEASON_LENGTHS: dict[str, int] = {
    'australian_electricity_demand_dataset.tsf':  336,
    'bitcoin_dataset_without_missing_values.tsf':   7,
    'saugeenday_dataset.tsf':                       7,
    'sunspot_dataset_without_missing_values.tsf': 365,
    'us_births_dataset.tsf':                        7,
    'dataset_teste.tsf':                            7,
}

# ---------------------------------------------------------------------------
# Features de calendario
# ---------------------------------------------------------------------------
# USE_CALENDAR_FEATURES: flag geral para ligar/desligar features de calendario
#   True  -> gera timestamp real + colunas de calendario ativas abaixo
#   False -> gera apenas o timestamp real por observacao
#
# Recomendacao por frequencia:
#   half_hourly -> hour, day_of_week, is_weekend, month
#   daily       -> day_of_week, day_of_month, month
# ---------------------------------------------------------------------------
USE_CALENDAR_FEATURES: bool = True

CALENDAR_FEATURES: dict[str, bool] = {
    'hour':         True,   # hora do dia (0-23) — relevante para half_hourly
    'day_of_week':  True,   # dia da semana (0=segunda, 6=domingo)
    'day_of_month': True,   # dia do mes (1-31)
    'month':        True,   # mes do ano (1-12)
    'is_weekend':   True,   # 1 se sabado ou domingo, 0 caso contrario
    'quarter':      False,  # trimestre (1-4) — opcional
    'week_of_year': False,  # semana do ano (1-52) — opcional
}

# ---------------------------------------------------------------------------
# Datasets ativos
# ---------------------------------------------------------------------------
# Controla quais datasets serao processados em cada etapa do pipeline.
# - True  : dataset sera incluido no processamento em lote
# - False : dataset sera ignorado em splitter, lag_creator e interval_creator
#
# Util para rodar o pipeline em um subset de datasets sem modificar
# nenhum modulo. Basta desativar os datasets nao desejados aqui.
# ---------------------------------------------------------------------------
DATASETS_ATIVOS: dict[str, bool] = {
    'australian_electricity_demand_dataset.tsf': True,
    'bitcoin_dataset_without_missing_values.tsf': True,
    'saugeenday_dataset.tsf':                     True,
    'sunspot_dataset_without_missing_values.tsf': True,
    'us_births_dataset.tsf':                      True,
    'dataset_teste.tsf':                          False,
}

# ---------------------------------------------------------------------------
# Modelos ativos
# ---------------------------------------------------------------------------
# Controla quais modelos serao executados no pipeline.
# - True  : modelo sera instanciado e executado
# - False : modelo sera ignorado
#
# Modelos estatisticos leem de data/splits/ (sem features).
# Modelos de ML leem de data/processed/ (lags ou intervalo).
# ---------------------------------------------------------------------------
MODELOS_ATIVOS: dict[str, bool] = {
    # Estatisticos
    'naive':             True,
    'seasonal_naive':    True,
    'simple_es':         True,
    'holt':              True,
    'damped_es':         True,
    'ets':               True,
    # Machine Learning
    'random_forest':     True,
    'xgboost':           True,
    'knn':               True,
    'svr':               False,
    'linear_regression': True,
    'prophet':           True,
}

# ---------------------------------------------------------------------------
# Otimizacao de hiperparametros (Optuna)
# ---------------------------------------------------------------------------
# USE_OPTUNA: flag geral para ligar/desligar a otimizacao
#   False -> cada modelo usa os valores de HYPERPARAMS abaixo
#   True  -> Optuna busca os melhores hiperparametros dentro dos
#            espacos de busca definidos em cada modelo
# ---------------------------------------------------------------------------
USE_OPTUNA: bool = False

OPTUNA_CONFIG: dict = {
    'n_trials':  100,     # numero de tentativas por modelo
    'timeout':   3600,    # limite de tempo em segundos (None = sem limite)
    'direction': 'minimize',
    'metric':    'rmse',  # metrica usada como objetivo da otimizacao
}

# ---------------------------------------------------------------------------
# Hiperparametros default dos modelos de ML
# ---------------------------------------------------------------------------
# Usados diretamente quando USE_OPTUNA=False.
# Quando USE_OPTUNA=True, servem como ponto de partida ou referencia.
# Os modelos estatisticos nao possuem hiperparametros configurados aqui
# pois sao ajustados automaticamente (ETS) ou nao possuem parametros
# livres (Naive, Seasonal Naive).
# ---------------------------------------------------------------------------
HYPERPARAMS: dict = {
    'random_forest': {
        'n_estimators':      100,
        'max_depth':         None,
        'min_samples_split': 2,
        'random_state':      42,
    },
    'xgboost': {
        'n_estimators':  100,
        'max_depth':     6,
        'learning_rate': 0.1,
        'random_state':  42,
    },
    'knn': {
        'n_neighbors': 5,
        'weights':     'uniform',
    },
    'svr': {
        'C':       1.0,
        'kernel':  'rbf',
        'epsilon': 0.1,
    },
    'linear_regression': {},
    'prophet': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
    },
}

# ---------------------------------------------------------------------------
# Metricas de avaliacao
# ---------------------------------------------------------------------------
# Controla quais metricas serao calculadas e salvas no CSV de resultados.
# - True  : metrica sera calculada e registrada
# - False : metrica sera ignorada
#
# Atencao:
#   mape -> pode explodir em series com zeros (sunspot, bitcoin em periodos
#            baixos). Tratamento defensivo aplicado no evaluation.py
#   mse  -> util internamente para o Optuna (diferenciavel)
#   rmse -> mesma escala dos dados, mais interpretavel que o MSE
# ---------------------------------------------------------------------------
METRICAS: dict[str, bool] = {
    'mae':  True,
    'mape': True,
    'mse':  True,
    'rmse': True,
}


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def resolver_serie(nome_arquivo: str, series_disponiveis: list[str]) -> str:
    """
    Resolve o nome da serie a utilizar para um dado arquivo .tsf.

    Consulta SERIES_SELECIONADA pelo nome do arquivo. Se o valor for None
    ou o arquivo nao estiver mapeado, retorna a primeira serie disponivel.

    Parametros:
        nome_arquivo (str): Nome do arquivo .tsf (apenas o basename,
            ex: 'bitcoin_dataset_without_missing_values.tsf').
        series_disponiveis (list[str]): Lista de series do dataset,
            obtida via loader.list_series_names().

    Returns:
        str: Nome da serie selecionada.

    Raises:
        ValueError: Se series_disponiveis estiver vazia.
    """
    if not series_disponiveis:
        raise ValueError(
            f"'series_disponiveis' esta vazia para o arquivo '{nome_arquivo}'."
        )

    escolha = SERIES_SELECIONADA.get(nome_arquivo, None)

    if escolha is not None and escolha in series_disponiveis:
        return escolha

    if escolha is not None and escolha not in series_disponiveis:
        import warnings
        warnings.warn(
            f"Serie '{escolha}' definida em SERIES_SELECIONADA nao encontrada "
            f"em '{nome_arquivo}'. Usando a primeira serie disponivel: "
            f"'{series_disponiveis[0]}'.",
            UserWarning,
            stacklevel=2,
        )

    return series_disponiveis[0]


def resolver_horizonte(nome_arquivo: str) -> int:
    """
    Retorna o horizonte de previsao definido para um dado arquivo .tsf.

    Parametros:
        nome_arquivo (str): Nome do arquivo .tsf (apenas o basename).

    Returns:
        int: Horizonte de previsao em numero de passos.

    Raises:
        KeyError: Se o arquivo nao estiver mapeado em HORIZONTES.
    """
    if nome_arquivo not in HORIZONTES:
        raise KeyError(
            f"Arquivo '{nome_arquivo}' nao encontrado em HORIZONTES. "
            f"Adicione-o manualmente em src/config.py."
        )
    return HORIZONTES[nome_arquivo]


def resolver_season_length(nome_arquivo: str) -> int:
    """
    Retorna o comprimento sazonal definido para um dado arquivo .tsf.

    Parametros:
        nome_arquivo (str): Nome do arquivo .tsf (apenas o basename).

    Returns:
        int: Comprimento sazonal (numero de observacoes por ciclo).

    Raises:
        KeyError: Se o arquivo nao estiver mapeado em SEASON_LENGTHS.
    """
    if nome_arquivo not in SEASON_LENGTHS:
        raise KeyError(
            f"Arquivo '{nome_arquivo}' nao encontrado em SEASON_LENGTHS. "
            f"Adicione-o manualmente em src/config.py."
        )
    return SEASON_LENGTHS[nome_arquivo]
