"""
feature_combiner.py

Combina diferentes tipos de features (lags, intervalo, percentil)
em um unico DataFrame para uso nos modelos de ML.

Regra: os DataFrames devem ter o mesmo tamanho e mesma ordem temporal.
O merge e feito por posicao (reset_index) — nao por timestamp —
para evitar problemas com formatos de data entre datasets.

Uso tipico:
    from feature_combiner import combine_features

    df_combined = combine_features(df_lags, df_intervalo)
    df_combined = combine_features(df_lags, df_percentil)
    df_combined = combine_features(df_lags, df_intervalo, df_percentil)

Localizacao no projeto:
    src/features/feature_combiner.py
"""

import pandas as pd


def combine_features(*dataframes: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
    """
    Combina dois ou mais DataFrames de features em um unico DataFrame.

    Mantém as colunas de identificacao (timestamp, series_name, value)
    do primeiro DataFrame e concatena as colunas de features dos demais.

    Parâmetros
    ----------
    *dataframes : dois ou mais DataFrames com features geradas
                  (lags, intervalo, percentil)
    value_col   : nome da coluna de valores (default: 'value')

    Retorna
    -------
    DataFrame combinado com todas as features

    Raises
    ------
    ValueError : se os DataFrames tiverem tamanhos diferentes
    """
    if len(dataframes) < 2:
        raise ValueError("Forneça pelo menos dois DataFrames para combinar.")

    # Verifica tamanhos
    tamanhos = [len(df) for df in dataframes]
    if len(set(tamanhos)) > 1:
        raise ValueError(
            f"DataFrames com tamanhos diferentes: {tamanhos}. "
            "Certifique-se de usar o mesmo split e mesma configuracao de janela."
        )

    # Colunas de identificacao — vem do primeiro DataFrame
    id_cols = ['series_name', 'timestamp', value_col]
    id_cols = [c for c in id_cols if c in dataframes[0].columns]

    result = dataframes[0].reset_index(drop=True)

    # Adiciona features dos demais DataFrames
    for df in dataframes[1:]:
        df = df.reset_index(drop=True)

        # Colunas de feature — exclui id_cols para nao duplicar
        feat_cols = [c for c in df.columns if c not in id_cols]
        result = pd.concat([result, df[feat_cols]], axis=1)

    return result


def get_feature_columns(df: pd.DataFrame, value_col: str = 'value') -> list:
    """
    Retorna apenas as colunas de features de um DataFrame combinado,
    excluindo colunas de identificacao.

    Util para extrair X_train e X_test de forma consistente.

    Parâmetros
    ----------
    df        : DataFrame combinado
    value_col : nome da coluna de valores

    Retorna
    -------
    Lista de nomes de colunas de features
    """
    exclude = {'series_name', 'state', 'timestamp', value_col}
    return [c for c in df.columns if c not in exclude]
