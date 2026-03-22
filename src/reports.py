"""
reports.py

Modulo de relatorios e organizacao dos resultados do pipeline de forecasting.

Responsabilidades (itens 4 e 5 do feedback do professor):
    - Organizar resultados por dataset e por metodo (tipo_feature)
    - Destacar MAPE e MAE como metricas principais
    - Gerar tabelas comparativas prontas para a dissertacao
    - Gerar heatmap visual de MAPE (datasets x modelos)
    - Gerar grafico de barras comparando metodos por dataset

Funcoes publicas:
    tabela_por_dataset(csv_path, metrica)
        → melhor resultado de cada modelo por dataset, ordenado por MAPE

    tabela_por_metodo(csv_path, metrica)
        → ranking por tipo de feature (lags, intervalo, percentile, etc.)

    heatmap_mape(csv_path, metrica, salvar, output_dir)
        → heatmap datasets x modelos com a melhor config de cada combinacao

    barras_por_dataset(csv_path, dataset, metrica, salvar, output_dir)
        → barras horizontais comparando todos os modelos num dataset

    resumo_geral(csv_path, salvar, output_dir)
        → executa todas as visualizacoes em sequencia

Uso no notebook:
    import sys
    sys.path.append('../src')

    from reports import resumo_geral, tabela_por_dataset, heatmap_mape

    # Relatorio completo
    resumo_geral('../results/resultados.csv', salvar=True)

    # So a tabela de um dataset especifico
    tabela_por_dataset('../results/resultados.csv', dataset='bitcoin_dataset_without_missing_values')

    # Heatmap interativo
    heatmap_mape('../results/resultados.csv')

Localizacao no projeto:
    src/reports.py

Autor: [Seu Nome]
Dissertacao: [Titulo da Dissertacao]
Data: 2024
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(name)s - %(message)s',
)

# ---------------------------------------------------------------------------
# Constantes de estilo
# ---------------------------------------------------------------------------

# Ordem de exibicao dos tipos de feature — da mais simples a mais complexa
_ORDEM_TIPOS = [
    'estatistico',
    'lags',
    'intervalo',
    'percentile',
    'lags+intervalo',
    'lags+percentile',
]

# Paleta de cores por tipo de feature (para graficos comparativos)
_CORES_TIPO = {
    'estatistico':     '#4e79a7',
    'lags':            '#f28e2b',
    'intervalo':       '#59a14f',
    'percentile':      '#e15759',
    'lags+intervalo':  '#76b7b2',
    'lags+percentile': '#b07aa1',
}

# Nome legivel dos datasets (sem sufixos redundantes)
_NOME_DATASET = {
    'australian_electricity_demand_dataset': 'Australian',
    'bitcoin_dataset_without_missing_values': 'Bitcoin',
    'saugeenday_dataset': 'Saugeen',
    'sunspot_dataset_without_missing_values': 'Sunspot',
    'us_births_dataset': 'US Births',
}


# ---------------------------------------------------------------------------
# Utilitarios internos
# ---------------------------------------------------------------------------

def _carregar_csv(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV de resultados e valida as colunas minimas."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Arquivo de resultados nao encontrado: '{csv_path}'\n"
            f"Execute o pipeline antes de gerar os relatorios."
        )

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()

    colunas_obrigatorias = {'dataset', 'modelo', 'tipo_feature', 'config', 'mae', 'mape'}
    faltando = colunas_obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(
            f"Colunas ausentes no CSV de resultados: {faltando}\n"
            f"Colunas encontradas: {list(df.columns)}"
        )

    return df


def _nome_legivel_dataset(dataset: str) -> str:
    """Converte nome interno do dataset para nome legivel."""
    return _NOME_DATASET.get(dataset, dataset.replace('_', ' ').title())


def _melhor_por_grupo(
    df: pd.DataFrame,
    grupo: list[str],
    metrica: str,
) -> pd.DataFrame:
    """
    Para cada combinacao de 'grupo', retorna a linha com o menor valor
    da metrica. Util para selecionar a melhor config de cada modelo.
    """
    idx = df.groupby(grupo)[metrica].idxmin()
    return df.loc[idx].reset_index(drop=True)


def _formatar_mape(valor: float | None) -> str:
    """Formata MAPE com 2 casas decimais e simbolo %."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return 'N/A'
    return f"{valor:.2f}%"


def _formatar_mae(valor: float | None) -> str:
    """Formata MAE com 4 casas decimais."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return 'N/A'
    return f"{valor:.4f}"


# ---------------------------------------------------------------------------
# 1. Tabela por dataset
# ---------------------------------------------------------------------------

def tabela_por_dataset(
    csv_path: str,
    dataset:  Optional[str] = None,
    metrica:  str = 'mape',
) -> pd.DataFrame:
    """
    Exibe e retorna tabela com o melhor resultado de cada modelo por dataset.

    Para cada (dataset, modelo), seleciona a configuracao com menor valor
    da metrica principal (padrao: MAPE). Exibe MAPE e MAE lado a lado
    conforme solicitado pelo professor.

    Parametros:
        csv_path (str): Caminho para o CSV de resultados.
        dataset  (str | None): Se informado, filtra para um dataset especifico.
        metrica  (str): Metrica usada para selecionar a melhor config.
            Padrao: 'mape'. Opcoes: 'mape', 'mae', 'rmse', 'mse'.

    Retorna:
        pd.DataFrame com colunas:
            dataset, modelo, tipo_feature, config, mape, mae
        ordenado por dataset e mape (crescente).

    Exemplo:
        df = tabela_por_dataset('../results/resultados.csv')
        df = tabela_por_dataset('../results/resultados.csv', dataset='bitcoin_dataset_without_missing_values')
    """
    df = _carregar_csv(csv_path)

    if dataset is not None:
        df = df[df['dataset'] == dataset]
        if df.empty:
            print(f"Nenhum resultado encontrado para o dataset '{dataset}'.")
            return pd.DataFrame()

    # Melhor config por (dataset, modelo) segundo a metrica principal
    melhor = _melhor_por_grupo(df, ['dataset', 'modelo'], metrica)

    # Colunas de saida: prioriza MAPE e MAE conforme feedback do professor
    colunas_saida = ['dataset', 'modelo', 'tipo_feature', 'config', 'mape', 'mae']
    colunas_saida = [c for c in colunas_saida if c in melhor.columns]

    resultado = (
        melhor[colunas_saida]
        .sort_values(['dataset', metrica])
        .reset_index(drop=True)
    )

    # Exibicao formatada por dataset
    datasets_unicos = resultado['dataset'].unique()

    for ds in datasets_unicos:
        bloco = resultado[resultado['dataset'] == ds].copy()
        nome_ds = _nome_legivel_dataset(ds)

        print(f"\n{'=' * 70}")
        print(f"  Dataset: {nome_ds}")
        print(f"  Ordenado por: {metrica.upper()} (melhor configuracao por modelo)")
        print(f"{'=' * 70}")

        bloco_exib = bloco.copy()
        bloco_exib['mape'] = bloco_exib['mape'].apply(_formatar_mape)
        bloco_exib['mae']  = bloco_exib['mae'].apply(_formatar_mae)
        bloco_exib = bloco_exib.drop(columns=['dataset'])

        print(bloco_exib.to_string(index=False))

    return resultado


# ---------------------------------------------------------------------------
# 2. Tabela por metodo (tipo_feature)
# ---------------------------------------------------------------------------

def tabela_por_metodo(
    csv_path: str,
    metrica:  str = 'mape',
) -> pd.DataFrame:
    """
    Exibe e retorna tabela comparando os metodos (tipos de feature) entre si.

    Para cada tipo_feature, calcula a media de MAPE e MAE entre todos os
    datasets e modelos (usando a melhor config de cada combinacao).
    Permite avaliar se lags, intervalo ou percentile geram features melhores.

    Parametros:
        csv_path (str): Caminho para o CSV de resultados.
        metrica  (str): Metrica usada para selecionar a melhor config
            dentro de cada (dataset, modelo, tipo_feature).

    Retorna:
        pd.DataFrame com colunas:
            tipo_feature, mape_medio, mae_medio, mape_min, mape_max, n_experimentos
        ordenado por mape_medio crescente.

    Exemplo:
        df = tabela_por_metodo('../results/resultados.csv')
    """
    df = _carregar_csv(csv_path)

    # Melhor config por (dataset, modelo, tipo_feature)
    melhor = _melhor_por_grupo(df, ['dataset', 'modelo', 'tipo_feature'], metrica)

    # Agrega por tipo_feature
    agg = (
        melhor.groupby('tipo_feature')
        .agg(
            mape_medio=('mape', 'mean'),
            mae_medio=('mae',  'mean'),
            mape_min=('mape',  'min'),
            mape_max=('mape',  'max'),
            n_experimentos=('mape', 'count'),
        )
        .reset_index()
        .sort_values('mape_medio')
    )

    # Reordena linhas conforme ordem logica do pipeline
    ordem = {t: i for i, t in enumerate(_ORDEM_TIPOS)}
    agg['_ordem'] = agg['tipo_feature'].map(ordem).fillna(99)
    agg = agg.sort_values('mape_medio').drop(columns=['_ordem'])

    print(f"\n{'=' * 75}")
    print(f"  COMPARACAO POR METODO (tipo de feature)")
    print(f"  Metrica de selecao da melhor config: {metrica.upper()}")
    print(f"  Valores: media entre todos os datasets e modelos")
    print(f"{'=' * 75}")

    agg_exib = agg.copy()
    agg_exib['mape_medio'] = agg_exib['mape_medio'].apply(_formatar_mape)
    agg_exib['mae_medio']  = agg_exib['mae_medio'].apply(_formatar_mae)
    agg_exib['mape_min']   = agg_exib['mape_min'].apply(_formatar_mape)
    agg_exib['mape_max']   = agg_exib['mape_max'].apply(_formatar_mape)

    print(agg_exib.to_string(index=False))
    print(f"{'=' * 75}\n")

    return agg


# ---------------------------------------------------------------------------
# 3. Heatmap MAPE — datasets x modelos
# ---------------------------------------------------------------------------

def heatmap_mape(
    csv_path:   str,
    metrica:    str = 'mape',
    salvar:     bool = False,
    output_dir: str = 'results/figures/',
    figsize:    tuple[int, int] = (14, 6),
) -> None:
    """
    Gera heatmap visual de MAPE (ou outra metrica) com datasets no eixo Y
    e modelos no eixo X.

    Para cada celula (dataset, modelo), usa a melhor configuracao encontrada
    em todos os tipos de feature — ou seja, o melhor resultado absoluto de
    cada modelo em cada dataset.

    Parametros:
        csv_path   (str): Caminho para o CSV de resultados.
        metrica    (str): Metrica a exibir no heatmap. Padrao: 'mape'.
        salvar    (bool): Se True, salva o grafico em arquivo PNG.
        output_dir (str): Diretorio de saida quando salvar=True.
        figsize   (tuple): Tamanho da figura em polegadas.

    Exemplo:
        heatmap_mape('../results/resultados.csv', salvar=True)
    """
    df = _carregar_csv(csv_path)

    # Melhor resultado absoluto por (dataset, modelo)
    melhor = _melhor_por_grupo(df, ['dataset', 'modelo'], metrica)

    # Pivota para matriz datasets x modelos
    pivot = melhor.pivot_table(
        index='dataset', columns='modelo', values=metrica, aggfunc='min'
    )

    # Nomes legíveis no eixo Y
    pivot.index = [_nome_legivel_dataset(d) for d in pivot.index]

    fig, ax = plt.subplots(figsize=figsize)

    # Colormap: verde = bom (baixo), vermelho = ruim (alto)
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r')

    # Eixos
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [c.replace('_', ' ').title() for c in pivot.columns],
        rotation=35, ha='right', fontsize=10,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Anotacoes nas celulas
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                texto = f"{val:.1f}%" if metrica == 'mape' else f"{val:.2f}"
                # Texto branco em celulas muito escuras, preto nas claras
                cor_fundo = im.cmap(im.norm(val))
                luminancia = 0.299*cor_fundo[0] + 0.587*cor_fundo[1] + 0.114*cor_fundo[2]
                cor_texto  = 'white' if luminancia < 0.5 else 'black'
                ax.text(j, i, texto, ha='center', va='center',
                        fontsize=8.5, color=cor_texto, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metrica.upper(), fontsize=11)

    ax.set_title(
        f'Heatmap {metrica.upper()} — Melhor resultado por modelo e dataset\n'
        f'(menor valor = melhor previsao)',
        fontsize=13, fontweight='bold', pad=14,
    )

    plt.tight_layout()

    if salvar:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        caminho = Path(output_dir) / f'heatmap_{metrica}.png'
        fig.savefig(caminho, dpi=150, bbox_inches='tight')
        logger.info(f"Heatmap salvo em: {caminho}")

    plt.show()


# ---------------------------------------------------------------------------
# 4. Grafico de barras por dataset
# ---------------------------------------------------------------------------

def barras_por_dataset(
    csv_path:   str,
    dataset:    str,
    metrica:    str = 'mape',
    top_n:      int = 20,
    salvar:     bool = False,
    output_dir: str = 'results/figures/',
) -> None:
    """
    Gera grafico de barras horizontais mostrando os N melhores resultados
    para um dataset especifico, coloridos por tipo de feature.

    Util para visualizar rapidamente quais combinacoes modelo+feature
    funcionaram melhor em cada dataset.

    Parametros:
        csv_path   (str): Caminho para o CSV de resultados.
        dataset    (str): Nome do dataset (chave interna, ex:
            'bitcoin_dataset_without_missing_values').
        metrica    (str): Metrica a ordenar. Padrao: 'mape'.
        top_n      (int): Numero de barras a exibir. Padrao: 20.
        salvar    (bool): Se True, salva o grafico.
        output_dir (str): Diretorio de saida quando salvar=True.

    Exemplo:
        barras_por_dataset(
            '../results/resultados.csv',
            dataset='bitcoin_dataset_without_missing_values',
            top_n=15,
            salvar=True,
        )
    """
    df = _carregar_csv(csv_path)
    df = df[df['dataset'] == dataset]

    if df.empty:
        print(f"Nenhum resultado encontrado para '{dataset}'.")
        return

    # Melhor config por (modelo, tipo_feature)
    melhor = _melhor_por_grupo(df, ['modelo', 'tipo_feature'], metrica)
    melhor = melhor.sort_values(metrica).head(top_n)

    # Rotulo: "modelo | tipo_feature | config"
    melhor['rotulo'] = (
        melhor['modelo'].str.replace('_', ' ').str.title()
        + ' | ' + melhor['tipo_feature']
        + ' | ' + melhor['config']
    )

    cores = melhor['tipo_feature'].map(_CORES_TIPO).fillna('#aaaaaa')
    nome_ds = _nome_legivel_dataset(dataset)

    altura = max(5, len(melhor) * 0.45)
    fig, ax = plt.subplots(figsize=(11, altura))

    barras = ax.barh(
        melhor['rotulo'], melhor[metrica],
        color=cores, edgecolor='white', height=0.7,
    )

    # Anotacao do valor em cada barra
    for barra, val in zip(barras, melhor[metrica]):
        texto = f"{val:.2f}%" if metrica == 'mape' else f"{val:.4f}"
        ax.text(
            barra.get_width() + barra.get_width() * 0.01,
            barra.get_y() + barra.get_height() / 2,
            texto, va='center', ha='left', fontsize=9,
        )

    ax.invert_yaxis()
    ax.set_xlabel(metrica.upper(), fontsize=11)
    ax.set_title(
        f'Top {top_n} resultados — {nome_ds}\n'
        f'(ordenado por {metrica.upper()}, melhor config por modelo+feature)',
        fontsize=12, fontweight='bold',
    )
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    # Legenda de cores por tipo_feature
    from matplotlib.patches import Patch
    tipos_presentes = melhor['tipo_feature'].unique()
    handles = [
        Patch(facecolor=_CORES_TIPO.get(t, '#aaaaaa'), label=t)
        for t in _ORDEM_TIPOS if t in tipos_presentes
    ]
    ax.legend(handles=handles, title='Tipo de Feature',
              loc='lower right', fontsize=9, title_fontsize=9)

    plt.tight_layout()

    if salvar:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        nome_safe = dataset.replace(' ', '_')
        caminho = Path(output_dir) / f'barras_{nome_safe}_{metrica}.png'
        fig.savefig(caminho, dpi=150, bbox_inches='tight')
        logger.info(f"Grafico salvo em: {caminho}")

    plt.show()


# ---------------------------------------------------------------------------
# 5. Comparativo de metodos por dataset (linhas empilhadas)
# ---------------------------------------------------------------------------

def barras_metodos_por_dataset(
    csv_path:   str,
    metrica:    str = 'mape',
    salvar:     bool = False,
    output_dir: str = 'results/figures/',
    figsize:    tuple[int, int] = (14, 6),
) -> None:
    """
    Gera grafico de barras agrupadas comparando o MAPE medio de cada
    tipo de feature em cada dataset.

    Permite responder visualmente: "intervalo e melhor que lags no
    Sunspot? E no Australian?"

    Parametros:
        csv_path   (str): Caminho para o CSV de resultados.
        metrica    (str): Metrica a comparar. Padrao: 'mape'.
        salvar    (bool): Se True, salva o grafico.
        output_dir (str): Diretorio de saida quando salvar=True.
        figsize   (tuple): Tamanho da figura.

    Exemplo:
        barras_metodos_por_dataset('../results/resultados.csv', salvar=True)
    """
    df = _carregar_csv(csv_path)

    # Melhor config por (dataset, modelo, tipo_feature)
    melhor = _melhor_por_grupo(df, ['dataset', 'modelo', 'tipo_feature'], metrica)

    # Media por (dataset, tipo_feature)
    agg = (
        melhor.groupby(['dataset', 'tipo_feature'])[metrica]
        .mean()
        .reset_index()
    )

    # Pivota para plotar barras agrupadas
    pivot = agg.pivot(index='dataset', columns='tipo_feature', values=metrica)
    pivot.index = [_nome_legivel_dataset(d) for d in pivot.index]

    # Reordena colunas na ordem logica
    colunas_ordenadas = [t for t in _ORDEM_TIPOS if t in pivot.columns]
    pivot = pivot[colunas_ordenadas]

    x     = np.arange(len(pivot.index))
    n_col = len(pivot.columns)
    w     = 0.8 / n_col  # largura de cada barra no grupo

    fig, ax = plt.subplots(figsize=figsize)

    for i, tipo in enumerate(pivot.columns):
        offset = (i - n_col / 2 + 0.5) * w
        cor    = _CORES_TIPO.get(tipo, '#aaaaaa')
        ax.bar(x + offset, pivot[tipo], width=w * 0.9,
               label=tipo, color=cor, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel(f'{metrica.upper()} medio (%)', fontsize=11)
    ax.set_title(
        f'MAPE medio por tipo de feature e dataset\n'
        f'(media entre todos os modelos, melhor config por modelo)',
        fontsize=13, fontweight='bold',
    )
    ax.legend(title='Tipo de Feature', fontsize=9, title_fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    if salvar:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        caminho = Path(output_dir) / f'metodos_por_dataset_{metrica}.png'
        fig.savefig(caminho, dpi=150, bbox_inches='tight')
        logger.info(f"Grafico salvo em: {caminho}")

    plt.show()


# ---------------------------------------------------------------------------
# 6. Resumo geral — executa tudo em sequencia
# ---------------------------------------------------------------------------

def resumo_geral(
    csv_path:   str,
    metrica:    str = 'mape',
    salvar:     bool = False,
    output_dir: str = 'results/figures/',
) -> None:
    """
    Executa todos os relatorios em sequencia:
        1. Tabela por dataset (todos os datasets)
        2. Tabela por metodo
        3. Heatmap MAPE (datasets x modelos)
        4. Barras de metodos por dataset
        5. Barras individuais por dataset (top 15)

    Parametros:
        csv_path   (str): Caminho para o CSV de resultados.
        metrica    (str): Metrica principal. Padrao: 'mape'.
        salvar    (bool): Se True, salva todos os graficos.
        output_dir (str): Diretorio de saida dos graficos.

    Exemplo de uso no notebook:
        from reports import resumo_geral
        resumo_geral('../results/resultados.csv', salvar=True)
    """
    df = _carregar_csv(csv_path)
    datasets = df['dataset'].unique()

    print("\n" + "=" * 70)
    print("  RELATORIO GERAL — PIPELINE DE FORECASTING")
    print(f"  Total de experimentos: {len(df)}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Modelos : {df['modelo'].nunique()}")
    print(f"  Tipos de feature: {df['tipo_feature'].nunique()}")
    print("=" * 70)

    # 1. Tabela por dataset
    print("\n\n>>> [1/4] TABELA POR DATASET")
    tabela_por_dataset(csv_path, metrica=metrica)

    # 2. Tabela por metodo
    print("\n\n>>> [2/4] TABELA POR METODO")
    tabela_por_metodo(csv_path, metrica=metrica)

    # 3. Heatmap
    print("\n\n>>> [3/4] HEATMAP MAPE")
    heatmap_mape(csv_path, metrica=metrica, salvar=salvar, output_dir=output_dir)

    # 4. Barras de metodos por dataset
    print("\n\n>>> [4/4] COMPARATIVO DE METODOS POR DATASET")
    barras_metodos_por_dataset(
        csv_path, metrica=metrica, salvar=salvar, output_dir=output_dir,
    )

    # 5. Barras individuais por dataset
    for ds in datasets:
        barras_por_dataset(
            csv_path, dataset=ds, metrica=metrica,
            top_n=15, salvar=salvar, output_dir=output_dir,
        )

    print("\n Relatorio completo gerado.")
