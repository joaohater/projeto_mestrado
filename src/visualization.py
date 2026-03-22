"""
visualization.py

Modulo de visualizacao do pipeline de forecasting.

Responsabilidades:
    - Plotar grafico individual de predito vs real
    - Montar grade automatica de graficos por dataset a partir das
      previsoes salvas em results/predictions/

Como usar:
    from src.visualization import plot_forecast, plot_grid_dataset

    # Grafico individual
    plot_forecast(
        y_true=y_test,
        y_pred=preds,
        modelo='xgboost',
        dataset='bitcoin',
        config='lags=7',
    )

    # Grade automatica por dataset (todos os modelos)
    plot_grid_dataset(
        dataset='bitcoin',
        predictions_dir='results/predictions/',
    )

    # Grade e salva em arquivo
    plot_grid_dataset(
        dataset='bitcoin',
        predictions_dir='results/predictions/',
        salvar=True,
        output_dir='results/figures/',
    )

Localizacao no projeto:
    src/visualization.py

Autor: [Seu Nome]
Dissertacao: [Titulo da Dissertacao]
Data: 2024
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(name)s - %(message)s',
)


# ---------------------------------------------------------------------------
# Utilitarios internos
# ---------------------------------------------------------------------------

def _montar_titulo(
    modelo:  str | None,
    dataset: str | None,
    config:  str | None,
) -> str:
    """Monta titulo a partir dos metadados do experimento."""
    partes = []
    if modelo:
        partes.append(modelo.replace('_', ' ').title())
    if dataset:
        nome = dataset.replace('_dataset', '').replace('_', ' ').title()
        partes.append(nome)
    titulo = ' — '.join(partes) if partes else 'Predicted vs Actual'
    if config:
        titulo += f' | {config}'
    return titulo


def _sanitizar_nome(texto: str) -> str:
    """Sanitiza string para uso como nome de arquivo."""
    for char in [' ', '/', '\\', '(', ')', '=', '|', '-']:
        texto = texto.replace(char, '_')
    while '__' in texto:
        texto = texto.replace('__', '_')
    return texto.strip('_')


def _parse_nome_arquivo(nome: str) -> dict[str, str]:
    """
    Extrai metadados do nome do arquivo de previsoes.

    Formato esperado:
        <dataset>_<modelo>_<tipo_feature>_<config>.csv

    Suporta tipos compostos com '+': lags+intervalo, lags+percentile
    e tipos simples: lags, intervalo, estatistico, percentile
    """
    stem = Path(nome).stem

    # Tipos conhecidos — ordem importa: compostos primeiro
    tipos_conhecidos = [
        'lags+intervalo',
        'lags+percentile',
        'percentile',
        'intervalo',
        'estatistico',
        'lags',
    ]

    tipo_encontrado = None
    idx_tipo_char   = None

    for tipo in tipos_conhecidos:
        marcador = f'_{tipo}_'
        pos = stem.find(marcador)
        if pos != -1:
            tipo_encontrado = tipo
            idx_tipo_char   = pos
            break

    if tipo_encontrado is None:
        return {'dataset': stem, 'modelo': '', 'tipo_feature': '', 'config': ''}

    # Tudo antes do marcador: dataset + modelo
    prefixo = stem[:idx_tipo_char]
    # Tudo depois do marcador: config
    sufixo  = stem[idx_tipo_char + len(tipo_encontrado) + 2:]
    config  = sufixo if sufixo else '-'

    # Modelos compostos conhecidos
    modelos_compostos = [
        'random_forest', 'seasonal_naive', 'simple_es',
        'damped_es', 'linear_regression',
    ]

    modelo  = ''
    dataset = prefixo

    for m in modelos_compostos:
        if prefixo.endswith(f'_{m}'):
            modelo  = m
            dataset = prefixo[:-(len(m) + 1)]
            break

    if not modelo:
        partes  = prefixo.rsplit('_', 1)
        dataset = partes[0] if len(partes) > 1 else prefixo
        modelo  = partes[1] if len(partes) > 1 else ''

    return {
        'dataset':      dataset,
        'modelo':       modelo,
        'tipo_feature': tipo_encontrado,
        'config':       config,
    }


# ---------------------------------------------------------------------------
# Grafico individual
# ---------------------------------------------------------------------------

def plot_forecast(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    modelo:       str | None = None,
    dataset:      str | None = None,
    config:       str | None = None,
    titulo:       str | None = None,
    xlabel:       str = 'Passos',
    ylabel:       str = 'Valor',
    salvar:       bool = False,
    output_dir:   str = 'results/figures/',
    nome_arquivo: str | None = None,
    figsize:      tuple[int, int] = (12, 4),
) -> None:
    """
    Plota grafico de valores preditos vs valores reais.

    O titulo e gerado automaticamente a partir de modelo, dataset e config.
    Se titulo for informado manualmente, ele sobrepoe a geracao automatica.

    Parametros:
        y_true       (np.ndarray): Valores reais do conjunto de teste.
        y_pred       (np.ndarray): Valores previstos pelo modelo.
        modelo       (str | None): Nome do modelo (ex: 'xgboost').
        dataset      (str | None): Nome do dataset (ex: 'bitcoin').
        config       (str | None): Configuracao (ex: 'lags=7', 'w=14').
        titulo       (str | None): Titulo manual — sobrepoe a geracao
            automatica quando informado.
        xlabel       (str): Rotulo do eixo X. Padrao: 'Passos'.
        ylabel       (str): Rotulo do eixo Y. Padrao: 'Valor'.
        salvar       (bool): Se True, salva o grafico em arquivo.
        output_dir   (str): Diretorio de saida quando salvar=True.
        nome_arquivo (str | None): Nome do arquivo. Se None, gerado
            automaticamente a partir do titulo.
        figsize      (tuple): Tamanho da figura em polegadas.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    titulo_final = titulo if titulo else _montar_titulo(modelo, dataset, config)
    passos = np.arange(1, len(y_true) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(passos, y_true, label='Real',    color='#2c7bb6', linewidth=1.8)
    ax.plot(passos, y_pred, label='Predito', color='#d7191c', linewidth=1.8,
            linestyle='--')
    ax.set_title(titulo_final, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if salvar:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if nome_arquivo is None:
            nome_arquivo = _sanitizar_nome(titulo_final) + '.png'
        caminho = Path(output_dir) / nome_arquivo
        fig.savefig(caminho, dpi=150, bbox_inches='tight')
        logger.info(f"Grafico salvo em: {caminho}")

    plt.show()


# ---------------------------------------------------------------------------
# Grade automatica por dataset
# ---------------------------------------------------------------------------

def plot_grid_dataset(
    dataset:         str,
    predictions_dir: str = 'results/predictions/',
    tipo_feature:    str | None = None,
    config:          str | None = None,
    xlabel:          str = 'Passos',
    ylabel:          str = 'Valor',
    salvar:          bool = False,
    output_dir:      str = 'results/figures/',
    nome_arquivo:    str | None = None,
    figsize_por_plot: tuple[int, int] = (10, 3),
) -> None:
    """
    Monta grade de graficos predito vs real para todos os modelos
    de um dataset, lendo automaticamente de results/predictions/.

    Cada arquivo de previsao salvo por save_result() vira um painel
    na grade. Os paineis sao organizados em 2 colunas.

    Parametros:
        dataset          (str): Nome do dataset a visualizar (ex: 'bitcoin').
        predictions_dir  (str): Diretorio com os CSVs de previsoes.
        tipo_feature     (str | None): Filtra por tipo ('lags', 'intervalo',
            'estatistico'). Se None, exibe todos.
        config           (str | None): Filtra por configuracao (ex: 'n7').
            Se None, exibe todos.
        xlabel           (str): Rotulo do eixo X.
        ylabel           (str): Rotulo do eixo Y.
        salvar           (bool): Se True, salva a grade em arquivo.
        output_dir       (str): Diretorio de saida quando salvar=True.
        nome_arquivo     (str | None): Nome do arquivo. Se None, gerado
            automaticamente.
        figsize_por_plot (tuple): Tamanho de cada painel (largura, altura).
    """
    pred_path = Path(predictions_dir)
    if not pred_path.exists():
        print(f"Diretorio '{predictions_dir}' nao encontrado.")
        return

    # Busca arquivos do dataset — usa iterdir em vez de glob
    # para suportar '+' no nome (lags+intervalo, lags+percentile)
    arquivos = sorted([
        f for f in pred_path.iterdir()
        if f.name.startswith(f"{dataset}_") and f.name.endswith('.csv')
    ])

    if not arquivos:
        print(f"Nenhuma previsao encontrada para o dataset '{dataset}'.")
        return

    # Filtra por tipo_feature e config se informados
    if tipo_feature:
        arquivos = [a for a in arquivos if f'_{tipo_feature}_' in a.name]
    if config:
        config_slug = config.replace('=', '').replace(' ', '_')
        arquivos = [a for a in arquivos if a.stem.endswith(f'_{config_slug}')]

    if not arquivos:
        print(f"Nenhuma previsao encontrada com os filtros aplicados.")
        return

    n_plots = len(arquivos)
    n_cols  = 2
    n_rows  = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_por_plot[0] * n_cols, figsize_por_plot[1] * n_rows),
    )
    axes = np.array(axes).flatten()

    dataset_titulo = dataset.replace('_dataset', '').replace('_', ' ').title()
    fig.suptitle(
        f'Predicted vs Actual — {dataset_titulo}',
        fontsize=14, fontweight='bold', y=1.01,
    )

    for i, arquivo in enumerate(arquivos):
        meta   = _parse_nome_arquivo(arquivo.name)
        df     = pd.read_csv(arquivo)
        y_true = df['y_true'].values
        y_pred = df['y_pred'].values
        passos = np.arange(1, len(y_true) + 1)

        ax = axes[i]
        ax.plot(passos, y_true, label='Real',    color='#2c7bb6', linewidth=1.5)
        ax.plot(passos, y_pred, label='Predito', color='#d7191c', linewidth=1.5,
                linestyle='--')

        subtitulo = meta['modelo'].replace('_', ' ').title()
        if meta['tipo_feature']:
            subtitulo += f" | {meta['tipo_feature']}"
        if meta['config'] and meta['config'] != '-':
            subtitulo += f"={meta['config']}"
        ax.set_title(subtitulo, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    # Oculta paineis vazios se n_plots for impar
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if salvar:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if nome_arquivo is None:
            filtro = f"_{tipo_feature}" if tipo_feature else ""
            filtro += f"_{config.replace('=','')}" if config else ""
            nome_arquivo = f"grid_{dataset}{filtro}.png"
        caminho = Path(output_dir) / nome_arquivo
        fig.savefig(caminho, dpi=150, bbox_inches='tight')
        logger.info(f"Grade salva em: {caminho}")

    plt.show()
