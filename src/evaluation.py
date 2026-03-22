"""
evaluation.py

Modulo de avaliacao de modelos do pipeline de forecasting.

Responsabilidades:
    - Calcular metricas de desempenho (MAE, MAPE, MSE, RMSE)
    - Registrar resultados num CSV central de forma incremental (thread-safe)
    - Salvar previsoes em results/predictions/ para uso na visualizacao
    - Exibir tabelas comparativas e rankings agregados

Localizacao no projeto:
    src/evaluation.py
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import METRICAS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(name)s - %(message)s',
)

# Lock global — garante escrita thread-safe no CSV central quando
# multiplos modelos rodam em paralelo via joblib
_csv_lock = threading.Lock()

_COLUNAS_META = ['dataset', 'modelo', 'tipo_feature', 'config']


# ---------------------------------------------------------------------------
# Calculos de metricas
# ---------------------------------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    mask = y_true != 0
    if mask.sum() == 0:
        logger.warning('MAPE nao pode ser calculado: todos os valores reais sao zero.')
        return None
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(_mse(y_true, y_pred)))


_FUNCOES_METRICAS = {
    'mae':  _mae,
    'mape': _mape,
    'mse':  _mse,
    'rmse': _rmse,
}


def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Optional[float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    resultados = {}
    for nome, ativo in METRICAS.items():
        if not ativo:
            continue
        fn = _FUNCOES_METRICAS.get(nome)
        if fn is None:
            logger.warning(
                f"Metrica '{nome}' definida em METRICAS nao tem implementacao "
                f"em evaluation.py. Ignorando."
            )
            continue
        resultados[nome] = fn(y_true, y_pred)

    return resultados


# ---------------------------------------------------------------------------
# Registro de resultados e previsoes
# ---------------------------------------------------------------------------

@dataclass
class ResultadoExperimento:
    """Representa o resultado de um experimento (dataset x modelo x config)."""
    dataset:      str
    modelo:       str
    tipo_feature: str
    config:       str
    metricas:     dict[str, Optional[float]] = field(default_factory=dict)

    def print_summary(self) -> None:
        print(f"\n{'=' * 55}")
        print(f"  Dataset      : {self.dataset}")
        print(f"  Modelo       : {self.modelo}")
        print(f"  Tipo feature : {self.tipo_feature}")
        print(f"  Config       : {self.config}")
        print(f"  {'Metrica':<10} {'Valor':>15}")
        print(f"  {'-' * 25}")
        for nome, valor in self.metricas.items():
            v = f"{valor:.6f}" if valor is not None else "N/A"
            print(f"  {nome.upper():<10} {v:>15}")
        print(f"{'=' * 55}\n")


def _salvar_predicoes(
    dataset:         str,
    modelo:          str,
    tipo_feature:    str,
    config:          str,
    y_true:          np.ndarray,
    y_pred:          np.ndarray,
    predictions_dir: str,
) -> None:
    """Salva y_true e y_pred num CSV em results/predictions/."""
    Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    config_slug = config.replace('=', '').replace(' ', '_')
    nome = f"{dataset}_{modelo}_{tipo_feature}_{config_slug}.csv"
    caminho = Path(predictions_dir) / nome
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).to_csv(caminho, index=False)
    logger.info(f"Previsoes salvas em: {caminho}")


def save_result(
    dataset:         str,
    modelo:          str,
    tipo_feature:    str,
    config:          str,
    y_true:          np.ndarray,
    y_pred:          np.ndarray,
    output_path:     str = 'results/resultados.csv',
    predictions_dir: str = 'results/predictions/',
    print_result:    bool = True,
) -> ResultadoExperimento:
    """
    Calcula as metricas, appenda no CSV central (thread-safe) e salva as previsoes.

    O _csv_lock garante que escritas simultaneas de workers paralelos nao
    corrompam o CSV. O calculo de metricas e o salvamento de previsoes
    ocorrem fora do lock para maximizar o paralelismo.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Calculo de metricas — fora do lock, sem I/O
    metricas = calcular_metricas(y_true, y_pred)

    resultado = ResultadoExperimento(
        dataset=dataset,
        modelo=modelo,
        tipo_feature=tipo_feature,
        config=config,
        metricas=metricas,
    )

    linha = {
        'dataset':      dataset,
        'modelo':       modelo,
        'tipo_feature': tipo_feature,
        'config':       config,
        **metricas,
    }

    # Salva previsoes — cada modelo tem seu proprio arquivo, sem conflito
    _salvar_predicoes(
        dataset, modelo, tipo_feature, config,
        y_true, y_pred, predictions_dir,
    )

    # Escrita no CSV central — protegida pelo lock
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with _csv_lock:
        escrever_cabecalho = not os.path.exists(output_path)
        pd.DataFrame([linha]).to_csv(
            output_path,
            mode='a',
            header=escrever_cabecalho,
            index=False,
        )

    logger.info(
        f"Resultado salvo: {dataset} | {modelo} | {tipo_feature} | {config}"
    )

    if print_result:
        resultado.print_summary()

    return resultado


# ---------------------------------------------------------------------------
# Relatorios comparativos
# ---------------------------------------------------------------------------

def print_summary(
    csv_path:    str,
    dataset:     Optional[str] = None,
    ordenar_por: str = 'mae',
) -> None:
    """Exibe tabela comparativa dos resultados registrados no CSV."""
    if not os.path.exists(csv_path):
        print(f"Arquivo '{csv_path}' nao encontrado. Nenhum resultado salvo ainda.")
        return

    df = pd.read_csv(csv_path)

    if dataset is not None:
        df = df[df['dataset'] == dataset]
        if df.empty:
            print(f"Nenhum resultado encontrado para o dataset '{dataset}'.")
            return

    if ordenar_por in df.columns:
        df = df.sort_values(ordenar_por)

    print(f"\n{'=' * 70}")
    print(f"  RESULTADOS COMPARATIVOS")
    if dataset:
        print(f"  Dataset: {dataset}")
    print(f"{'=' * 70}")
    print(df.to_string(index=False))
    print(f"{'=' * 70}\n")


def print_ranking(csv_path: str, metrica: str = 'mae') -> None:
    """
    Exibe ranking agregado dos modelos considerando todos os datasets.

    Para cada (dataset, modelo) seleciona a melhor configuracao (menor
    valor da metrica), depois rankeia dentro de cada dataset e calcula
    a posicao media entre datasets.

    Posicao media menor = modelo mais consistente entre os datasets.
    """
    if not os.path.exists(csv_path):
        print(f"Arquivo '{csv_path}' nao encontrado. Nenhum resultado salvo ainda.")
        return

    df = pd.read_csv(csv_path)

    if metrica not in df.columns:
        print(f"Metrica '{metrica}' nao encontrada no CSV.")
        return

    # Melhor configuracao de cada modelo por dataset
    # (modelos estatisticos tem config='-', ML tem n=7/14/30 ou w=7/14/30)
    melhor = (
        df.groupby(['dataset', 'modelo'])[metrica]
        .min()
        .reset_index()
    )

    # Rank dentro de cada dataset (1 = melhor)
    melhor['rank'] = melhor.groupby('dataset')[metrica].rank(
        ascending=True, method='min'
    )

    # Posicao media entre todos os datasets
    ranking = (
        melhor.groupby('modelo')['rank']
        .mean()
        .reset_index()
        .rename(columns={'rank': 'posicao_media'})
        .sort_values('posicao_media')
        .reset_index(drop=True)
    )

    # Numero de datasets em que o modelo ficou em 1o lugar
    primeiro = (
        melhor[melhor['rank'] == 1]
        .groupby('modelo')
        .size()
        .reset_index(name='n_vezes_1o')
    )
    ranking = ranking.merge(primeiro, on='modelo', how='left')
    ranking['n_vezes_1o'] = ranking['n_vezes_1o'].fillna(0).astype(int)

    print(f"\n{'=' * 60}")
    print(f"  RANKING AGREGADO  |  Metrica base: {metrica.upper()}")
    print(f"  (melhor configuracao por modelo e dataset)")
    print(f"{'=' * 60}")
    print(ranking.to_string(index=False))
    print(f"{'=' * 60}\n")
