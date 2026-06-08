"""
averaging.py — Balanceo de grupos + Grand Average.

Réplica EXACTA de la lógica del Script 04 v2:

  • igualar_grupos:  selección aleatoria de N alcohólicos con
      rng = np.random.default_rng(seed); rng.choice(alc, size=N, replace=False)
    sobre los sujetos alcohólicos ordenados. (Verificado: reproduce
    bit-a-bit 'sujetos_seleccionados.csv' para N=45, SEED=42.)

  • calcular_grand_average:  promedio de los PE individuales por grupo/muestra
      groupby(...).agg(mean, std, count);  sem = std / sqrt(count)
    (Verificado: max|diff| = 0.0 contra 'eeg_GA_homogeneo.parquet' a N=45.)

IMPORTANTE: el slider NO re-promedia trials crudos. Re-balancea el Grand
Average a partir de los PE individuales ya guardados por el Script 04.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def seleccionar_alcoholicos(pe: pd.DataFrame, n: int, seed: int) -> list[str]:
    """
    Selecciona aleatoriamente 'n' sujetos alcohólicos, replicando
    igualar_grupos() del Script 04 v2.

    El Script toma df[grupo=='alcoholic'].sujeto.unique() del parquet crudo,
    que está ordenado; aquí ordenamos explícitamente para garantizar el mismo
    orden de entrada a rng.choice y, por lo tanto, la misma selección.

    Si n >= total de alcohólicos, se devuelven todos (sin submuestrear),
    igual que el Script ("El grupo alcohólico ya tiene <= n sujetos").
    """
    alc = np.array(sorted(pe.loc[pe["grupo"] == "alcoholic", "sujeto"].unique()))
    if n >= len(alc):
        return list(alc)
    rng = np.random.default_rng(seed)
    seleccionados = rng.choice(alc, size=n, replace=False)
    return sorted(seleccionados.tolist())


def sujetos_incluidos(pe: pd.DataFrame, n_alc: int, seed: int) -> set[str]:
    """Conjunto = TODOS los controles + los n_alc alcohólicos seleccionados."""
    controles = pe.loc[pe["grupo"] == "control", "sujeto"].unique().tolist()
    alcoholicos = seleccionar_alcoholicos(pe, n_alc, seed)
    return set(controles) | set(alcoholicos)


@dataclass
class CurvaGA:
    """Grand Average de un grupo para un canal/condición."""

    grupo: str
    tiempo_ms: np.ndarray   # (N_SAMPLES,)
    media: np.ndarray       # grand_avg_uV
    sem: np.ndarray         # error estándar entre sujetos
    n_sujetos: int


def calcular_grand_average(
    pe: pd.DataFrame,
    incluidos: set[str],
    canal: str,
    condicion: str,
) -> dict[str, CurvaGA]:
    """
    Grand Average por grupo para un canal y condición, sobre los sujetos
    'incluidos'. Replica calcular_grand_average() del Script 04:

        groupby(['grupo','muestra'])['valor_uV'].agg(mean, std, count)
        sem = std / sqrt(count)

    Devuelve  {grupo: CurvaGA}  (solo grupos con datos).
    """
    sub = pe[
        (pe["canal"] == canal)
        & (pe["condicion"] == condicion)
        & (pe["sujeto"].isin(incluidos))
    ]

    tiempo = np.asarray(config.TIEMPO_MS)
    salida: dict[str, CurvaGA] = {}

    if sub.empty:
        return salida

    agg = (
        sub.groupby(["grupo", "muestra"])["valor_uV"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    for grupo in config.GRUPOS:
        g = agg[agg["grupo"] == grupo].sort_values("muestra")
        if g.empty:
            continue
        media = g["mean"].to_numpy()
        # std de pandas usa ddof=1 (igual que el Script). count = nº de sujetos.
        n_suj = int(g["count"].iloc[0])
        sem = g["std"].to_numpy() / np.sqrt(g["count"].to_numpy())
        salida[grupo] = CurvaGA(
            grupo=grupo,
            tiempo_ms=tiempo[: len(media)],
            media=media,
            sem=np.nan_to_num(sem),     # con 1 sujeto std=NaN -> banda 0
            n_sujetos=n_suj,
        )

    return salida
