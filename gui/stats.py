"""
stats.py — Métricas descriptivas para el panel de KPIs.

Réplica de las fórmulas descriptivas del Script 06:
  - media ± SEM por grupo y diferencia control - alcohólico, en la ventana c240.
    (SEM = SD/√n entre sujetos, igual que la banda de los Grand Average y el
     resto de las figuras de los módulos.)

Política HÍBRIDA (elegida con el usuario):
  - KPI EN VIVO: se recalcula media_c240 ± SEM sobre la cohorte actual del slider
    (n_alc + 45 ctrl) a partir de eeg_c240_extraido.csv, para que coincida con el
    Grand Average mostrado.
    (Verificado: a n=77 reproduce tabla_estadistica.csv a precisión de máquina.)
  - REFERENCIA OFICIAL: se muestran además los valores del Script 06 sobre la
    muestra completa 77 vs 45, tomados de tabla_estadistica.csv.

La métrica PRINCIPAL es la MEDIA en ventana (media_c240), nunca el máximo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


# =============================================================================
# KPI en vivo sobre la cohorte del slider
# =============================================================================
@dataclass
class KpiVivo:
    """Estadística recalculada sobre la cohorte actual (consistente con el GA)."""

    canal: str
    condicion: str
    metodo: str
    n_control: int
    n_alcoholic: int
    # Métrica principal: media en ventana c240 (220-260 ms)
    control_media: float
    control_sem: float
    alcoholic_media: float
    alcoholic_sem: float
    diferencia: float          # control - alcohólico
    # Secundarios (medias por grupo) — claramente etiquetados como tales
    control_max: float
    alcoholic_max: float
    control_lat: float
    alcoholic_lat: float
    control_auc: float
    alcoholic_auc: float
    control_media_c320: float
    alcoholic_media_c320: float


def _media_sem(vals: np.ndarray) -> tuple[float, float]:
    """Media y error estándar de la media (SEM = SD/√n) entre sujetos."""
    if len(vals) == 0:
        return float("nan"), float("nan")
    n = len(vals)
    sem = vals.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return float(vals.mean()), float(sem)


def kpi_en_vivo(
    c240: pd.DataFrame,
    metodo: str,
    canal: str,
    condicion: str,
    incluidos: set[str],
) -> KpiVivo:
    """
    Recalcula la estadística del c240 sobre la cohorte actual (incluidos),
    usando las fórmulas de los Scripts 05/06. La métrica principal es media_c240.
    """
    sub = c240[
        (c240["metodo"] == metodo)
        & (c240["canal"] == canal)
        & (c240["condicion"] == condicion)
        & (c240["sujeto"].isin(incluidos))
    ]
    ctrl = sub[sub["grupo"] == "control"]
    alc = sub[sub["grupo"] == "alcoholic"]

    c_media = ctrl["media_c240"].dropna().to_numpy()
    a_media = alc["media_c240"].dropna().to_numpy()

    mc, sc = _media_sem(c_media)
    ma, sa = _media_sem(a_media)

    def media_de(df, col):
        v = df[col].dropna().to_numpy()
        return float(v.mean()) if len(v) else float("nan")

    return KpiVivo(
        canal=canal,
        condicion=condicion,
        metodo=metodo,
        n_control=len(c_media),
        n_alcoholic=len(a_media),
        control_media=mc,
        control_sem=sc,
        alcoholic_media=ma,
        alcoholic_sem=sa,
        diferencia=mc - ma,
        control_max=media_de(ctrl, "max_c240"),
        alcoholic_max=media_de(alc, "max_c240"),
        control_lat=media_de(ctrl, "lat_max_c240"),
        alcoholic_lat=media_de(alc, "lat_max_c240"),
        control_auc=media_de(ctrl, "auc_c240"),
        alcoholic_auc=media_de(alc, "auc_c240"),
        control_media_c320=media_de(ctrl, "media_c320"),
        alcoholic_media_c320=media_de(alc, "media_c320"),
    )


# =============================================================================
# Referencia oficial (Script 06, muestra completa 77 vs 45)
# =============================================================================
@dataclass
class KpiOficial:
    disponible: bool
    control_media: float = float("nan")
    control_sem: float = float("nan")
    alcoholic_media: float = float("nan")
    alcoholic_sem: float = float("nan")
    diferencia: float = float("nan")
    n_control: int = 0
    n_alcoholic: int = 0


def kpi_oficial(
    estadistica: pd.DataFrame,
    metodo: str,
    canal: str,
    condicion: str,
) -> KpiOficial:
    """
    Lee la fila oficial del Script 06 (media_c240) para el método dado.
    Homogéneo -> ventana 'c240 (220-260 ms)'; inhomogéneo -> 'c240 inh.'.
    """
    ventana = config.OFICIAL_VENTANA_C240.get(metodo)
    fila = estadistica[
        (estadistica["ventana"] == ventana)
        & (estadistica["metrica"] == "media_c240")
        & (estadistica["canal"] == canal)
        & (estadistica["condicion"] == condicion)
    ]
    if fila.empty:
        return KpiOficial(disponible=False)
    r = fila.iloc[0]
    nc = int(r["n_control"])
    na = int(r["n_alcoholic"])
    # El CSV guarda SD entre sujetos; lo convertimos a SEM = SD/√n para ser
    # consistentes con la banda de los Grand Average y el resto de las figuras.
    return KpiOficial(
        disponible=True,
        control_media=float(r["control_media"]),
        control_sem=float(r["control_sd"]) / np.sqrt(nc) if nc > 1 else float("nan"),
        alcoholic_media=float(r["alcoholic_media"]),
        alcoholic_sem=float(r["alcoholic_sd"]) / np.sqrt(na) if na > 1 else float("nan"),
        diferencia=float(r["diferencia"]),
        n_control=nc,
        n_alcoholic=na,
    )


# =============================================================================
# SNR (cohorte de referencia 45+45 del Script 04)
# =============================================================================
@dataclass
class KpiSnr:
    disponible: bool
    snr_homogeneo: float = float("nan")     # mediana por canal×condición
    snr_inhomogeneo: float = float("nan")
    n: int = 0


def kpi_snr(snr: pd.DataFrame, canal: str, condicion: str) -> KpiSnr:
    """
    Mediana de SNR (homogéneo / inhomogéneo) para canal×condición, igual que
    comparar_snr() del Script 04. Proviene de tabla_snr_comparacion.csv, que
    corresponde a la cohorte de referencia 45+45 (SEED=42): es una cantidad
    por-sujeto y NO se recalcula con el slider (eso requeriría re-promediar
    trials, justamente lo que evitamos).
    """
    sub = snr[(snr["canal"] == canal) & (snr["condicion"] == condicion)]
    if sub.empty:
        return KpiSnr(disponible=False)
    h = sub["snr_homogeneo"].dropna()
    i = sub["snr_inhomogeneo"].dropna()
    return KpiSnr(
        disponible=True,
        snr_homogeneo=float(h.median()) if len(h) else float("nan"),
        snr_inhomogeneo=float(i.median()) if len(i) else float("nan"),
        n=int(sub["sujeto"].nunique()),
    )
