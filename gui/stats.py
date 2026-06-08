"""
stats.py — Métricas y contrastes para el panel de KPIs.

Réplica EXACTA de las fórmulas de los Scripts 05/06:

  • cohens_d(x, y): pooled SD, d>0 => x tiene media mayor   (Scripts 05 y 06)
  • Welch una cola: stats.ttest_ind(ctrl, alc, equal_var=False,
                                    alternative='greater')   (Script 06)

Política HÍBRIDA (elegida con el usuario):
  - KPI EN VIVO: se recalcula media_c240 ± SD / Cohen's d / p-Welch sobre la
    cohorte actual del slider (n_alc + 45 ctrl) a partir de eeg_c240_extraido.csv,
    para que coincida con el Grand Average mostrado.
    (Verificado: a n=77 reproduce tabla_estadistica.csv a precisión de máquina.)
  - REFERENCIA OFICIAL: se muestran además los valores del Script 06 sobre la
    muestra completa 77 vs 45 (incluido p_fdr), tomados de tabla_estadistica.csv.

La métrica PRINCIPAL es la MEDIA en ventana (media_c240), nunca el máximo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config


# =============================================================================
# Fórmulas base (idénticas a los scripts)
# =============================================================================
def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d con pooled SD. d>0 => x tiene media mayor. (Scripts 05/06)."""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp < config.EPS:
        return float("nan")
    return float((x.mean() - y.mean()) / sp)


def welch_greater(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """T-test de Welch, H1: media(a) > media(b). Devuelve (t, p). (Script 06)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = st.ttest_ind(a, b, equal_var=False, alternative="greater")
    return float(t), float(p)


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
    control_sd: float
    alcoholic_media: float
    alcoholic_sd: float
    diferencia: float          # control - alcohólico
    cohens_d: float
    t_welch: float
    p_welch: float             # H1: control > alcohólico (una cola)
    # Secundarios (medias por grupo) — claramente etiquetados como tales
    control_max: float
    alcoholic_max: float
    control_lat: float
    alcoholic_lat: float
    control_auc: float
    alcoholic_auc: float
    control_media_c320: float
    alcoholic_media_c320: float


def _media_sd(vals: np.ndarray) -> tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    sd = vals.std(ddof=1) if len(vals) > 1 else float("nan")
    return float(vals.mean()), float(sd)


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

    mc, sc = _media_sd(c_media)
    ma, sa = _media_sd(a_media)
    d = cohens_d(c_media, a_media)
    t, p = welch_greater(c_media, a_media)

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
        control_sd=sc,
        alcoholic_media=ma,
        alcoholic_sd=sa,
        diferencia=mc - ma,
        cohens_d=d,
        t_welch=t,
        p_welch=p,
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
    control_sd: float = float("nan")
    alcoholic_media: float = float("nan")
    alcoholic_sd: float = float("nan")
    diferencia: float = float("nan")
    cohens_d: float = float("nan")
    p_welch: float = float("nan")
    p_fdr: float = float("nan")
    sig_fdr: bool = False
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
    return KpiOficial(
        disponible=True,
        control_media=float(r["control_media"]),
        control_sd=float(r["control_sd"]),
        alcoholic_media=float(r["alcoholic_media"]),
        alcoholic_sd=float(r["alcoholic_sd"]),
        diferencia=float(r["diferencia"]),
        cohens_d=float(r["cohens_d"]),
        p_welch=float(r["p_welch"]),
        p_fdr=float(r["p_fdr"]),
        sig_fdr=bool(r["sig_fdr"]),
        n_control=int(r["n_control"]),
        n_alcoholic=int(r["n_alcoholic"]),
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
