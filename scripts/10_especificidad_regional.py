"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 10: Especificidad Regional con Canales Control
==============================================================================

Propósito
---------
Zhang et al. (1997) reportaron que las diferencias entre grupos en el
componente c240/VMP son selectivas de regiones temporo-occipitales
derechas. Si nuestro hallazgo es genuino (y no un efecto global de EEG),
debería ser MÁS marcado en P8/PO8/T8/TP8 que en canales control de la
línea media (Cz, Fz).

Este script:
1. Reprocesa desde cero los canales control (Cz, Fz) con el mismo
   pipeline v2 (filtro Butterworth + baseline 30 ms + rechazo +/- 100uV).
2. Calcula PE individuales y extrae |c240| con magnitud absoluta.
3. Aplica t-test entre grupos en los canales control.
4. Compara visualmente la magnitud del efecto (Cohen's d) entre canales
   de interés y canales control, por condición.

Resultado esperado: |d| significativamente mayor en P8/PO8/T8/TP8 que
en Cz/Fz, lo que respalda que el efecto es regionalmente específico.

Entrada:  outputs/eeg_data_cargado.parquet  (Script 01)
          outputs/tabla_estadistica_v2.csv  (para los d de canales de interés)
Salida:   outputs/eeg_c240_control_chans_v2.csv
          outputs/tabla_especificidad_regional_v2.csv
          outputs/figura_especificidad_regional_v2.png

Uso:
    python 10_especificidad_regional.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import ttest_ind

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS              = 256
F_LOW, F_HIGH   = 0.1, 30.0
ORDEN           = 4
T_BASELINE_MS   = 30
N_BASELINE      = int(T_BASELINE_MS / 1000 * FS)
UMBRAL_UV       = 100.0

# Canales control (línea media) — sirven como referencia anatómica
CANALES_CONTROL = ["Cz", "Fz"]

# Canales de interés (temporo-occipitales derechos)
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]

CONDICIONES     = ["S1 obj", "S2 nomatch"]

T_INI_MS, T_FIN_MS = 220, 260
M_INI = int(T_INI_MS / 1000 * FS)
M_FIN = int(T_FIN_MS / 1000 * FS)

ENTRADA_CRUDO   = Path("eeg_data_cargado.parquet")
ENTRADA_INTERES = Path("../outputs/tabla_estadistica_v2.csv")
SALIDA_C240     = Path("../outputs/eeg_c240_control_chans_v2.csv")
SALIDA_TABLA    = Path("../outputs/tabla_especificidad_regional_v2.csv")

# =============================================================================
# FUNCIONES
# =============================================================================

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if s_pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / s_pooled


def preprocesar_canales_control(df_crudo):
    """Filtro + baseline + rechazo en los canales Cz/Fz."""
    print("Preprocesando canales control...")
    df = df_crudo[
        df_crudo["condicion"].isin(CONDICIONES) &
        df_crudo["canal"].isin(CANALES_CONTROL)
    ].copy()
    if df.empty:
        raise ValueError(
            f"No se encontraron datos para los canales control {CANALES_CONTROL}. "
            "Verificar que esos canales existan en el dataset."
        )
    print(f"  {len(df):,} filas tras filtrado")

    nyquist = FS / 2.0
    b, a = butter(ORDEN, [F_LOW / nyquist, F_HIGH / nyquist], btype="band")

    grupos = df.groupby(["sujeto", "trial_num", "canal"])
    n_total = len(grupos)
    n_ok    = 0
    out     = []
    print(f"  Procesando {n_total:,} trials...")
    for idx, (nombre, grupo) in enumerate(grupos):
        if idx % 5000 == 0 and idx > 0:
            print(f"    {idx:,}/{n_total:,}")
        df_t = grupo.sort_values("muestra").copy()
        if len(df_t) != 256:
            continue
        senal = df_t["valor_uV"].values.astype(float)
        senal = filtfilt(b, a, senal)
        senal = senal - np.mean(senal[:N_BASELINE])
        if np.any(np.abs(senal) > UMBRAL_UV):
            continue
        df_t["valor_uV"] = senal
        out.append(df_t)
        n_ok += 1

    print(f"  Trials limpios: {n_ok:,} / {n_total:,}")
    return pd.concat(out, ignore_index=True)


def calcular_c240_canales_control(df_preproc):
    """Promedia trials por sujeto x canal x condicion y extrae |c240|."""
    print("\nPromediando y extrayendo c240 de canales control...")
    PE = (
        df_preproc.groupby(["sujeto", "grupo", "canal", "condicion", "muestra"])
        ["valor_uV"]
        .agg(PE_uV="mean", n_trials="count")
        .reset_index()
    )

    filas = []
    for (sujeto, grupo, canal, condicion), df_sub in PE.groupby(
            ["sujeto", "grupo", "canal", "condicion"]):
        ventana = df_sub[
            (df_sub["muestra"] >= M_INI) &
            (df_sub["muestra"] <= M_FIN)
        ].sort_values("muestra")
        if ventana.empty:
            continue
        idx_pico = ventana["PE_uV"].abs().idxmax()
        amplitud = ventana.loc[idx_pico, "PE_uV"]
        filas.append({
            "sujeto":          sujeto,
            "grupo":           grupo,
            "canal":           canal,
            "condicion":       condicion,
            "amplitud_uV":     amplitud,
            "amplitud_abs_uV": abs(amplitud),
            "latencia_ms":     ventana.loc[idx_pico, "muestra"] / FS * 1000,
        })
    return pd.DataFrame(filas)


def contraste_canales_control(df_c240_ctrl):
    """t-test entre grupos en canales control."""
    filas = []
    for cond in CONDICIONES:
        for canal in CANALES_CONTROL:
            sel = (df_c240_ctrl["canal"] == canal) & (df_c240_ctrl["condicion"] == cond)
            ctrl = df_c240_ctrl[sel & (df_c240_ctrl["grupo"] == "control")
                                ]["amplitud_abs_uV"].dropna().values
            alc  = df_c240_ctrl[sel & (df_c240_ctrl["grupo"] == "alcoholic")
                                ]["amplitud_abs_uV"].dropna().values
            if len(ctrl) < 2 or len(alc) < 2:
                continue
            t_stat, p_valor = ttest_ind(ctrl, alc, equal_var=False)
            d = cohen_d(ctrl, alc)
            filas.append({
                "canal":           canal,
                "condicion":       cond,
                "tipo":            "control",
                "n_control":       len(ctrl),
                "n_alcoholic":     len(alc),
                "media_ctrl":      float(np.mean(ctrl)),
                "media_alc":       float(np.mean(alc)),
                "t_estadistico":   float(t_stat),
                "p_valor":         float(p_valor),
                "cohen_d":         float(d),
            })
    return pd.DataFrame(filas)


def graficar_especificidad(df_total):
    """
    Compara |Cohen's d| entre canales de interés (4) y canales control (2),
    por condición. Resultado esperado: barras de interés visiblemente
    más altas que las de control.
    """
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5),
                             sharey=True)
    fig.suptitle(
        "Especificidad regional: |Cohen's d| en canales de interes vs controles\n"
        "Hipotesis: efecto mayor en temporo-occipitales derechos (P8/PO8/T8/TP8)",
        fontsize=12
    )

    for ax, cond in zip(axes, CONDICIONES):
        sub = df_total[df_total["condicion"] == cond].copy()
        sub["abs_d"] = sub["cohen_d"].abs()

        # Ordenar: interes primero, luego control
        sub["orden"] = sub["tipo"].map({"interes": 0, "control": 1})
        sub = sub.sort_values(["orden", "canal"]).reset_index(drop=True)

        colores = ["#dc2626" if t == "interes" else "#94a3b8"
                   for t in sub["tipo"]]
        x = np.arange(len(sub))
        ax.bar(x, sub["abs_d"], color=colores, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["canal"], fontsize=10)
        ax.set_ylabel("|Cohen's d|")
        ax.set_title(f"Condicion: {cond}", fontsize=11)
        ax.axhline(0.2, color="gray", linestyle=":", linewidth=0.8, label="d=0.2 (chico)")
        ax.axhline(0.5, color="orange", linestyle=":", linewidth=0.8, label="d=0.5 (medio)")
        ax.axhline(0.8, color="red", linestyle=":", linewidth=0.8, label="d=0.8 (grande)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        for i, d in enumerate(sub["abs_d"]):
            ax.text(i, d + 0.02, f"{d:.2f}", ha="center", fontsize=9)

        # Leyenda de colores
        ax.text(0.02, 0.98,
                "Rojo = canales de interes\nGris = canales control",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))

    plt.tight_layout()
    plt.savefig("../outputs/figura_especificidad_regional_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_especificidad_regional_v2.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Script 10: Especificidad Regional (Cz/Fz vs P8/PO8/T8/TP8)")
    print("=" * 60)

    if not ENTRADA_CRUDO.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA_CRUDO}'.\n"
            "Asegurate de haber corrido el Script 01."
        )
    if not ENTRADA_INTERES.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA_INTERES}'.\n"
            "Asegurate de haber corrido el Script 06 v2."
        )

    print(f"\nCargando datos crudos...")
    df_crudo = pd.read_parquet(ENTRADA_CRUDO)
    canales_disponibles = set(df_crudo["canal"].unique())
    no_disp = [c for c in CANALES_CONTROL if c not in canales_disponibles]
    if no_disp:
        print(f"  [!] Canales control no disponibles: {no_disp}")
        print(f"      Canales disponibles que contienen 'z': "
              f"{sorted([c for c in canales_disponibles if 'z' in c.lower()])}")
        # Filtrar a los que si esten
        CANALES_CONTROL_DISP = [c for c in CANALES_CONTROL if c in canales_disponibles]
    else:
        CANALES_CONTROL_DISP = CANALES_CONTROL

    if not CANALES_CONTROL_DISP:
        raise ValueError(
            "Ninguno de los canales control esperados esta disponible. "
            "Revisar 'canales_disponibles' arriba y editar CANALES_CONTROL."
        )

    # Preprocesar y extraer
    df_preproc = preprocesar_canales_control(df_crudo)
    df_c240_ctrl = calcular_c240_canales_control(df_preproc)
    df_c240_ctrl.to_csv(SALIDA_C240, index=False)
    print(f"\nDatos guardados: {SALIDA_C240}")

    # Contraste
    df_ctrl = contraste_canales_control(df_c240_ctrl)
    df_ctrl["tipo"] = "control"
    print("\nT-test en canales control (Cz/Fz):")
    print(df_ctrl[["canal", "condicion", "media_ctrl", "media_alc",
                   "t_estadistico", "p_valor", "cohen_d"]].to_string(index=False))

    # Cargar Cohen's d de canales de interes
    df_int = pd.read_csv(ENTRADA_INTERES)[
        ["canal", "condicion", "media_abs_control", "media_abs_alcoholic",
         "t_estadistico", "p_valor", "cohen_d"]
    ].copy()
    df_int = df_int.rename(columns={
        "media_abs_control": "media_ctrl",
        "media_abs_alcoholic": "media_alc",
    })
    df_int["tipo"] = "interes"

    # Tabla unificada
    df_total = pd.concat([df_int, df_ctrl], ignore_index=True)
    df_total.to_csv(SALIDA_TABLA, index=False)
    print(f"\nTabla unificada: {SALIDA_TABLA}")

    print("\nGenerando grafico de especificidad regional...")
    graficar_especificidad(df_total)

    print("\n[OK] Script 10 finalizado.")
