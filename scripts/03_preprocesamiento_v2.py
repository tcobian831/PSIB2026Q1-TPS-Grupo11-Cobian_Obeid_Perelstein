"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 03 v2: Preprocesamiento de Señales EEG
==============================================================================

CAMBIOS RESPECTO DE v1
----------------------
1. Ventana de baseline reducida de 100 ms a 30 ms.

   Motivo: el dataset UCI EEG no registra período pre-estímulo; t=0
   coincide con el inicio del estímulo. Los primeros 100 ms post-estímulo
   ya contienen componentes evocados tempranos (típicamente C1/P1 entre
   80–120 ms en occipitales). Usar esos 100 ms como baseline introduce
   una corrección sesgada: estaríamos restando una porción de señal que
   ya está modulada por el estímulo.

   Reducimos a 30 ms (primeras 7 muestras a 256 Hz), que corresponde al
   intervalo previo a la aparición de los primeros componentes evocados
   y por lo tanto refleja actividad de fondo del cerebro.

   Esta limitación del dataset queda documentada en el informe como
   "ausencia de baseline pre-estímulo, sustituido por baseline post-
   estímulo temprano (0–30 ms)".

2. Sin otros cambios funcionales. El resto del pipeline (filtro Butterworth
   0.1–30 Hz, rechazo ±100 µV) se mantiene idéntico.

Entrada:  eeg_data_cargado.parquet  (generado por Script 01)
Salida:   eeg_data_preprocesado_v2.parquet

Uso:
    python 03_preprocesamiento_v2.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS = 256          # Frecuencia de muestreo (Hz)
F_LOW  = 0.1      # Corte inferior del pasa-banda (Hz)
F_HIGH = 30.0     # Corte superior del pasa-banda (Hz)
ORDEN  = 4        # Orden del filtro Butterworth

# Baseline reducido: primeros 30 ms post-estímulo (≈ 7 muestras a 256 Hz)
# antes de la aparición de los componentes evocados tempranos.
T_BASELINE_MS = 30
N_BASELINE    = int(T_BASELINE_MS / 1000 * FS)   # = 7 muestras

# Umbral de rechazo de artefactos (µV)
UMBRAL_UV = 100.0

CANALES_INTERES     = ["P8", "PO8", "T8", "TP8"]
CONDICIONES_INTERES = ["S1 obj", "S2 nomatch"]

ENTRADA  = Path("eeg_data_cargado.parquet")
SALIDA   = Path("eeg_data_preprocesado_v2.parquet")

# =============================================================================
# FUNCIONES (sin cambios respecto de v1)
# =============================================================================

def disenar_filtro_butterworth(f_low, f_high, fs, orden):
    """Diseña un filtro pasa-banda Butterworth digital."""
    nyquist = fs / 2.0
    b, a = butter(orden, [f_low / nyquist, f_high / nyquist], btype="band")
    return b, a


def aplicar_filtro(senal, b, a):
    """Filtfilt: filtrado bidireccional, fase cero (preserva latencias)."""
    return filtfilt(b, a, senal)


def corregir_baseline(senal, n_baseline):
    """Resta el promedio de los primeros n_baseline puntos."""
    return senal - np.mean(senal[:n_baseline])


def tiene_artefacto(senal, umbral):
    """True si la señal supera ±umbral µV en algún punto."""
    return bool(np.any(np.abs(senal) > umbral))


def preprocesar_trial(grupo_trial, b, a):
    """Pipeline: filtro → baseline → rechazo de artefactos."""
    df = grupo_trial.sort_values("muestra").copy()

    if len(df) != 256:
        return None

    senal = df["valor_uV"].values.astype(float)
    senal = aplicar_filtro(senal, b, a)
    senal = corregir_baseline(senal, N_BASELINE)

    if tiene_artefacto(senal, UMBRAL_UV):
        return None

    df["valor_uV"] = senal
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 03 v2: Preprocesamiento (baseline 30 ms)")
    print("=" * 60)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 01."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas ({df['sujeto'].nunique()} sujetos)")

    # Filtrado por condicion y canal
    print(f"\nFiltrado de condiciones: {CONDICIONES_INTERES}")
    print(f"Filtrado de canales:     {CANALES_INTERES}")
    df = df[
        df["condicion"].isin(CONDICIONES_INTERES) &
        df["canal"].isin(CANALES_INTERES)
    ].copy()
    print(f"  Filas tras filtrado: {len(df):,}")

    # Diseno del filtro
    print(f"\nDisenando filtro Butterworth pasa-banda "
          f"({F_LOW}–{F_HIGH} Hz, orden {ORDEN})...")
    b, a = disenar_filtro_butterworth(F_LOW, F_HIGH, FS, ORDEN)
    print(f"Baseline: primeros {T_BASELINE_MS} ms post-estimulo "
          f"({N_BASELINE} muestras)")

    # Preprocesamiento trial por trial
    print("\nPreprocesando trials...")
    grupos     = df.groupby(["sujeto", "trial_num", "canal"])
    n_total    = len(grupos)
    n_ok       = 0
    n_rechazados = 0
    resultados = []

    for idx, (nombre, grupo) in enumerate(grupos):
        if idx % 5000 == 0:
            print(f"  Progreso: {idx:,}/{n_total:,} trials procesados...")

        df_proc = preprocesar_trial(grupo, b, a)
        if df_proc is not None:
            resultados.append(df_proc)
            n_ok += 1
        else:
            n_rechazados += 1

    print(f"\nResultados:")
    print(f"  Trials procesados:  {n_ok:,}")
    print(f"  Trials rechazados:  {n_rechazados:,}  "
          f"({100*n_rechazados/(n_ok+n_rechazados):.1f}%)")

    df_proc = pd.concat(resultados, ignore_index=True)

    # Distribucion por grupo y condicion
    print("\nDistribucion de trials limpios:")
    resumen = (
        df_proc.groupby(["grupo", "condicion", "sujeto", "trial_num"])
        .size().reset_index()
        .groupby(["grupo", "condicion"]).size().reset_index(name="n_trials")
    )
    print(resumen.to_string(index=False))

    df_proc.to_parquet(SALIDA, index=False)
    print(f"\nDatos preprocesados guardados en '{SALIDA}'")
    print("\n[OK] Script 03 v2 finalizado.")
