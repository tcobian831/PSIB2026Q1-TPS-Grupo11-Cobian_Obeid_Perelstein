"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 03 v2b: Preprocesamiento incluyendo S2 match
==============================================================================

PROPÓSITO
---------
Variante del Script 03 v2 que procesa las TRES condiciones (S1 obj,
S2 nomatch, S2 match) para habilitar el análisis secundario del efecto
nonmatch-match previsto en el anteproyecto.

Es exactamente el mismo pipeline (filtro Butterworth 0.1-30 Hz, baseline
30 ms, rechazo +/- 100 uV); cambia únicamente la lista de condiciones
filtradas y el nombre del parquet de salida.

Entrada:  eeg_data_cargado.parquet
Salida:   eeg_data_preprocesado_v2b.parquet  (incluye S2 match)

Uso:
    python 03_preprocesamiento_v2b.py
==============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

FS     = 256
F_LOW  = 0.1
F_HIGH = 30.0
ORDEN  = 4

T_BASELINE_MS = 30
N_BASELINE    = int(T_BASELINE_MS / 1000 * FS)

UMBRAL_UV = 100.0

CANALES_INTERES     = ["P8", "PO8", "T8", "TP8"]
# Diferencia con v2: agregamos S2 match
CONDICIONES_INTERES = ["S1 obj", "S2 nomatch", "S2 match"]

ENTRADA = Path("../outputs/eeg_data_cargado.parquet")
SALIDA  = Path("../outputs/eeg_data_preprocesado_v2b.parquet")


def disenar_filtro_butterworth(f_low, f_high, fs, orden):
    nyquist = fs / 2.0
    b, a = butter(orden, [f_low / nyquist, f_high / nyquist], btype="band")
    return b, a


def aplicar_filtro(senal, b, a):
    return filtfilt(b, a, senal)


def corregir_baseline(senal, n_baseline):
    return senal - np.mean(senal[:n_baseline])


def tiene_artefacto(senal, umbral):
    return bool(np.any(np.abs(senal) > umbral))


def preprocesar_trial(grupo_trial, b, a):
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


if __name__ == "__main__":
    print("=" * 60)
    print("Script 03 v2b: Preprocesamiento (incluye S2 match)")
    print("=" * 60)

    if not ENTRADA.exists():
        raise FileNotFoundError(f"No se encontro '{ENTRADA}'.")

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas")

    print(f"\nFiltrando condiciones: {CONDICIONES_INTERES}")
    print(f"Filtrando canales:     {CANALES_INTERES}")
    df = df[
        df["condicion"].isin(CONDICIONES_INTERES) &
        df["canal"].isin(CANALES_INTERES)
    ].copy()
    print(f"  Filas tras filtrado: {len(df):,}")

    b, a = disenar_filtro_butterworth(F_LOW, F_HIGH, FS, ORDEN)
    print(f"\nFiltro Butterworth {F_LOW}-{F_HIGH} Hz orden {ORDEN} disenado")
    print(f"Baseline: primeros {T_BASELINE_MS} ms ({N_BASELINE} muestras)")

    grupos    = df.groupby(["sujeto", "trial_num", "canal"])
    n_total   = len(grupos)
    n_ok      = 0
    n_rech    = 0
    resultados = []

    print(f"\nPreprocesando {n_total:,} trials...")
    for idx, (nombre, grupo) in enumerate(grupos):
        if idx % 5000 == 0:
            print(f"  {idx:,}/{n_total:,}")
        df_proc = preprocesar_trial(grupo, b, a)
        if df_proc is not None:
            resultados.append(df_proc)
            n_ok += 1
        else:
            n_rech += 1

    print(f"\n  Trials procesados:  {n_ok:,}")
    print(f"  Trials rechazados:  {n_rech:,}  ({100*n_rech/(n_ok+n_rech):.1f}%)")

    df_proc = pd.concat(resultados, ignore_index=True)

    print("\nDistribucion por condicion x grupo:")
    res = (
        df_proc.groupby(["grupo", "condicion", "sujeto", "trial_num"])
        .size().reset_index()
        .groupby(["grupo", "condicion"]).size().reset_index(name="n_trials")
    )
    print(res.to_string(index=False))

    df_proc.to_parquet(SALIDA, index=False)
    print(f"\nGuardado en '{SALIDA}'")
    print("\n[OK] Script 03 v2b finalizado.")
