"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 03: Preprocesamiento de Señales EEG
==============================================================================

Pasos:
    1. Filtro pasa-banda Butterworth (0.1 – 30 Hz)
    2. Corrección de baseline (primeros 100 ms)
    3. Selección de condiciones y canales de interés
    4. Rechazo de trials con artefactos (umbral ±100 µV)

Entrada:  eeg_data_cargado.parquet  (generado por Script 01)
Salida:   eeg_data_preprocesado.parquet

Uso:
    Correr desde la carpeta scripts/
    python 03_preprocesamiento.py
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
F_LOW  = 0.1      # Frecuencia de corte inferior del pasa-banda (Hz)
F_HIGH = 30.0     # Frecuencia de corte superior del pasa-banda (Hz)
ORDEN  = 4        # Orden del filtro Butterworth

# Ventana de baseline: primeros 100 ms = primeras 26 muestras a 256 Hz
N_BASELINE = int(0.100 * FS)   # = 25 muestras (índices 0–24)

# Umbral de rechazo de artefactos (µV)
UMBRAL_UV = 100.0

# Canales de interés (temporo-occipitales derechos)
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]

# Condiciones de interés
CONDICIONES_INTERES = ["S1 obj", "S2 nomatch"]

# Rutas
ENTRADA  = Path("eeg_data_cargado.parquet")
SALIDA   = Path("eeg_data_preprocesado.parquet")

# =============================================================================
# FUNCIONES
# =============================================================================

def disenar_filtro_butterworth(f_low: float, f_high: float,
                                fs: int, orden: int):
    """
    Diseña un filtro pasa-banda Butterworth digital.

    Butterworth es el filtro estándar en EEG/ERP por dos razones:
    - Respuesta en frecuencia maximalmente plana en la banda de paso
      (no ondula, no distorsiona la amplitud de las frecuencias que nos
      interesan)
    - Fase no lineal pero usamos filtfilt (filtrado bidireccional) que
      cancela el desfase de fase → fase cero efectiva

    Args:
        f_low:  frecuencia de corte inferior (Hz)
        f_high: frecuencia de corte superior (Hz)
        fs:     frecuencia de muestreo (Hz)
        orden:  orden del filtro (4 es estándar en EEG)

    Retorna:
        b, a: coeficientes del filtro
    """
    nyquist = fs / 2.0
    low  = f_low  / nyquist
    high = f_high / nyquist
    b, a = butter(orden, [low, high], btype="band")
    return b, a


def aplicar_filtro(señal: np.ndarray, b: np.ndarray,
                   a: np.ndarray) -> np.ndarray:
    """
    Aplica el filtro pasa-banda a una señal 1D usando filtfilt.

    filtfilt aplica el filtro dos veces (ida y vuelta) para lograr
    fase cero — crítico en ERPs porque preserva la latencia exacta
    de los componentes (ej: el pico c240 queda en 240 ms, no desplazado).

    Args:
        señal: array 1D con las muestras del trial (256 puntos)
        b, a:  coeficientes del filtro Butterworth

    Retorna:
        señal filtrada (mismo largo que la entrada)
    """
    # filtfilt necesita que la señal sea más larga que 3 * max(len(a), len(b))
    # Con 256 muestras y orden 4 no hay problema
    return filtfilt(b, a, señal)


def corregir_baseline(señal: np.ndarray, n_baseline: int) -> np.ndarray:
    """
    Corrección de baseline: resta el promedio de los primeros n_baseline
    puntos a toda la señal.

    En ERPs, el baseline son los milisegundos previos al estímulo.
    Como en este dataset el estímulo ocurre en t=0 (muestra 0), usamos
    los primeros ~100 ms como aproximación al pre-estímulo.

    Args:
        señal:      array 1D de 256 muestras
        n_baseline: cantidad de muestras a usar como baseline (ej: 25)

    Retorna:
        señal corregida (media de baseline = 0)
    """
    baseline_mean = np.mean(señal[:n_baseline])
    return señal - baseline_mean


def tiene_artefacto(señal: np.ndarray, umbral: float) -> bool:
    """
    Detecta si un trial tiene artefactos por amplitud.

    Criterio estándar en EEG: si la señal supera ±umbral µV en algún
    punto, se considera contaminada (parpadeo, movimiento ocular,
    artefacto muscular, etc.).

    Args:
        señal:   array 1D con las muestras del trial
        umbral:  valor absoluto máximo permitido (µV)

    Retorna:
        True si hay artefacto, False si el trial es limpio
    """
    return bool(np.any(np.abs(señal) > umbral))


def preprocesar_trial(grupo_trial: pd.DataFrame,
                      b: np.ndarray, a: np.ndarray) -> pd.DataFrame | None:
    """
    Aplica el pipeline completo de preprocesamiento a un trial de un canal.

    Pipeline:
        señal cruda → filtro pasa-banda → corrección baseline → 
        → verificar artefactos → señal lista para ERP

    Args:
        grupo_trial: DataFrame con 256 filas (un trial, un canal)
                     columnas: muestra, valor_uV, + metadatos
        b, a:        coeficientes del filtro

    Retorna:
        DataFrame con valor_uV reemplazado por señal preprocesada,
        o None si el trial fue rechazado por artefacto.
    """
    df = grupo_trial.sort_values("muestra").copy()

    if len(df) != 256:
        return None  # Trial incompleto

    señal = df["valor_uV"].values.astype(float)

    # Paso 1: filtro pasa-banda Butterworth
    señal_filtrada = aplicar_filtro(señal, b, a)

    # Paso 2: corrección de baseline
    señal_corregida = corregir_baseline(señal_filtrada, N_BASELINE)

    # Paso 3: rechazo de artefactos
    if tiene_artefacto(señal_corregida, UMBRAL_UV):
        return None

    df["valor_uV"] = señal_corregida
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 02: Preprocesamiento")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Cargar datos del Script 01
    # -------------------------------------------------------------------------
    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 01."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas ({df['sujeto'].nunique()} sujetos)")

    # -------------------------------------------------------------------------
    # Paso 1: Selección de condiciones y canales de interés
    # -------------------------------------------------------------------------
    print(f"\nFiltrado de condiciones: {CONDICIONES_INTERES}")
    print(f"Filtrado de canales:     {CANALES_INTERES}")

    df = df[
        df["condicion"].isin(CONDICIONES_INTERES) &
        df["canal"].isin(CANALES_INTERES)
    ].copy()

    print(f"  Filas tras filtrado: {len(df):,}")
    print(f"  Trials únicos: {df.groupby(['sujeto','trial_num','canal']).ngroups:,}")

    # -------------------------------------------------------------------------
    # Paso 2: Diseñar filtro Butterworth
    # -------------------------------------------------------------------------
    print(f"\nDiseñando filtro Butterworth pasa-banda "
          f"({F_LOW}–{F_HIGH} Hz, orden {ORDEN})...")
    b, a = disenar_filtro_butterworth(F_LOW, F_HIGH, FS, ORDEN)
    print("  Filtro diseñado correctamente.")

    # -------------------------------------------------------------------------
    # Paso 3: Preprocesar trial por trial
    # -------------------------------------------------------------------------
    print("\nPreprocesando trials (filtro + baseline + rechazo artefactos)...")

    grupos = df.groupby(["sujeto", "trial_num", "canal"])
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

    print(f"\nResultados del preprocesamiento:")
    print(f"  Trials procesados:  {n_ok:,}")
    print(f"  Trials rechazados:  {n_rechazados:,}  "
          f"({100*n_rechazados/(n_ok+n_rechazados):.1f}%)")

    df_proc = pd.concat(resultados, ignore_index=True)

    # -------------------------------------------------------------------------
    # Paso 4: Verificación por grupo y condición
    # -------------------------------------------------------------------------
    print("\nDistribución de trials limpios por grupo y condición:")
    resumen = (
        df_proc.groupby(["grupo", "condicion", "sujeto", "trial_num"])
        .size()
        .reset_index()
        .groupby(["grupo", "condicion"])
        .size()
        .reset_index(name="n_trials")
    )
    print(resumen.to_string(index=False))

    # -------------------------------------------------------------------------
    # Paso 5: Gráfico comparativo — señal cruda vs preprocesada
    # -------------------------------------------------------------------------
    print("\nGenerando gráfico comparativo cruda vs preprocesada...")

    # Tomar un trial de ejemplo: primer sujeto control, canal P8, S1 obj
    sujeto_ej = df_proc[df_proc["grupo"] == "control"]["sujeto"].iloc[0]
    trial_ej  = df_proc[
        (df_proc["sujeto"] == sujeto_ej) &
        (df_proc["canal"] == "P8") &
        (df_proc["condicion"] == "S1 obj")
    ]["trial_num"].iloc[0]

    # Señal preprocesada
    señal_proc = df_proc[
        (df_proc["sujeto"] == sujeto_ej) &
        (df_proc["trial_num"] == trial_ej) &
        (df_proc["canal"] == "P8")
    ].sort_values("muestra")["valor_uV"].values

    # Señal cruda (del DataFrame original antes de procesar)
    df_cruda_ej = df[
        (df["sujeto"] == sujeto_ej) &
        (df["trial_num"] == trial_ej) &
        (df["canal"] == "P8")
    ].sort_values("muestra")
    señal_cruda = df_cruda_ej["valor_uV"].values

    tiempo_ms = np.arange(256) / FS * 1000

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f"Efecto del preprocesamiento — Canal P8 — "
        f"Sujeto: {sujeto_ej} (control) — S1 obj",
        fontsize=12
    )

    axes[0].plot(tiempo_ms, señal_cruda, color="#64748b", linewidth=1)
    axes[0].set_ylabel("Amplitud (µV)")
    axes[0].set_title("Señal cruda")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tiempo_ms, señal_proc, color="#2563eb", linewidth=1)
    axes[1].set_ylabel("Amplitud (µV)")
    axes[1].set_xlabel("Tiempo (ms)")
    axes[1].set_title(
        f"Señal preprocesada "
        f"(Butterworth {F_LOW}–{F_HIGH} Hz + baseline)"
    )
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axvspan(220, 260, alpha=0.15, color="orange", label="Ventana c240")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figura_preprocesamiento.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada como 'figura_preprocesamiento.png'")

    # -------------------------------------------------------------------------
    # Guardar resultado
    # -------------------------------------------------------------------------
    df_proc.to_parquet(SALIDA, index=False)
    print(f"\nDatos preprocesados guardados en '{SALIDA}'")
    print("\n[OK] Script 03 finalizado.")
