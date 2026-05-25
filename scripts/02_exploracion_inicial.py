"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 02: Visualización y Análisis Exploratorio
==============================================================================

Propósito:
    Analizar las señales EEG CRUDAS en el dominio del tiempo y la frecuencia
    ANTES de aplicar cualquier filtro. Este análisis justifica las decisiones
    de preprocesamiento tomadas en el Script 03.

Análisis realizados:
    1. Visualización en el dominio del tiempo (señales crudas)
    2. Espectro de potencia (PSD) via Welch — dominio de la frecuencia
    3. Análisis de artefactos (distribución de amplitudes)
    4. Conclusiones que justifican el filtro pasa-banda elegido

Entrada:  eeg_data_cargado.parquet  (generado por Script 01)
Salida:   figura_exploracion_tiempo.png
          figura_exploracion_psd.png
          figura_exploracion_artefactos.png

Uso:
    Correr desde la carpeta scripts/
    python 02_exploracion.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import welch

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS        = 256    # Frecuencia de muestreo (Hz)
N_SAMPLES = 256    # Muestras por trial

# Canales de interés
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]

# Cuántos trials de ejemplo mostrar por grupo en el dominio del tiempo
N_TRIALS_EJEMPLO = 5

ENTRADA = Path("eeg_data_cargado.parquet")

# =============================================================================
# FUNCIONES — DOMINIO DEL TIEMPO
# =============================================================================

def graficar_señales_tiempo(df: pd.DataFrame):
    """
    Grafica señales EEG crudas de ejemplo en el dominio del tiempo.

    Muestra N_TRIALS_EJEMPLO trials individuales de un sujeto control
    y uno alcohólico, para el canal P8 en condición S1 obj.
    Permite observar visualmente:
        - Nivel de ruido de la señal cruda
        - Presencia o ausencia de deriva de línea de base
        - Artefactos puntuales (picos abruptos)
        - Variabilidad trial a trial
    """
    fig, axes = plt.subplots(2, N_TRIALS_EJEMPLO,
                             figsize=(4 * N_TRIALS_EJEMPLO, 6),
                             sharex=True, sharey=True)

    fig.suptitle(
        "Señales EEG crudas — Canal P8 — Condición S1 obj\n"
        f"({N_TRIALS_EJEMPLO} trials individuales por grupo)",
        fontsize=13
    )

    tiempo_ms = np.arange(N_SAMPLES) / FS * 1000
    colores   = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for fila, grupo in enumerate(["control", "alcoholic"]):
        subset = df[
            (df["grupo"] == grupo) &
            (df["canal"] == "P8") &
            (df["condicion"] == "S1 obj")
        ]

        if subset.empty:
            continue

        # Tomar el primer sujeto disponible y sus primeros N trials
        primer_sujeto = subset["sujeto"].iloc[0]
        trials_disp   = subset[subset["sujeto"] == primer_sujeto]["trial_num"].unique()
        trials_ej     = trials_disp[:N_TRIALS_EJEMPLO]

        for col, trial in enumerate(trials_ej):
            ax = axes[fila][col]
            señal = (
                subset[
                    (subset["sujeto"] == primer_sujeto) &
                    (subset["trial_num"] == trial)
                ]
                .sort_values("muestra")["valor_uV"]
                .values
            )

            ax.plot(tiempo_ms, señal,
                    color=colores[grupo], linewidth=0.9, alpha=0.85)
            ax.axhline(0, color="black", linewidth=0.4)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.7)
            ax.grid(True, alpha=0.25)

            if col == 0:
                ax.set_ylabel(f"{grupo.capitalize()}\nAmplitud (µV)", fontsize=9)
            if fila == 0:
                ax.set_title(f"Trial {trial}", fontsize=9)
            if fila == 1:
                ax.set_xlabel("Tiempo (ms)", fontsize=9)

    plt.tight_layout()
    plt.savefig("figura_exploracion_tiempo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_exploracion_tiempo.png'")


def graficar_señales_multicanal(df: pd.DataFrame):
    """
    Grafica un trial de ejemplo en los 4 canales de interés,
    para control y alcohólico. Permite comparar la morfología
    de la señal cruda entre canales.
    """
    fig, axes = plt.subplots(2, len(CANALES_INTERES),
                             figsize=(4 * len(CANALES_INTERES), 6),
                             sharex=True)

    fig.suptitle(
        "Señal EEG cruda — Canales de interés — Un trial de ejemplo",
        fontsize=13
    )

    tiempo_ms = np.arange(N_SAMPLES) / FS * 1000
    colores   = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for fila, grupo in enumerate(["control", "alcoholic"]):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            subset = df[
                (df["grupo"] == grupo) &
                (df["canal"] == canal) &
                (df["condicion"] == "S1 obj")
            ]

            if subset.empty:
                ax.set_title(f"{canal} — sin datos")
                continue

            primer_sujeto = subset["sujeto"].iloc[0]
            primer_trial  = subset[
                subset["sujeto"] == primer_sujeto
            ]["trial_num"].iloc[0]

            señal = (
                subset[
                    (subset["sujeto"] == primer_sujeto) &
                    (subset["trial_num"] == primer_trial)
                ]
                .sort_values("muestra")["valor_uV"]
                .values
            )

            ax.plot(tiempo_ms, señal,
                    color=colores[grupo], linewidth=0.9)
            ax.axhline(0, color="black", linewidth=0.4)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.7)
            ax.grid(True, alpha=0.25)

            if col == 0:
                ax.set_ylabel(f"{grupo.capitalize()}\nAmplitud (µV)", fontsize=9)
            if fila == 0:
                ax.set_title(f"Canal: {canal}", fontsize=10)
            if fila == 1:
                ax.set_xlabel("Tiempo (ms)", fontsize=9)

    plt.tight_layout()
    plt.savefig("figura_exploracion_multicanal.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_exploracion_multicanal.png'")


# =============================================================================
# FUNCIONES — DOMINIO DE LA FRECUENCIA (PSD)
# =============================================================================

def calcular_psd_promedio(df: pd.DataFrame, canal: str,
                           grupo: str, n_sujetos: int = 10) -> tuple:
    """
    Calcula el Power Spectral Density (PSD) promedio para un grupo y canal.

    Usa el método de Welch, que divide la señal en ventanas solapadas,
    calcula el espectro de cada una y promedia. Es más robusto que la FFT
    directa para señales ruidosas como el EEG.

    Se calcula sobre trials individuales (señal cruda de 256 muestras) y
    se promedia el PSD de todos los trials disponibles. Esto da una
    estimación estable del espectro típico del grupo.

    Args:
        df:        DataFrame con señales crudas
        canal:     nombre del canal (ej: "P8")
        grupo:     "control" o "alcoholic"
        n_sujetos: cuántos sujetos incluir en el promedio

    Retorna:
        freqs: array de frecuencias (Hz)
        psd_mean: PSD promedio (µV²/Hz)
        psd_sem:  error estándar del PSD entre trials
    """
    subset = df[
        (df["grupo"] == grupo) &
        (df["canal"] == canal) &
        (df["condicion"] == "S1 obj")
    ]

    sujetos = subset["sujeto"].unique()[:n_sujetos]
    psds = []

    for sujeto in sujetos:
        trials = subset[subset["sujeto"] == sujeto]["trial_num"].unique()
        for trial in trials:
            señal = (
                subset[
                    (subset["sujeto"] == sujeto) &
                    (subset["trial_num"] == trial)
                ]
                .sort_values("muestra")["valor_uV"]
                .values
            )
            if len(señal) == N_SAMPLES:
                # nperseg=128: ventanas de 128 muestras (0.5 s)
                # con solapamiento del 50% por defecto
                freqs, psd = welch(señal, fs=FS, nperseg=128)
                psds.append(psd)

    psds     = np.array(psds)
    psd_mean = np.mean(psds, axis=0)
    psd_sem  = np.std(psds, axis=0) / np.sqrt(len(psds))

    return freqs, psd_mean, psd_sem


def graficar_psd(df: pd.DataFrame):
    """
    Grafica el Power Spectral Density (PSD) de las señales crudas.

    Para cada canal de interés, muestra el espectro promedio de control
    y alcohólico en escala logarítmica (dB). Este gráfico es el que
    justifica la elección del filtro pasa-banda:

        - La mayor parte de la energía útil del EEG está por debajo de 30 Hz
        - Por encima de 30 Hz domina el ruido de alta frecuencia
        - Por debajo de 0.1 Hz hay deriva de línea de base

    Las líneas verticales rojas marcan los límites del filtro elegido.
    """
    fig, axes = plt.subplots(1, len(CANALES_INTERES),
                             figsize=(5 * len(CANALES_INTERES), 5),
                             sharey=True)

    fig.suptitle(
        "Espectro de Potencia (PSD) — Señales EEG crudas\n"
        "Método de Welch — Promedio sobre trials de condición S1 obj",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for col, canal in enumerate(CANALES_INTERES):
        ax = axes[col]

        for grupo in ["control", "alcoholic"]:
            print(f"    Calculando PSD: {grupo} — {canal}...")
            freqs, psd_mean, psd_sem = calcular_psd_promedio(
                df, canal, grupo, n_sujetos=15
            )

            # Convertir a dB: 10 * log10(PSD)
            psd_db  = 10 * np.log10(psd_mean + 1e-12)
            sem_db  = 10 * np.log10(psd_mean + psd_sem + 1e-12) - psd_db

            color = colores[grupo]
            ax.plot(freqs, psd_db, color=color, linewidth=1.8,
                    label=grupo.capitalize())
            ax.fill_between(freqs,
                            psd_db - sem_db,
                            psd_db + sem_db,
                            color=color, alpha=0.15)

        # Marcar límites del filtro pasa-banda
        ax.axvline(0.1, color="orange", linestyle="--", linewidth=1.2,
                   label="Corte inferior (0.1 Hz)")
        ax.axvline(30,  color="green",  linestyle="--", linewidth=1.2,
                   label="Corte superior (30 Hz)")

        # Sombrear zonas rechazadas por el filtro
        ax.axvspan(0, 0.1, alpha=0.08, color="red",
                   label="Zona filtrada")
        ax.axvspan(30, freqs[-1], alpha=0.08, color="red")

        ax.set_title(f"Canal: {canal}", fontsize=11)
        ax.set_xlabel("Frecuencia (Hz)")
        if col == 0:
            ax.set_ylabel("Potencia (dB re µV²/Hz)")
        ax.set_xlim([0, FS / 2])
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig("figura_exploracion_psd.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_exploracion_psd.png'")


# =============================================================================
# FUNCIONES — ANÁLISIS DE ARTEFACTOS
# =============================================================================

def graficar_distribucion_amplitudes(df: pd.DataFrame):
    """
    Grafica la distribución de amplitudes de las señales crudas.

    Muestra un histograma del valor_uV de todas las muestras de los
    canales de interés. Permite identificar:
        - Si la distribución es aproximadamente gaussiana (esperado en EEG limpio)
        - La presencia de valores extremos (artefactos)
        - El rango típico de amplitud, que justifica el umbral de ±100 µV
          usado en el rechazo de artefactos del Script 03
    """
    fig, axes = plt.subplots(1, len(CANALES_INTERES),
                             figsize=(4 * len(CANALES_INTERES), 4),
                             sharey=True)

    fig.suptitle(
        "Distribución de amplitudes — Señales crudas\n"
        "Líneas rojas: umbral de rechazo de artefactos (±100 µV)",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for col, canal in enumerate(CANALES_INTERES):
        ax = axes[col]

        for grupo in ["control", "alcoholic"]:
            valores = df[
                (df["grupo"] == grupo) &
                (df["canal"] == canal)
            ]["valor_uV"].values

            ax.hist(valores, bins=120, alpha=0.45,
                    color=colores[grupo], label=grupo.capitalize(),
                    density=True)

        # Umbral de artefactos
        ax.axvline( 100, color="red", linestyle="--",
                   linewidth=1.2, label="±100 µV")
        ax.axvline(-100, color="red", linestyle="--", linewidth=1.2)

        ax.set_title(f"Canal: {canal}", fontsize=11)
        ax.set_xlabel("Amplitud (µV)")
        if col == 0:
            ax.set_ylabel("Densidad")
        ax.set_xlim([-200, 200])
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figura_exploracion_artefactos.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_exploracion_artefactos.png'")


# =============================================================================
# CONCLUSIONES
# =============================================================================

def imprimir_conclusiones(df: pd.DataFrame):
    """
    Imprime un resumen cuantitativo del análisis exploratorio y las
    conclusiones que justifican las decisiones de preprocesamiento.
    """
    print("\n" + "=" * 60)
    print("CONCLUSIONES DEL ANÁLISIS EXPLORATORIO")
    print("=" * 60)

    for canal in CANALES_INTERES:
        print(f"\nCanal {canal}:")
        for grupo in ["control", "alcoholic"]:
            vals = df[
                (df["grupo"] == grupo) &
                (df["canal"] == canal)
            ]["valor_uV"]

            pct_artefactos = 100 * (np.abs(vals) > 100).mean()
            print(f"  {grupo.capitalize():12s} — "
                  f"Media: {vals.mean():6.2f} µV | "
                  f"Std: {vals.std():6.2f} µV | "
                  f"Rango: [{vals.min():.1f}, {vals.max():.1f}] µV | "
                  f"Artefactos (>±100µV): {pct_artefactos:.2f}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 02: Visualización y Análisis Exploratorio")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Cargar datos crudos
    # -------------------------------------------------------------------------
    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 01."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas")
    print(f"  Sujetos: {df['sujeto'].nunique()} "
          f"({df[df['grupo']=='alcoholic']['sujeto'].nunique()} alcohólicos, "
          f"{df[df['grupo']=='control']['sujeto'].nunique()} controles)")

    # Filtrar solo canales de interés para agilizar
    df_canales = df[df["canal"].isin(CANALES_INTERES)].copy()

    # -------------------------------------------------------------------------
    # Análisis en el dominio del tiempo
    # -------------------------------------------------------------------------
    print("\n--- Dominio del tiempo ---")
    print("  Graficando trials individuales (P8, S1 obj)...")
    graficar_señales_tiempo(df_canales)

    print("  Graficando señales en los 4 canales de interés...")
    graficar_señales_multicanal(df_canales)

    # -------------------------------------------------------------------------
    # Análisis en el dominio de la frecuencia
    # -------------------------------------------------------------------------
    print("\n--- Dominio de la frecuencia (PSD - Método de Welch) ---")
    graficar_psd(df_canales)

    # -------------------------------------------------------------------------
    # Análisis de artefactos
    # -------------------------------------------------------------------------
    print("\n--- Distribución de amplitudes ---")
    graficar_distribucion_amplitudes(df_canales)

    # -------------------------------------------------------------------------
    # Conclusiones
    # -------------------------------------------------------------------------
    imprimir_conclusiones(df_canales)

    print("\n[OK] Script 02 finalizado.")
