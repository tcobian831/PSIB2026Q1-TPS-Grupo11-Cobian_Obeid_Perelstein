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
Salidas:  distintas figuras de exploración guardadas en outputs/

Uso:
    Correr desde la carpeta scripts/
    python 02_exploracion_inicial.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import welch
import os
import sys

# El script corre desde cualquier carpeta: anclamos el CWD a la raiz del proyecto
# (donde esta outputs/) para que todas las rutas relativas resuelvan igual.
os.chdir(Path(__file__).resolve().parent.parent)
Path("outputs").mkdir(exist_ok=True)

# Salida UTF-8 robusta: evita UnicodeEncodeError al redirigir/pipear en Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS        = 256    # Frecuencia de muestreo (Hz)
N_SAMPLES = 256    # Muestras por trial

# Canales de interés
CANALES_DERECHO   = ["P8",  "PO8",  "T8",  "TP8"]
CANALES_IZQUIERDO = ["P7",  "PO7",  "T7",  "TP7"]
CANALES_INTERES   = CANALES_DERECHO + CANALES_IZQUIERDO

# Cuántos trials de ejemplo mostrar por grupo en el dominio del tiempo
N_TRIALS_EJEMPLO = 5

ENTRADA = Path("outputs/eeg_data_cargado.parquet") 

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

    # Identificar el sujeto de ejemplo de cada grupo (P8, S1 obj) para el titulo.
    # Misma logica (.iloc[0]) que el loop de abajo -> el codigo coincide con el
    # sujeto realmente graficado en cada fila.
    sujetos_ej = {}
    for grupo in ["control", "alcoholic"]:
        s = df[
            (df["grupo"] == grupo) &
            (df["canal"] == "P8") &
            (df["condicion"] == "S1 obj")
        ]
        sujetos_ej[grupo] = s["sujeto"].iloc[0] if not s.empty else "—"

    fig.suptitle(
        "Señales EEG crudas — Canal P8 — Condición S1 obj\n"
        f"Sujetos de ejemplo: {sujetos_ej['control']} (control) — "
        f"{sujetos_ej['alcoholic']} (alcoholic)  |  "
        f"{N_TRIALS_EJEMPLO} trials individuales por grupo",
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
    plt.savefig("outputs/figura_exploracion_tiempo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_exploracion_tiempo.png'")


def graficar_señales_multicanal(df: pd.DataFrame, 
                                 canales: list, 
                                 titulo_sufijo: str,
                                 nombre_archivo: str):
    """
    Grafica un trial de ejemplo en los canales especificados,
    para control y alcohólico.
    """
    fig, axes = plt.subplots(2, len(canales),
                             figsize=(4 * len(canales), 6),
                             sharex=True)
    
    # Identificar sujeto y trial de ejemplo para cada grupo
    ejemplos = {}

    for grupo in ["control", "alcoholic"]:
        ejemplo = df[
            (df["grupo"] == grupo) &
            (df["canal"] == canales[0]) &
            (df["condicion"] == "S1 obj")
        ]

        if ejemplo.empty:
            ejemplos[grupo] = ("—", "—")
        else:
            sujeto = ejemplo["sujeto"].iloc[0]
            trial = ejemplo[ejemplo["sujeto"] == sujeto]["trial_num"].iloc[0]
            ejemplos[grupo] = (sujeto, trial)

    fig.suptitle(
        f"Señal EEG cruda — {titulo_sufijo}\n"
        f"Control: {ejemplos['control'][0]}, trial {ejemplos['control'][1]} — "
        f"Alcoholic: {ejemplos['alcoholic'][0]}, trial {ejemplos['alcoholic'][1]} — "
        f"Condición: S1 obj",
        fontsize=12
    )

    tiempo_ms = np.arange(N_SAMPLES) / FS * 1000
    colores   = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for fila, grupo in enumerate(["control", "alcoholic"]):
        for col, canal in enumerate(canales):
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
                ax.set_ylabel(f"{grupo.capitalize()}\nAmplitud (µV)", 
                              fontsize=9)
            if fila == 0:
                ax.set_title(f"Canal: {canal}", fontsize=10)
            if fila == 1:
                ax.set_xlabel("Tiempo (ms)", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"outputs/{nombre_archivo}", 
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{nombre_archivo}'")


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


def graficar_psd(df: pd.DataFrame, canales: list,
                 titulo_sufijo: str, nombre_archivo: str):
    """
    Grafica el Power Spectral Density (PSD) de las señales crudas.
    Estimado por el método de Welch (ventanas de 128 muestras, 50% solapamiento).

    Dos paneles por canal:
    - Izquierdo: espectro completo (0-128 Hz)
    - Derecho: zoom en la banda de interés (0-30 Hz)
    """
    fig, axes = plt.subplots(
        len(canales), 2,
        figsize=(14, 4 * len(canales))
    )

    fig.suptitle(
        f"Espectro de Potencia (PSD) — Señales EEG crudas — {titulo_sufijo}\n"
        "Estimado por el método de Welch "
        "(ventanas de 128 muestras, solapamiento 50%)",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for fila, canal in enumerate(canales):
        ax_full = axes[fila][0]
        ax_zoom = axes[fila][1]

        for grupo in ["control", "alcoholic"]:
            print(f"    Calculando PSD: {grupo} — {canal}...")
            freqs, psd_mean, psd_sem = calcular_psd_promedio(
                df, canal, grupo
            )
            psd_db = 10 * np.log10(psd_mean + 1e-12)
            sem_db = 10 * np.log10(psd_mean + psd_sem + 1e-12) - psd_db
            color  = colores[grupo]

            for ax in [ax_full, ax_zoom]:
                ax.plot(freqs, psd_db, color=color,
                        linewidth=1.8, label=grupo.capitalize())
                ax.fill_between(freqs,
                                psd_db - sem_db,
                                psd_db + sem_db,
                                color=color, alpha=0.15)

        # Panel izquierdo — espectro completo
        ax_full.axvspan(0, 30, alpha=0.12, color="green",
                        label="Banda conservada (0-30 Hz)")
        ax_full.axvline(30, color="green", linestyle="--",
                        linewidth=1.5, label="Corte superior (30 Hz)")
        ax_full.axvline(0.1, color="blue", linestyle=":",
                        linewidth=1.2, label="Corte inferior (0.1 Hz)")
        ax_full.set_xlim([0, FS / 2])
        ax_full.set_title(f"Canal {canal} — Espectro completo (0-128 Hz)",
                          fontsize=10)
        ax_full.set_xlabel("Frecuencia (Hz)")
        ax_full.set_ylabel("Potencia (dB)", fontsize=9)
        ax_full.grid(True, alpha=0.3)
        if fila == 0:
            ax_full.legend(fontsize=8, loc="upper right")

        # Panel derecho — zoom banda de interés
        ax_zoom.axvspan(0, 30, alpha=0.12, color="green")
        ax_zoom.axvline(30, color="green", linestyle="--", linewidth=1.5)
        ax_zoom.set_xlim([0, 35])
        ax_zoom.set_title(f"Canal {canal} — Zoom banda de interés (0-35 Hz)",
                          fontsize=10)
        ax_zoom.set_xlabel("Frecuencia (Hz)")
        ax_zoom.set_ylabel("Potencia (dB)", fontsize=9)
        ax_zoom.grid(True, alpha=0.3)
        if fila == 0:
            ax_zoom.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"outputs/{nombre_archivo}",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{nombre_archivo}'")


# =============================================================================
# FUNCIONES — ANÁLISIS DE ARTEFACTOS
# =============================================================================

def graficar_distribucion_amplitudes(df: pd.DataFrame, canales: list,
                                      titulo_sufijo: str,
                                      nombre_archivo: str):
    """
    Grafica la distribución de amplitudes de las señales crudas.

    Muestra un histograma del valor_uV de todas las muestras de los
    canales especificados. Permite identificar:
        - Si la distribución es aproximadamente gaussiana (esperado en EEG limpio)
        - La presencia de valores extremos (artefactos)
        - El rango típico de amplitud, que justifica el umbral de ±100 µV
          usado en el rechazo de artefactos del Script 03
    """
    fig, axes = plt.subplots(1, len(canales),
                             figsize=(4 * len(canales), 4),
                             sharey=True)

    fig.suptitle(
        f"Distribución de amplitudes — Señales crudas — {titulo_sufijo}\n"
        "Líneas rojas: umbral de rechazo de artefactos (±100 µV)",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for col, canal in enumerate(canales):
        ax = axes[col]

        for grupo in ["control", "alcoholic"]:
            valores = df[
                (df["grupo"] == grupo) &
                (df["canal"] == canal)
            ]["valor_uV"].values

            ax.hist(valores, bins=120, alpha=0.45,
                    color=colores[grupo], label=grupo.capitalize(),
                    density=True)

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
    plt.savefig(f"outputs/{nombre_archivo}",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{nombre_archivo}'")


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

    print("  Graficando señales en los 8 canales de interés...")
    
    graficar_señales_multicanal(
        df,
        canales=CANALES_DERECHO,
        titulo_sufijo="Hemisferio derecho (P8, PO8, T8, TP8)",
        nombre_archivo="figura_exploracion_multicanal_derecho.png"
    )

    graficar_señales_multicanal(
        df,
        canales=CANALES_IZQUIERDO,
        titulo_sufijo="Hemisferio izquierdo (P7, PO7, T7, TP7)",
        nombre_archivo="figura_exploracion_multicanal_izquierdo.png"
    )

    # -------------------------------------------------------------------------
    # Análisis en el dominio de la frecuencia
    # -------------------------------------------------------------------------
    print("\n--- Dominio de la frecuencia (PSD - Método de Welch) ---")
    
    graficar_psd(df_canales,
             canales=CANALES_DERECHO,
             titulo_sufijo="Hemisferio derecho (P8, PO8, T8, TP8)",
             nombre_archivo="figura_exploracion_psd_derecho.png")

    graficar_psd(df_canales,
             canales=CANALES_IZQUIERDO,
             titulo_sufijo="Hemisferio izquierdo (P7, PO7, T7, TP7)",
             nombre_archivo="figura_exploracion_psd_izquierdo.png")

    # -------------------------------------------------------------------------
    # Análisis de artefactos
    # -------------------------------------------------------------------------
    print("\n--- Distribución de amplitudes ---")
    
    graficar_distribucion_amplitudes(
    df_canales,
    canales=CANALES_DERECHO,
    titulo_sufijo="Hemisferio derecho (P8, PO8, T8, TP8)",
    nombre_archivo="figura_exploracion_artefactos_derecho.png"
    )

    graficar_distribucion_amplitudes(
    df_canales,
    canales=CANALES_IZQUIERDO,
    titulo_sufijo="Hemisferio izquierdo (P7, PO7, T7, TP7)",
    nombre_archivo="figura_exploracion_artefactos_izquierdo.png"
    )

    # -------------------------------------------------------------------------
    # Conclusiones
    # -------------------------------------------------------------------------
    imprimir_conclusiones(df_canales)

    # -------------------------------------------------------------------------
    # Análisis de trials por sujeto y condición
    # -------------------------------------------------------------------------
    print("\n--- Cantidad de trials por sujeto ---")

    resumen_trials = (
        df_canales[df_canales["canal"] == "P8"]
        .groupby(["grupo", "condicion", "sujeto"])["trial_num"]
        .nunique()
        .reset_index()
        .groupby(["grupo", "condicion"])["trial_num"]
        .agg(media="mean", minimo="min", maximo="max")
        .round(1)
    )
    print(resumen_trials)

    print("\n[OK] Script 02 finalizado.")