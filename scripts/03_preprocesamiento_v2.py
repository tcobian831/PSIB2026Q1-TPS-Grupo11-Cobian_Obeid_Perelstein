"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 03: Preprocesamiento de Señales EEG  [VERSIÓN 2 — modificado]
==============================================================================

Pasos:
    1. Filtro pasa-banda Butterworth (0.1 – 30 Hz)
    2. Corrección de baseline (primeros 50 ms)
    3. Selección de condiciones y canales de interés
    4. Rechazo de trials con artefactos (umbral ±100 µV)

Cambios respecto a la versión anterior:
    - Se agregaron los canales homólogos del HEMISFERIO IZQUIERDO
      (P7, PO7, T7, TP7) para poder replicar y comparar la asimetría
      hemisférica reportada en Zhang et al. (1997), quienes encontraron
      que las diferencias entre grupos se concentran principalmente en
      el hemisferio derecho.

    Hemisferio derecho (referencia del paper):  P8, PO8, T8, TP8
    Hemisferio izquierdo (homólogos):           P7, PO7, T7, TP7

    P8 es el electrodo de referencia del paper (mayor amplitud del VMP).
    Su homólogo izquierdo P7 permite la comparación directa.

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

# Ventana de baseline: primeros 50 ms = primeras 13 muestras a 256 Hz
N_BASELINE = int(0.050 * FS)   # = 12 muestras (índices 0–11)

# Umbral de rechazo de artefactos (µV)
UMBRAL_UV = 100.0

# ---------------------------------------------------------------------------
# Canales de interés — hemisferio derecho (referencia del paper) + izquierdo
#
# Zhang et al. (1997) eligieron P8 como electrodo de referencia por tener la
# mayor amplitud y morfología más consistente del VMP. El análisis estadístico
# del paper se organizó en regiones; la diferencia entre grupos fue más fuerte
# en el hemisferio derecho (temporal y frontal derechos).
#
# Agregamos los 4 homólogos izquierdos para poder REPLICAR y COMPARAR la
# asimetría hemisférica: si nuestros datos muestran el mismo patrón que el
# paper (derecho > izquierdo en la diferencia control − alcohólico), eso
# fortalece nuestras conclusiones.
# ---------------------------------------------------------------------------
CANALES_DERECHO   = ["P8",  "PO8",  "T8",  "TP8"]   # hemisferio derecho
CANALES_IZQUIERDO = ["P7",  "PO7",  "T7",  "TP7"]   # homólogos izquierdos
CANALES_INTERES   = CANALES_DERECHO + CANALES_IZQUIERDO   # 8 canales en total

# Condiciones de interés
CONDICIONES_INTERES = ["S1 obj", "S2 nomatch"]

# Rutas
ENTRADA  = Path("../outputs/eeg_data_cargado.parquet")
SALIDA   = Path("../outputs/eeg_data_preprocesado.parquet")

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
    return filtfilt(b, a, señal)


def corregir_baseline(señal: np.ndarray, n_baseline: int) -> np.ndarray:
    """
    Corrección de baseline: resta el promedio de los primeros n_baseline
    puntos a toda la señal.

    En ERPs, el baseline son los milisegundos previos al estímulo.
    Como en este dataset el estímulo ocurre en t=0 (muestra 0), usamos
    los primeros ~50 ms como aproximación al pre-estímulo.

    Args:
        señal:      array 1D de 256 muestras
        n_baseline: cantidad de muestras a usar como baseline (ej: 12)

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
                      b: np.ndarray, a: np.ndarray):
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

    print("=" * 64)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 03: Preprocesamiento  [v2 — 8 canales, 2 hemisferios]")
    print("=" * 64)

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
    print(f"\nCanales seleccionados:")
    print(f"  Hemisferio derecho:    {CANALES_DERECHO}")
    print(f"  Hemisferio izquierdo:  {CANALES_IZQUIERDO}")
    print(f"  Total: {len(CANALES_INTERES)} canales")
    print(f"\nCondiciones: {CONDICIONES_INTERES}")

    # Verificar que los canales izquierdos existen en el dataset
    canales_disponibles = df["canal"].unique().tolist()
    canales_faltantes = [c for c in CANALES_INTERES
                         if c not in canales_disponibles]
    if canales_faltantes:
        print(f"\n  ADVERTENCIA: canales no encontrados en el dataset: "
              f"{canales_faltantes}")
        CANALES_INTERES = [c for c in CANALES_INTERES
                           if c in canales_disponibles]
        print(f"  Continuando con: {CANALES_INTERES}")
    else:
        print(f"  Todos los canales verificados en el dataset.")

    df = df[
        df["condicion"].isin(CONDICIONES_INTERES) &
        df["canal"].isin(CANALES_INTERES)
    ].copy()

    print(f"\n  Filas tras filtrado: {len(df):,}")
    print(f"  Trials únicos: "
          f"{df.groupby(['sujeto','trial_num','canal']).ngroups:,}")

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
    n_total      = len(grupos)
    n_ok         = 0
    n_rechazados = 0
    resultados   = []

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
    # Paso 4: Verificación por hemisferio, grupo y condición
    # -------------------------------------------------------------------------
    print("\nDistribución de trials limpios por hemisferio, grupo y condición:")

    df_proc["hemisferio"] = df_proc["canal"].apply(
        lambda c: "derecho" if c in CANALES_DERECHO else "izquierdo"
    )

    resumen = (
        df_proc.groupby(["hemisferio", "grupo", "condicion", "sujeto", "trial_num"])
        .size()
        .reset_index()
        .groupby(["hemisferio", "grupo", "condicion"])
        .size()
        .reset_index(name="n_trials")
    )
    print(resumen.to_string(index=False))

    # -------------------------------------------------------------------------
    # Paso 5: Gráfico comparativo — señal cruda vs preprocesada (canal P8)
    # -------------------------------------------------------------------------
    print("\nGenerando gráfico comparativo cruda vs preprocesada (P8)...")

    sujeto_ej = df_proc[df_proc["grupo"] == "control"]["sujeto"].iloc[0]
    trial_ej  = df_proc[
        (df_proc["sujeto"] == sujeto_ej) &
        (df_proc["canal"] == "P8") &
        (df_proc["condicion"] == "S1 obj")
    ]["trial_num"].iloc[0]

    señal_proc = df_proc[
        (df_proc["sujeto"] == sujeto_ej) &
        (df_proc["trial_num"] == trial_ej) &
        (df_proc["canal"] == "P8")
    ].sort_values("muestra")["valor_uV"].values

    df_cruda_ej = df[
        (df["sujeto"] == sujeto_ej) &
        (df["trial_num"] == trial_ej) &
        (df["canal"] == "P8")
    ].sort_values("muestra")
    señal_cruda = df_cruda_ej["valor_uV"].values

    tiempo_ms = np.arange(256) / FS * 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(
        f"Efecto del preprocesamiento — Canal P8 — "
        f"Sujeto: {sujeto_ej} (control) — S1 obj",
        fontsize=12
    )

    axes[0].plot(tiempo_ms, señal_cruda, color="#64748b", linewidth=1)
    axes[0].set_ylabel("Amplitud (µV)")
    axes[0].set_title("Señal cruda")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tiempo_ms, señal_proc, color="#2563eb", linewidth=1)
    axes[1].set_ylabel("Amplitud (µV)")
    axes[1].set_title(
        f"Señal preprocesada (Butterworth {F_LOW}–{F_HIGH} Hz + baseline)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvspan(220, 260, alpha=0.15, color="orange", label="Ventana c240")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    diferencia = señal_cruda[:len(señal_proc)] - señal_proc
    axes[2].plot(tiempo_ms[:len(diferencia)], diferencia,
                 color="#dc2626", linewidth=1)
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_xlabel("Tiempo (ms)")
    axes[2].set_ylabel("Amplitud (µV)")
    axes[2].set_title(
        "Diferencia (cruda − preprocesada) = componentes eliminados por el filtro"
    )
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_preprocesamiento.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada como 'figura_preprocesamiento.png'")

    # -------------------------------------------------------------------------
    # Guardar resultado (sin la columna auxiliar 'hemisferio')
    # -------------------------------------------------------------------------
    df_proc.drop(columns=["hemisferio"], inplace=True)
    df_proc.to_parquet(SALIDA, index=False)
    print(f"\nDatos preprocesados guardados en '{SALIDA}'")
    print(f"  Canales incluidos: {sorted(df_proc['canal'].unique().tolist())}")
    print("\n[OK] Script 03 finalizado.")
