"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 01: Carga y Exploración Inicial del Dataset
==============================================================================

Dataset: UCI EEG Database
Fuente:  https://archive.ics.uci.edu/dataset/121/eeg+database
Formato: Un archivo .csv por trial, dentro de carpetas por sujeto.

Estructura esperada de archivos:
    eeg_full/
        co2a0000364.tar.gz   <- sujeto alcohólico (4ta letra = 'a')
        co2c0000337.tar.gz   <- sujeto control    (4ta letra = 'c')
        ...

Uso:
    python 01_carga_exploracion.py
==============================================================================
"""

import os
import re
import tarfile
import io
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURACIÓN — ajustá esta ruta a donde tenés descargado el dataset
# =============================================================================

# Ruta a la carpeta que contiene los .tar.gz de cada sujeto
DATA_DIR = Path("C:\Users\Fran\Desktop\TPS PSIB\TPS\PSIB2026Q1-TPS-Grupo11-Cobian_Obeid_Perelstein\data")

# Frecuencia de muestreo del dataset (256 Hz)
FS = 256  # Hz

# Número de muestras por trial (0–255 = 256 muestras = 1 segundo)
N_SAMPLES = 256

# Canales de interés (temporo-occipitales derechos, según Zhang et al. 1997)
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]

# Condiciones de interés
CONDICION_S1 = "S1 obj"
CONDICION_S2_NOMATCH = "S2 nomatch"
CONDICION_S2_MATCH   = "S2 match"

# =============================================================================
# FUNCIONES DE CARGA
# =============================================================================

def identificar_grupo(nombre_sujeto: str) -> str:
    """
    Determina si un sujeto es alcohólico o control a partir del nombre del archivo.
    La 4ta letra del nombre identifica el grupo: 'a' = alcoholic, 'c' = control.

    Ejemplo:
        co2a0000364 -> 'alcoholic'
        co2c0000337 -> 'control'
    """
    # El nombre puede venir con o sin extensión
    base = Path(nombre_sujeto).stem.split(".")[0]
    if len(base) >= 4:
        letra = base[3].lower()
        if letra == "a":
            return "alcoholic"
        elif letra == "c":
            return "control"
    return "unknown"


def parsear_trial(contenido: str, nombre_archivo: str) -> pd.DataFrame | None:
    """
    Parsea el contenido de texto de un trial individual.

    Formato del archivo:
        # co2a0000364.rd          <- línea 1: identificador sujeto
        # 120 trials, 64 chans, 416 samples 368 samples post_stim
        # 3.906000 ms uV
        # S1 obj , trial 0        <- línea 4: condición y número de trial
        # FP1 chan 0               <- inicio datos canal FP1
        0 FP1 0 -8.921            <- trial_num canal muestra valor_uV
        0 FP1 1 -8.433
        ...

    Retorna un DataFrame con columnas:
        trial_num, canal, muestra, valor_uV, sujeto, grupo, condicion
    O None si el trial tiene error o está vacío.
    """
    lineas = contenido.strip().split("\n")

    if len(lineas) < 5:
        return None

    # Extraer información del header
    header = {i: lineas[i].strip() for i in range(min(5, len(lineas)))}

    # Verificar si hay error reportado en el trial
    if any("err" in l.lower() for l in lineas[:10]):
        return None

    # Línea 1: identificador de sujeto (ej: "# co2a0000364.rd")
    match_sujeto = re.search(r"#\s*(\w+)", header.get(0, ""))
    sujeto = match_sujeto.group(1) if match_sujeto else Path(nombre_archivo).stem

    # Línea 4: condición del trial (ej: "# S1 obj , trial 0")
    linea_condicion = header.get(3, "")
    condicion = "unknown"
    for cond in [CONDICION_S1, CONDICION_S2_NOMATCH, CONDICION_S2_MATCH]:
        if cond.lower() in linea_condicion.lower():
            condicion = cond
            break

    grupo = identificar_grupo(sujeto)

    # Parsear líneas de datos (saltar las que empiezan con #)
    filas = []
    for linea in lineas:
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        partes = linea.split()
        if len(partes) == 4:
            try:
                filas.append({
                    "trial_num": int(partes[0]),
                    "canal":     partes[1],
                    "muestra":   int(partes[2]),
                    "valor_uV":  float(partes[3]),
                    "sujeto":    sujeto,
                    "grupo":     grupo,
                    "condicion": condicion,
                })
            except ValueError:
                continue

    if not filas:
        return None

    return pd.DataFrame(filas)


def cargar_sujeto_tar(ruta_tar: Path) -> list[pd.DataFrame]:
    """
    Carga todos los trials de un sujeto desde su archivo .tar.gz.
    Retorna una lista de DataFrames, uno por trial válido.
    """
    trials = []
    try:
        with tarfile.open(ruta_tar, "r:gz") as tar:
            for miembro in tar.getmembers():
                if miembro.isfile():
                    f = tar.extractfile(miembro)
                    if f is None:
                        continue
                    # Cada archivo dentro del tar puede ser .gz o texto plano
                    contenido_raw = f.read()
                    try:
                        # Intentar descomprimir si es .gz
                        if miembro.name.endswith(".gz"):
                            contenido = gzip.decompress(contenido_raw).decode("utf-8", errors="replace")
                        else:
                            contenido = contenido_raw.decode("utf-8", errors="replace")
                    except Exception:
                        continue

                    df_trial = parsear_trial(contenido, miembro.name)
                    if df_trial is not None and not df_trial.empty:
                        trials.append(df_trial)
    except Exception as e:
        print(f"  [!] Error abriendo {ruta_tar.name}: {e}")
    return trials


def cargar_dataset(data_dir: Path, max_sujetos: int = None) -> pd.DataFrame:
    """
    Carga el dataset completo desde la carpeta data_dir.
    Itera sobre todos los .tar.gz de sujetos.

    Args:
        data_dir:     Ruta a la carpeta con los .tar.gz de cada sujeto.
        max_sujetos:  Si se especifica, limita la cantidad de sujetos cargados
                      (útil para pruebas rápidas).

    Retorna un DataFrame combinado con todos los trials.
    """
    archivos_tar = sorted(data_dir.glob("*.tar.gz"))

    if not archivos_tar:
        raise FileNotFoundError(
            f"No se encontraron archivos .tar.gz en '{data_dir}'.\n"
            f"Verificá que DATA_DIR apunte a la carpeta correcta."
        )

    print(f"Encontrados {len(archivos_tar)} archivos de sujetos en '{data_dir}'")

    if max_sujetos is not None:
        archivos_tar = archivos_tar[:max_sujetos]
        print(f"Cargando solo los primeros {max_sujetos} sujetos (modo prueba).")

    todos_trials = []
    for i, ruta_tar in enumerate(archivos_tar):
        grupo = identificar_grupo(ruta_tar.name)
        print(f"  [{i+1:3d}/{len(archivos_tar)}] {ruta_tar.name}  ({grupo})")
        trials = cargar_sujeto_tar(ruta_tar)
        todos_trials.extend(trials)

    if not todos_trials:
        raise ValueError("No se cargó ningún trial válido. Revisá el formato de los archivos.")

    print(f"\nCombinando {len(todos_trials)} trials válidos...")
    df = pd.concat(todos_trials, ignore_index=True)
    return df


# =============================================================================
# FUNCIONES DE EXPLORACIÓN
# =============================================================================

def resumen_dataset(df: pd.DataFrame):
    """Imprime un resumen estadístico del dataset cargado."""
    print("\n" + "="*60)
    print("RESUMEN DEL DATASET")
    print("="*60)

    sujetos = df["sujeto"].unique()
    n_alcoholic = sum(1 for s in sujetos if identificar_grupo(s) == "alcoholic")
    n_control   = sum(1 for s in sujetos if identificar_grupo(s) == "control")

    print(f"Total de sujetos:       {len(sujetos)}")
    print(f"  - Alcohólicos:        {n_alcoholic}")
    print(f"  - Controles:          {n_control}")
    print(f"\nTotal de filas:         {len(df):,}")
    print(f"Canales disponibles:    {sorted(df['canal'].unique())}")
    print(f"Número de canales:      {df['canal'].nunique()}")
    print(f"\nCondiciones presentes:")
    for cond, cnt in df["condicion"].value_counts().items():
        trials_cond = df[df["condicion"] == cond].groupby(["sujeto", "trial_num"]).ngroups
        print(f"  '{cond}': {trials_cond} trials")

    print(f"\nMuestras por trial (esperado 256):")
    muestras_por_trial = (
        df.groupby(["sujeto", "trial_num", "canal"])["muestra"].count()
    )
    print(f"  Min: {muestras_por_trial.min()}, Max: {muestras_por_trial.max()}, "
          f"Mediana: {muestras_por_trial.median():.0f}")

    print(f"\nRango de valores (µV): [{df['valor_uV'].min():.2f}, {df['valor_uV'].max():.2f}]")
    print("="*60)


def verificar_canales_interes(df: pd.DataFrame):
    """Verifica si los canales de interés están presentes en el dataset."""
    canales_disponibles = set(df["canal"].unique())
    print("\nVerificación de canales de interés (Zhang et al. 1997):")
    for canal in CANALES_INTERES:
        estado = "✓ presente" if canal in canales_disponibles else "✗ NO encontrado"
        print(f"  {canal}: {estado}")


def graficar_trial_ejemplo(df: pd.DataFrame, canal: str = "P8"):
    """
    Grafica un trial de ejemplo para un sujeto alcohólico y uno control,
    para la condición S1 obj, en el canal especificado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    fig.suptitle(f"Trial de ejemplo — Canal: {canal} — Condición: S1 obj", fontsize=13)

    tiempo_ms = np.arange(N_SAMPLES) / FS * 1000  # en milisegundos

    for ax, grupo in zip(axes, ["control", "alcoholic"]):
        subset = df[
            (df["grupo"] == grupo) &
            (df["condicion"] == CONDICION_S1) &
            (df["canal"] == canal)
        ]

        if subset.empty:
            ax.set_title(f"{grupo.capitalize()} — sin datos")
            continue

        # Tomar el primer sujeto y primer trial disponibles
        primer_sujeto = subset["sujeto"].iloc[0]
        primer_trial  = subset[subset["sujeto"] == primer_sujeto]["trial_num"].iloc[0]
        trial_data    = subset[
            (subset["sujeto"] == primer_sujeto) &
            (subset["trial_num"] == primer_trial)
        ].sort_values("muestra")

        ax.plot(tiempo_ms, trial_data["valor_uV"].values, linewidth=1.2,
                color="#2563eb" if grupo == "control" else "#dc2626")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, label="Onset S1")
        ax.axvspan(220, 260, alpha=0.15, color="orange", label="Ventana c240")
        ax.set_title(f"{grupo.capitalize()} — Sujeto: {primer_sujeto}")
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylabel("Amplitud (µV)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figura_trial_ejemplo.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figura guardada como 'figura_trial_ejemplo.png'")


def graficar_distribucion_trials(df: pd.DataFrame):
    """Grafica la cantidad de trials por condición y grupo."""
    resumen = (
        df.groupby(["grupo", "condicion", "sujeto", "trial_num"])
        .size()
        .reset_index()
        .groupby(["grupo", "condicion"])
        .size()
        .reset_index(name="n_trials")
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    condiciones = resumen["condicion"].unique()
    x = np.arange(len(condiciones))
    ancho = 0.35

    for i, (grupo, color) in enumerate([("control", "#2563eb"), ("alcoholic", "#dc2626")]):
        valores = [
            resumen[(resumen["grupo"] == grupo) & (resumen["condicion"] == c)]["n_trials"].sum()
            if not resumen[(resumen["grupo"] == grupo) & (resumen["condicion"] == c)].empty else 0
            for c in condiciones
        ]
        ax.bar(x + i * ancho, valores, ancho, label=grupo.capitalize(), color=color, alpha=0.8)

    ax.set_xticks(x + ancho / 2)
    ax.set_xticklabels(condiciones, rotation=15)
    ax.set_ylabel("Número de trials")
    ax.set_title("Distribución de trials por condición y grupo")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figura_distribucion_trials.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figura guardada como 'figura_distribucion_trials.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("="*60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 01: Carga y Exploración Inicial")
    print("="*60)

    # -------------------------------------------------------------------------
    # OPCIÓN A: Carga completa (todos los sujetos)
    # Descomentá esta línea para producción:
    # df = cargar_dataset(DATA_DIR)

    # OPCIÓN B: Carga de prueba (primeros N sujetos, más rápido)
    # Útil para verificar que el parsing funciona antes de cargar todo.
    df = cargar_dataset(DATA_DIR, max_sujetos=10)
    # -------------------------------------------------------------------------

    # Resumen del dataset
    resumen_dataset(df)

    # Verificar canales de interés
    verificar_canales_interes(df)

    # Gráficos exploratorios
    if "P8" in df["canal"].unique():
        graficar_trial_ejemplo(df, canal="P8")
    else:
        # Usar el primer canal disponible si P8 no está
        primer_canal = df["canal"].unique()[0]
        print(f"\n[!] Canal P8 no encontrado, graficando '{primer_canal}' de ejemplo.")
        graficar_trial_ejemplo(df, canal=primer_canal)

    graficar_distribucion_trials(df)

    # Guardar el DataFrame cargado para los scripts siguientes
    salida = "eeg_data_cargado.parquet"
    df.to_parquet(salida, index=False)
    print(f"\nDataset guardado en '{salida}' para uso en scripts posteriores.")
    print("\n[OK] Script 01 finalizado.")
