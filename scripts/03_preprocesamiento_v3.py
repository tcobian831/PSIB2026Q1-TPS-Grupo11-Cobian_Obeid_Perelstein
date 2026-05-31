"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 03 v3: Preprocesamiento unificado
==============================================================================

Este script reemplaza a:
    - 03_preprocesamiento_v2.py
    - 03_preprocesamiento_v2b.py

Qué hace:
    1. Lee eeg_data_cargado.parquet aunque esté en la raíz del repo o en outputs/.
    2. Procesa las tres condiciones necesarias:
        - S1 obj
        - S2 nomatch
        - S2 match
    3. Aplica el mismo pipeline a todas:
        - Filtro Butterworth pasa-banda 0.1-30 Hz, orden 4
        - Corrección de baseline con primeros 30 ms post-estímulo
        - Rechazo de artefactos si supera ±100 µV
    4. Agrupa correctamente por:
        sujeto + grupo + condicion + trial_num + canal
       para evitar mezclar condiciones cuando trial_num se repite.
    5. Guarda salidas nuevas y compatibles con los scripts viejos.

Entradas posibles:
    outputs/eeg_data_cargado.parquet
    eeg_data_cargado.parquet

Salidas principales:
    outputs/eeg_data_preprocesado_v3.parquet

Salidas de compatibilidad:
    outputs/eeg_data_preprocesado_v2.parquet      -> solo S1 obj + S2 nomatch
    outputs/eeg_data_preprocesado_v2b.parquet     -> S1 obj + S2 nomatch + S2 match

También guarda copias en la raíz del repo para que scripts viejos con rutas relativas sigan funcionando.

Uso recomendado desde la raíz del repo:
    python .\\scripts\\03_preprocesamiento_v3.py
==============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENTRADAS_POSIBLES = [
    OUTPUT_DIR / "eeg_data_cargado.parquet",
    PROJECT_DIR / "eeg_data_cargado.parquet",
]

SALIDA_V3_OUTPUTS = OUTPUT_DIR / "eeg_data_preprocesado_v3.parquet"
SALIDA_V2_OUTPUTS = OUTPUT_DIR / "eeg_data_preprocesado_v2.parquet"
SALIDA_V2B_OUTPUTS = OUTPUT_DIR / "eeg_data_preprocesado_v2b.parquet"

# Copias en raíz para compatibilidad con scripts que usan Path("archivo.parquet")
SALIDA_V3_ROOT = PROJECT_DIR / "eeg_data_preprocesado_v3.parquet"
SALIDA_V2_ROOT = PROJECT_DIR / "eeg_data_preprocesado_v2.parquet"
SALIDA_V2B_ROOT = PROJECT_DIR / "eeg_data_preprocesado_v2b.parquet"

SALIDA_RESUMEN = OUTPUT_DIR / "tabla_resumen_preprocesamiento_v3.csv"


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS = 256
N_SAMPLES = 256

F_LOW = 0.1
F_HIGH = 30.0
ORDEN = 4

T_BASELINE_MS = 30
N_BASELINE = int(T_BASELINE_MS / 1000 * FS)

UMBRAL_UV = 100.0

CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]

# V3 procesa todo lo necesario.
CONDICIONES_TODAS = ["S1 obj", "S2 nomatch", "S2 match"]

# Compatibilidad con el pipeline principal viejo.
CONDICIONES_V2 = ["S1 obj", "S2 nomatch"]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def resolver_entrada() -> Path:
    """Devuelve la ruta existente de eeg_data_cargado.parquet."""
    for ruta in ENTRADAS_POSIBLES:
        if ruta.exists():
            return ruta

    rutas_txt = "\n".join(f"  - {ruta}" for ruta in ENTRADAS_POSIBLES)
    raise FileNotFoundError(
        "No se encontró eeg_data_cargado.parquet.\n"
        "Corré primero el Script 01 y verificá que exista en alguna de estas rutas:\n"
        f"{rutas_txt}"
    )


def verificar_columnas(df: pd.DataFrame) -> None:
    """Valida que el DataFrame tenga las columnas mínimas necesarias."""
    columnas_necesarias = [
        "sujeto",
        "grupo",
        "condicion",
        "trial_num",
        "canal",
        "muestra",
        "valor_uV",
    ]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if faltantes:
        raise KeyError(
            "El parquet de entrada no tiene las columnas esperadas.\n"
            f"Columnas faltantes: {faltantes}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )


def disenar_filtro_butterworth(f_low: float, f_high: float, fs: int, orden: int):
    """Diseña filtro pasa-banda Butterworth."""
    nyquist = fs / 2.0
    b, a = butter(orden, [f_low / nyquist, f_high / nyquist], btype="band")
    return b, a


def preprocesar_trial(df_trial: pd.DataFrame, b: np.ndarray, a: np.ndarray) -> tuple[pd.DataFrame | None, str]:
    """
    Preprocesa un trial de un canal.

    Retorna:
        (df_procesado, motivo)
        - df_procesado es None si se rechaza
        - motivo indica OK, largo_incorrecto o artefacto
    """
    df_trial = df_trial.sort_values("muestra").copy()

    if len(df_trial) != N_SAMPLES:
        return None, "largo_incorrecto"

    senal = df_trial["valor_uV"].to_numpy(dtype=float)

    # Filtrado fase cero para no alterar latencias.
    senal = filtfilt(b, a, senal)

    # Baseline temprano 0-30 ms. El dataset no tiene pre-estímulo real.
    senal = senal - np.mean(senal[:N_BASELINE])

    # Rechazo simple de artefactos.
    if np.any(np.abs(senal) > UMBRAL_UV):
        return None, "artefacto"

    df_trial["valor_uV"] = senal
    return df_trial, "OK"


def guardar_parquet(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda un parquet creando carpeta si hace falta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta, index=False)
    print(f"  Guardado: {ruta}")


def imprimir_resumen_trials(df_proc: pd.DataFrame) -> pd.DataFrame:
    """Arma e imprime resumen de trials limpios por grupo y condición."""
    resumen = (
        df_proc.groupby(["grupo", "condicion", "sujeto", "trial_num"])
        .size()
        .reset_index(name="n_filas_trial")
        .groupby(["grupo", "condicion"])
        .size()
        .reset_index(name="n_trials_limpios")
        .sort_values(["condicion", "grupo"])
    )

    print("\nDistribución de trials limpios:")
    print(resumen.to_string(index=False))
    return resumen


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 03 v3: Preprocesamiento unificado")
    print("=" * 70)

    entrada = resolver_entrada()

    print(f"\nLeyendo entrada:")
    print(f"  {entrada}")

    df = pd.read_parquet(entrada)
    verificar_columnas(df)

    print(f"\nDataset cargado:")
    print(f"  Filas:    {len(df):,}")
    print(f"  Sujetos:  {df['sujeto'].nunique()}")
    print(f"  Canales:  {df['canal'].nunique()}")
    print(f"  Condiciones disponibles: {sorted(df['condicion'].dropna().unique())}")

    # -------------------------------------------------------------------------
    # Filtrado por condiciones/canales de interés
    # -------------------------------------------------------------------------
    print("\nFiltrando datos de interés:")
    print(f"  Canales:     {CANALES_INTERES}")
    print(f"  Condiciones: {CONDICIONES_TODAS}")

    df = df[
        df["canal"].isin(CANALES_INTERES)
        & df["condicion"].isin(CONDICIONES_TODAS)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "Después de filtrar por canales y condiciones no quedó ningún dato.\n"
            "Revisá nombres de canales y condiciones en el parquet."
        )

    print(f"  Filas tras filtrado: {len(df):,}")

    # -------------------------------------------------------------------------
    # Diseño del filtro
    # -------------------------------------------------------------------------
    print("\nConfiguración de preprocesamiento:")
    print(f"  Filtro: Butterworth pasa-banda {F_LOW}-{F_HIGH} Hz, orden {ORDEN}")
    print(f"  Baseline: primeros {T_BASELINE_MS} ms ({N_BASELINE} muestras)")
    print(f"  Rechazo de artefactos: ±{UMBRAL_UV:.0f} µV")

    b, a = disenar_filtro_butterworth(F_LOW, F_HIGH, FS, ORDEN)

    # -------------------------------------------------------------------------
    # Preprocesamiento trial por trial
    # IMPORTANTE: condicion está incluida en el groupby para evitar mezclar trials.
    # -------------------------------------------------------------------------
    group_cols = ["sujeto", "grupo", "condicion", "trial_num", "canal"]
    grupos = df.groupby(group_cols, sort=False)
    n_total = len(grupos)

    resultados = []
    conteo_motivos = {
        "OK": 0,
        "largo_incorrecto": 0,
        "artefacto": 0,
    }

    print(f"\nPreprocesando {n_total:,} combinaciones sujeto-condición-trial-canal...")
    for idx, (_, df_trial) in enumerate(grupos):
        if idx % 5000 == 0:
            print(f"  Progreso: {idx:,}/{n_total:,}")

        df_proc, motivo = preprocesar_trial(df_trial, b, a)
        conteo_motivos[motivo] = conteo_motivos.get(motivo, 0) + 1

        if df_proc is not None:
            resultados.append(df_proc)

    print(f"  Progreso: {n_total:,}/{n_total:,}")

    print("\nResultado del preprocesamiento:")
    for motivo, cantidad in conteo_motivos.items():
        pct = 100 * cantidad / n_total if n_total else 0
        print(f"  {motivo:18s}: {cantidad:8,} ({pct:5.1f}%)")

    if not resultados:
        raise RuntimeError("No quedó ningún trial válido después del preprocesamiento.")

    df_proc_v3 = pd.concat(resultados, ignore_index=True)

    resumen = imprimir_resumen_trials(df_proc_v3)
    resumen.to_csv(SALIDA_RESUMEN, index=False)
    print(f"\nResumen guardado: {SALIDA_RESUMEN}")

    # -------------------------------------------------------------------------
    # Guardado de salidas
    # -------------------------------------------------------------------------
    print("\nGuardando salidas principales:")

    # V3: todo lo procesado, incluyendo S2 match.
    guardar_parquet(df_proc_v3, SALIDA_V3_OUTPUTS)
    guardar_parquet(df_proc_v3, SALIDA_V3_ROOT)

    # V2B: compatibilidad, también incluye las 3 condiciones.
    guardar_parquet(df_proc_v3, SALIDA_V2B_OUTPUTS)
    guardar_parquet(df_proc_v3, SALIDA_V2B_ROOT)

    # V2: compatibilidad con pipeline principal viejo, solo S1 obj + S2 nomatch.
    df_proc_v2 = df_proc_v3[df_proc_v3["condicion"].isin(CONDICIONES_V2)].copy()
    guardar_parquet(df_proc_v2, SALIDA_V2_OUTPUTS)
    guardar_parquet(df_proc_v2, SALIDA_V2_ROOT)

    print("\nListo. A partir de ahora podés correr scripts que esperen cualquiera de estos archivos:")
    print("  - eeg_data_preprocesado_v3.parquet")
    print("  - eeg_data_preprocesado_v2.parquet")
    print("  - eeg_data_preprocesado_v2b.parquet")

    print("\n[OK] Script 03 v3 finalizado.")
