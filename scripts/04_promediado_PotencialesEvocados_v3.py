"""
==============================================================================
Script 04 v3: Promediado y cálculo de Potenciales Evocados
==============================================================================

Reemplaza al 04 v2.
Lee el preprocesado v3 si existe; si no, intenta v2b/v2.
Guarda salidas v3 y copias de compatibilidad v2.

Uso:
    python .\\scripts\\04_promediado_PotencialesEvocados_v3.py
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FS = 256
N_SAMPLES = 256
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES_PRINCIPALES = ["S1 obj", "S2 nomatch"]
CONDICIONES_TODAS = ["S1 obj", "S2 nomatch", "S2 match"]
T_C240_INI_MS = 220
T_C240_FIN_MS = 260
ALPHA = 0.05

def get_project_dirs():
    """
    Devuelve PROJECT_DIR y OUTPUT_DIR.
    Está pensado para correr scripts desde la raíz del repo:
        python .\\scripts\\archivo.py
    o desde la carpeta scripts:
        python archivo.py
    """
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "scripts":
        project_dir = here.parent.parent
    else:
        project_dir = here.parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return project_dir, output_dir

PROJECT_DIR, OUTPUT_DIR = get_project_dirs()

def first_existing(paths, label):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    msg = "\n".join(f"  - {Path(p)}" for p in paths)
    raise FileNotFoundError(f"No se encontró {label}. Rutas buscadas:\n{msg}")

def save_csv_compat(df, *paths):
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"  Guardado: {p}")

def save_parquet_compat(df, *paths):
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        print(f"  Guardado: {p}")

def ms_a_muestra(t_ms):
    return int(t_ms / 1000 * FS)

def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    return float((np.mean(x) - np.mean(y)) / sp)

def correccion_bh(p_valores):
    p = np.asarray(p_valores, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    p_sorted = pv[order]
    q_sorted = p_sorted * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q_valid = np.empty(n)
    q_valid[order] = q_sorted
    q[valid] = q_valid
    return q

def flatten_axes(axes):
    return np.array(axes, dtype=object).reshape(-1)

def filter_existing_conditions(df, desired):
    present = list(df["condicion"].dropna().unique())
    return [c for c in desired if c in present]

def safe_ttest_ind(x, y):
    from scipy.stats import ttest_ind
    x = pd.Series(x).dropna().values
    y = pd.Series(y).dropna().values
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    t, p = ttest_ind(x, y, equal_var=False)
    return float(t), float(p)

def safe_ttest_1samp_greater(x, popmean=0.0):
    """p one-sided para H1: mean(x) > popmean."""
    from scipy.stats import ttest_1samp
    x = pd.Series(x).dropna().values
    if len(x) < 2:
        return np.nan, np.nan
    t, p_two = ttest_1samp(x, popmean)
    if np.isnan(t) or np.isnan(p_two):
        return np.nan, np.nan
    p_one = p_two / 2 if t > 0 else 1 - p_two / 2
    return float(t), float(p_one)


ENTRADA = first_existing([
    OUTPUT_DIR / "eeg_data_preprocesado_v3.parquet",
    OUTPUT_DIR / "eeg_data_preprocesado_v2b.parquet",
    OUTPUT_DIR / "eeg_data_preprocesado_v2.parquet",
    PROJECT_DIR / "eeg_data_preprocesado_v3.parquet",
    PROJECT_DIR / "eeg_data_preprocesado_v2b.parquet",
    PROJECT_DIR / "eeg_data_preprocesado_v2.parquet",
], "datos preprocesados")

SALIDA_IND_V3 = OUTPUT_DIR / "eeg_PE_individual_v3.parquet"
SALIDA_GRAND_V3 = OUTPUT_DIR / "eeg_PE_grandaverage_v3.parquet"
SALIDA_IND_V2 = OUTPUT_DIR / "eeg_PE_individual_v2.parquet"
SALIDA_GRAND_V2 = OUTPUT_DIR / "eeg_PE_grandaverage_v2.parquet"

def calcular_pe_individual(df):
    # Promedio de trials por sujeto, canal, condición y muestra.
    # n_trials queda como cantidad de trials válidos que aportaron a cada punto.
    return (
        df.groupby(["sujeto", "grupo", "canal", "condicion", "muestra"], observed=True)["valor_uV"]
        .agg(PE_uV="mean", n_trials="count")
        .reset_index()
    )

def calcular_grand_average(pe_ind):
    grand = (
        pe_ind.groupby(["grupo", "canal", "condicion", "muestra"], observed=True)["PE_uV"]
        .agg(grand_avg_uV="mean", std_uV="std", n_sujetos="count")
        .reset_index()
    )
    grand["sem_uV"] = grand["std_uV"] / np.sqrt(grand["n_sujetos"].clip(lower=1))
    return grand

def tiempo_ms(muestra):
    return np.asarray(muestra) / FS * 1000

def graficar_grand_average(grand, condiciones):
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    n_rows = len(condiciones)
    n_cols = len(CANALES_INTERES)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    fig.suptitle(
        "Grand Average PE — Alcoholic vs Control\n"
        "Banda sombreada: ±1 SEM entre sujetos",
        fontsize=13,
        y=1.01
    )

    for i, condicion in enumerate(condiciones):
        for j, canal in enumerate(CANALES_INTERES):
            ax = axes[i, j]
            subset = grand[(grand["canal"] == canal) & (grand["condicion"] == condicion)]
            if subset.empty:
                ax.set_title(f"{canal} / {condicion}\nSin datos")
                continue

            for grupo in ["control", "alcoholic"]:
                datos = subset[subset["grupo"] == grupo].sort_values("muestra")
                if datos.empty:
                    continue
                t = tiempo_ms(datos["muestra"].values)
                avg = datos["grand_avg_uV"].values
                sem = datos["sem_uV"].fillna(0).values
                n = int(datos["n_sujetos"].iloc[0])
                color = colores.get(grupo, None)
                ax.plot(t, avg, linewidth=1.8, color=color, label=f"{grupo} (n={n})")
                ax.fill_between(t, avg - sem, avg + sem, color=color, alpha=0.15)

            ax.axvspan(T_C240_INI_MS, T_C240_FIN_MS, alpha=0.12, color="orange", label="Ventana c240")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"{canal} — {condicion}", fontsize=10)
            ax.set_xlabel("Tiempo (ms)")
            ax.set_ylabel("Amplitud (µV)")
            ax.grid(True, alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_grand_average_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

def resumen_pe(grand, condiciones):
    print("\n" + "=" * 70)
    print("RESUMEN GRAND AVERAGE — pico por magnitud absoluta en ventana c240")
    print("=" * 70)
    m_ini = ms_a_muestra(T_C240_INI_MS)
    m_fin = ms_a_muestra(T_C240_FIN_MS)

    for condicion in condiciones:
        print(f"\nCondición: {condicion}")
        print(f"  {'Canal':<6} {'Control (µV)':>14} {'Alcoholic (µV)':>16}")
        for canal in CANALES_INTERES:
            vals = []
            for grupo in ["control", "alcoholic"]:
                datos = grand[
                    (grand["grupo"] == grupo)
                    & (grand["canal"] == canal)
                    & (grand["condicion"] == condicion)
                    & (grand["muestra"].between(m_ini, m_fin))
                ]
                if datos.empty:
                    vals.append(np.nan)
                else:
                    idx = datos["grand_avg_uV"].abs().idxmax()
                    vals.append(datos.loc[idx, "grand_avg_uV"])
            print(f"  {canal:<6} {vals[0]:>14.3f} {vals[1]:>16.3f}")
    print("=" * 70)

if __name__ == "__main__":
    print("=" * 70)
    print("Script 04 v3: Promediado de Potenciales Evocados")
    print("=" * 70)

    print(f"\nCargando: {ENTRADA}")
    df = pd.read_parquet(ENTRADA)
    condiciones = filter_existing_conditions(df, CONDICIONES_TODAS)
    print(f"  Filas: {len(df):,}")
    print(f"  Sujetos: {df['sujeto'].nunique()}")
    print(f"  Condiciones usadas: {condiciones}")

    df = df[df["canal"].isin(CANALES_INTERES) & df["condicion"].isin(condiciones)].copy()

    print("\nCalculando PE individual...")
    pe_ind = calcular_pe_individual(df)
    print(f"  Puntos PE individual: {len(pe_ind):,}")

    print("\nCalculando Grand Average...")
    grand = calcular_grand_average(pe_ind)
    print(f"  Puntos Grand Average: {len(grand):,}")

    print("\nGuardando salidas v3...")
    save_parquet_compat(pe_ind, SALIDA_IND_V3, PROJECT_DIR / "eeg_PE_individual_v3.parquet")
    save_parquet_compat(grand, SALIDA_GRAND_V3, PROJECT_DIR / "eeg_PE_grandaverage_v3.parquet")

    # Compatibilidad v2: solo condiciones principales.
    pe_ind_v2 = pe_ind[pe_ind["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()
    grand_v2 = grand[grand["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()
    print("\nGuardando copias de compatibilidad v2...")
    save_parquet_compat(pe_ind_v2, SALIDA_IND_V2, PROJECT_DIR / "eeg_PE_individual_v2.parquet")
    save_parquet_compat(grand_v2, SALIDA_GRAND_V2, PROJECT_DIR / "eeg_PE_grandaverage_v2.parquet")

    resumen_pe(grand, condiciones)
    graficar_grand_average(grand, condiciones)

    print("\n[OK] Script 04 v3 finalizado.")
