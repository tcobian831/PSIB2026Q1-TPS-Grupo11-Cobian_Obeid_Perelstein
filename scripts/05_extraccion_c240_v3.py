"""
==============================================================================
Script 05 v3: Extracción del componente c240/VMP
==============================================================================

Reemplaza al 05 v2.
Extrae el pico de mayor magnitud absoluta en la ventana c240, conservando signo.
Procesa S1 obj, S2 nomatch y S2 match si existen.

Uso:
    python .\\scripts\\05_extraccion_c240_v3.py
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


T_C240_INI_SENS_MS = 200
T_C240_FIN_SENS_MS = 280

ENTRADA = first_existing([
    OUTPUT_DIR / "eeg_PE_individual_v3.parquet",
    OUTPUT_DIR / "eeg_PE_individual_v2.parquet",
    PROJECT_DIR / "eeg_PE_individual_v3.parquet",
    PROJECT_DIR / "eeg_PE_individual_v2.parquet",
], "PE individual")

SALIDA_V3 = OUTPUT_DIR / "eeg_c240_extraido_v3.csv"
SALIDA_V3_SENS = OUTPUT_DIR / "eeg_c240_extraido_v3_sens.csv"
SALIDA_V2 = OUTPUT_DIR / "eeg_c240_extraido_v2.csv"
SALIDA_V2_SENS = OUTPUT_DIR / "eeg_c240_extraido_v2_sens.csv"
SALIDA_MATCH = OUTPUT_DIR / "eeg_c240_match_extraido_v3.csv"

def extraer_c240_sujeto(df_sujeto, m_ini, m_fin):
    ventana = df_sujeto[df_sujeto["muestra"].between(m_ini, m_fin)]
    if ventana.empty:
        return {
            "amplitud_uV": np.nan,
            "amplitud_abs_uV": np.nan,
            "latencia_ms": np.nan,
            "polaridad": "n/a",
            "n_trials": 0,
        }
    idx = ventana["PE_uV"].abs().idxmax()
    amp = float(ventana.loc[idx, "PE_uV"])
    muestra = int(ventana.loc[idx, "muestra"])
    n_trials = int(round(float(ventana["n_trials"].median()))) if "n_trials" in ventana else 0
    return {
        "amplitud_uV": amp,
        "amplitud_abs_uV": abs(amp),
        "latencia_ms": muestra / FS * 1000,
        "polaridad": "+" if amp >= 0 else "-",
        "n_trials": n_trials,
    }

def extraer_todos(pe_ind, m_ini, m_fin, etiqueta):
    filas = []
    grupos = pe_ind.groupby(["sujeto", "grupo", "canal", "condicion"], observed=True)
    print(f"  Extrayendo c240 [{etiqueta}] de {len(grupos):,} sujeto×canal×condición...")
    for (sujeto, grupo, canal, condicion), df_sub in grupos:
        vals = extraer_c240_sujeto(df_sub.sort_values("muestra"), m_ini, m_fin)
        filas.append({
            "sujeto": sujeto,
            "grupo": grupo,
            "canal": canal,
            "condicion": condicion,
            **vals,
        })
    return pd.DataFrame(filas)

def resumen_c240(df_c240, condiciones, etiqueta):
    print("\n" + "=" * 80)
    print(f"RESUMEN c240/VMP — {etiqueta}")
    print("=" * 80)
    for condicion in condiciones:
        print(f"\nCondición: {condicion}")
        print(f"  {'Canal':<6} {'Grupo':<11} {'N':>4} {'|Amp| media':>13} {'SD':>8} {'Lat media':>11} {'% pico -':>9}")
        for canal in CANALES_INTERES:
            for grupo in ["control", "alcoholic"]:
                sub = df_c240[
                    (df_c240["canal"] == canal)
                    & (df_c240["condicion"] == condicion)
                    & (df_c240["grupo"] == grupo)
                ]
                amp = sub["amplitud_abs_uV"].dropna()
                lat = sub["latencia_ms"].dropna()
                pct_neg = 100 * (sub["polaridad"] == "-").mean() if len(sub) else np.nan
                print(f"  {canal:<6} {grupo:<11} {len(amp):>4} {amp.mean():>13.3f} {amp.std():>8.3f} {lat.mean():>10.1f} {pct_neg:>8.1f}%")
    print("=" * 80)

def plot_boxplot(df_c240, condiciones):
    n_rows, n_cols = len(condiciones), len(CANALES_INTERES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.8*n_rows), squeeze=False)
    fig.suptitle("Magnitud del componente c240/VMP por sujeto — |amplitud pico|", fontsize=13)
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    rng = np.random.default_rng(42)
    for i, condicion in enumerate(condiciones):
        for j, canal in enumerate(CANALES_INTERES):
            ax = axes[i, j]
            data, labels, cols = [], [], []
            for grupo in ["control", "alcoholic"]:
                vals = df_c240[
                    (df_c240["canal"] == canal)
                    & (df_c240["condicion"] == condicion)
                    & (df_c240["grupo"] == grupo)
                ]["amplitud_abs_uV"].dropna().values
                data.append(vals)
                labels.append(f"{grupo}\n(n={len(vals)})")
                cols.append(colores[grupo])
            bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
            for patch, col in zip(bp["boxes"], cols):
                patch.set_facecolor(col); patch.set_alpha(0.55)
            for k, vals in enumerate(data):
                ax.scatter(rng.normal(k+1, 0.06, len(vals)), vals, s=14, alpha=0.35, color=cols[k])
            ax.set_title(f"{canal} — {condicion}", fontsize=9)
            ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("|Amplitud c240| (µV)")
            ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_c240_boxplot_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

def plot_latencias(df_c240, condiciones):
    n_rows, n_cols = len(condiciones), len(CANALES_INTERES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.6*n_rows), squeeze=False, sharex=True)
    fig.suptitle("Distribución de latencias del pico c240/VMP", fontsize=13)
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    bins = np.linspace(T_C240_INI_MS, T_C240_FIN_MS, 12)

    for i, condicion in enumerate(condiciones):
        for j, canal in enumerate(CANALES_INTERES):
            ax = axes[i, j]
            for grupo in ["control", "alcoholic"]:
                lats = df_c240[
                    (df_c240["canal"] == canal)
                    & (df_c240["condicion"] == condicion)
                    & (df_c240["grupo"] == grupo)
                ]["latencia_ms"].dropna().values
                ax.hist(lats, bins=bins, alpha=0.5, color=colores[grupo], label=grupo, density=True)
            ax.axvline(240, color="orange", linestyle="--", linewidth=1.0)
            ax.set_title(f"{canal} — {condicion}", fontsize=9)
            ax.set_xlabel("Latencia (ms)")
            ax.set_ylabel("Densidad")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_c240_latencias_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

def plot_polaridad(df_c240, condiciones):
    fig, axes = plt.subplots(1, len(condiciones), figsize=(6*len(condiciones), 4), squeeze=False, sharey=True)
    axes = axes[0]
    fig.suptitle("Porcentaje de sujetos con pico c240 negativo", fontsize=12)
    x = np.arange(len(CANALES_INTERES)); ancho = 0.35

    for ax, condicion in zip(axes, condiciones):
        for i, grupo in enumerate(["control", "alcoholic"]):
            vals = []
            for canal in CANALES_INTERES:
                sub = df_c240[(df_c240["canal"] == canal) & (df_c240["condicion"] == condicion) & (df_c240["grupo"] == grupo)]
                vals.append(100 * (sub["polaridad"] == "-").mean() if len(sub) else 0)
            ax.bar(x + i*ancho, vals, ancho, label=grupo, alpha=0.75)
        ax.set_xticks(x + ancho/2); ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(condicion)
        ax.set_ylabel("% pico negativo")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_c240_polaridad_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

if __name__ == "__main__":
    print("=" * 70)
    print("Script 05 v3: Extracción c240/VMP")
    print("=" * 70)

    print(f"\nCargando: {ENTRADA}")
    pe_ind = pd.read_parquet(ENTRADA)
    pe_ind = pe_ind[pe_ind["canal"].isin(CANALES_INTERES)].copy()
    condiciones = filter_existing_conditions(pe_ind, CONDICIONES_TODAS)
    pe_ind = pe_ind[pe_ind["condicion"].isin(condiciones)].copy()
    print(f"  Sujetos: {pe_ind['sujeto'].nunique()}")
    print(f"  Condiciones: {condiciones}")

    m_ini, m_fin = ms_a_muestra(T_C240_INI_MS), ms_a_muestra(T_C240_FIN_MS)
    df_c240 = extraer_todos(pe_ind, m_ini, m_fin, "220-260 ms")
    resumen_c240(df_c240, condiciones, "ventana principal 220-260 ms")

    m_ini_s, m_fin_s = ms_a_muestra(T_C240_INI_SENS_MS), ms_a_muestra(T_C240_FIN_SENS_MS)
    df_sens = extraer_todos(pe_ind, m_ini_s, m_fin_s, "200-280 ms")

    print("\nGuardando salidas...")
    save_csv_compat(df_c240, SALIDA_V3, PROJECT_DIR / "eeg_c240_extraido_v3.csv")
    save_csv_compat(df_sens, SALIDA_V3_SENS, PROJECT_DIR / "eeg_c240_extraido_v3_sens.csv")

    df_v2 = df_c240[df_c240["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()
    df_v2_sens = df_sens[df_sens["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()
    save_csv_compat(df_v2, SALIDA_V2, PROJECT_DIR / "eeg_c240_extraido_v2.csv")
    save_csv_compat(df_v2_sens, SALIDA_V2_SENS, PROJECT_DIR / "eeg_c240_extraido_v2_sens.csv")

    if "S2 match" in condiciones:
        df_match = df_c240[df_c240["condicion"] == "S2 match"].copy()
        save_csv_compat(df_match, SALIDA_MATCH, OUTPUT_DIR / "eeg_c240_match_extraido_v2.csv", PROJECT_DIR / "eeg_c240_match_extraido_v3.csv")

    print("\nGenerando figuras...")
    plot_boxplot(df_c240, condiciones)
    plot_latencias(df_c240, condiciones)
    plot_polaridad(df_c240, condiciones)

    print("\n[OK] Script 05 v3 finalizado.")
