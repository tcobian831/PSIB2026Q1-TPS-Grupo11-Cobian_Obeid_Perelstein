"""
==============================================================================
Script 07 v3: Análisis de latencia c240/VMP
==============================================================================

Uso:
    python .\\scripts\\07_latencia_v3.py
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


ENTRADA = first_existing([
    OUTPUT_DIR / "eeg_c240_extraido_v3.csv",
    OUTPUT_DIR / "eeg_c240_extraido_v2.csv",
    PROJECT_DIR / "eeg_c240_extraido_v3.csv",
    PROJECT_DIR / "eeg_c240_extraido_v2.csv",
], "CSV c240")

SALIDA = OUTPUT_DIR / "tabla_latencia_v3.csv"
LATENCIA_INI_BORDE = 222
LATENCIA_FIN_BORDE = 257

def calcular_estadisticas_latencia(df):
    filas = []
    for condicion in CONDICIONES_PRINCIPALES:
        for canal in CANALES_INTERES:
            sel = (df["canal"] == canal) & (df["condicion"] == condicion)
            ctrl = df[sel & (df["grupo"] == "control")]["latencia_ms"].dropna().values
            alc = df[sel & (df["grupo"] == "alcoholic")]["latencia_ms"].dropna().values
            t_stat, p_valor = safe_ttest_ind(ctrl, alc)
            filas.append({
                "canal": canal,
                "condicion": condicion,
                "n_control": len(ctrl),
                "n_alcoholic": len(alc),
                "media_lat_control": float(np.mean(ctrl)) if len(ctrl) else np.nan,
                "sd_lat_control": float(np.std(ctrl, ddof=1)) if len(ctrl) > 1 else np.nan,
                "media_lat_alcoholic": float(np.mean(alc)) if len(alc) else np.nan,
                "sd_lat_alcoholic": float(np.std(alc, ddof=1)) if len(alc) > 1 else np.nan,
                "t_estadistico": t_stat,
                "p_valor": p_valor,
                "cohen_d": cohen_d(ctrl, alc),
            })
    res = pd.DataFrame(filas)
    res["p_bonferroni"] = np.clip(res["p_valor"].values * len(res), 0, 1)
    res["p_fdr_bh"] = correccion_bh(res["p_valor"].values)
    res["sig_fdr"] = (res["p_fdr_bh"] < ALPHA).map({True: "Si", False: "No"})
    return res

def diagnostico_bordes(df):
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO DE LATENCIAS EN BORDES")
    print("=" * 80)
    for cond in CONDICIONES_PRINCIPALES:
        print(f"\nCondición: {cond}")
        for canal in CANALES_INTERES:
            for grupo in ["control", "alcoholic"]:
                vals = df[(df["canal"] == canal) & (df["condicion"] == cond) & (df["grupo"] == grupo)]["latencia_ms"].dropna()
                if len(vals) == 0:
                    continue
                n_inf = (vals < LATENCIA_INI_BORDE).sum()
                n_sup = (vals > LATENCIA_FIN_BORDE).sum()
                pct = 100 * (n_inf + n_sup) / len(vals)
                marca = " <-- revisar" if pct > 30 else ""
                print(f"  {canal:<5} {grupo:<10} n={len(vals):3d} borde_inf={n_inf:3d} borde_sup={n_sup:3d} ({pct:5.1f}%){marca}")
    print("=" * 80)

def imprimir_tabla(res):
    print("\n" + "=" * 90)
    print("LATENCIA c240/VMP")
    print("=" * 90)
    for condicion in CONDICIONES_PRINCIPALES:
        print(f"\nCondición: {condicion}")
        print(f"  {'Canal':<5} {'Ctrl ms':>16} {'Alc ms':>16} {'t':>8} {'p':>9} {'q(FDR)':>9} {'d':>7}")
        for _, f in res[res["condicion"] == condicion].iterrows():
            ctrl = f"{f['media_lat_control']:.1f}±{f['sd_lat_control']:.1f}"
            alc = f"{f['media_lat_alcoholic']:.1f}±{f['sd_lat_alcoholic']:.1f}"
            mark = "*" if pd.notna(f["p_fdr_bh"]) and f["p_fdr_bh"] < ALPHA else ""
            print(f"  {f['canal']:<5} {ctrl:>16} {alc:>16} {f['t_estadistico']:>8.3f} {f['p_valor']:>9.4f} {f['p_fdr_bh']:>9.4f} {f['cohen_d']:>7.2f} {mark}")
    print("=" * 90)

def plot_boxplot(df, res):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(CONDICIONES_PRINCIPALES), len(CANALES_INTERES),
                             figsize=(5*len(CANALES_INTERES), 4*len(CONDICIONES_PRINCIPALES)),
                             squeeze=False, sharey=True)
    fig.suptitle("Latencia del pico c240/VMP por grupo", fontsize=13)
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for i, condicion in enumerate(CONDICIONES_PRINCIPALES):
        for j, canal in enumerate(CANALES_INTERES):
            ax = axes[i, j]
            data, labels, cols = [], [], []
            for grupo in ["control", "alcoholic"]:
                vals = df[(df["canal"] == canal) & (df["condicion"] == condicion) & (df["grupo"] == grupo)]["latencia_ms"].dropna().values
                data.append(vals)
                labels.append(f"{grupo}\n(n={len(vals)})")
                cols.append(colores[grupo])
            bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
            for patch, col in zip(bp["boxes"], cols):
                patch.set_facecolor(col); patch.set_alpha(0.55)
            for k, vals in enumerate(data):
                ax.scatter(rng.normal(k+1, 0.06, len(vals)), vals, s=14, alpha=0.35, color=cols[k])
            rf = res[(res["canal"] == canal) & (res["condicion"] == condicion)].iloc[0]
            q = rf["p_fdr_bh"]; d = rf["cohen_d"]
            ax.set_title(f"{canal} — {condicion}\nq={q:.4f} | d={d:.2f}", fontsize=9,
                         color="red" if pd.notna(q) and q < ALPHA else "gray")
            ax.set_xticks([1,2]); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("Latencia (ms)")
            ax.axhline(240, color="orange", linestyle="--", linewidth=0.8)
            ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_latencia_boxplot_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

def plot_barras(res):
    fig, axes = plt.subplots(1, len(CONDICIONES_PRINCIPALES), figsize=(7*len(CONDICIONES_PRINCIPALES), 5), squeeze=False, sharey=True)
    axes = axes[0]
    x = np.arange(len(CANALES_INTERES)); ancho = 0.35
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for ax, condicion in zip(axes, CONDICIONES_PRINCIPALES):
        sub = res[res["condicion"] == condicion]
        for i, grupo in enumerate(["control", "alcoholic"]):
            medias = [sub[sub["canal"] == c][f"media_lat_{grupo}"].values[0] for c in CANALES_INTERES]
            sds = [sub[sub["canal"] == c][f"sd_lat_{grupo}"].values[0] for c in CANALES_INTERES]
            ax.bar(x + i*ancho, medias, ancho, yerr=sds, capsize=4, label=grupo, color=colores[grupo], alpha=0.75)
        ax.set_xticks(x + ancho/2); ax.set_xticklabels(CANALES_INTERES)
        ax.set_ylabel("Latencia media (ms)")
        ax.set_ylim(215, 275)
        ax.axhline(240, color="orange", linestyle="--", linewidth=0.8)
        ax.set_title(condicion)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_latencia_barras_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

if __name__ == "__main__":
    print("=" * 70)
    print("Script 07 v3: Latencia c240/VMP")
    print("=" * 70)

    print(f"\nCargando: {ENTRADA}")
    df = pd.read_csv(ENTRADA)
    df = df[df["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()

    diagnostico_bordes(df)
    res = calcular_estadisticas_latencia(df)
    imprimir_tabla(res)
    save_csv_compat(res, SALIDA, OUTPUT_DIR / "tabla_latencia_v2.csv", PROJECT_DIR / "tabla_latencia_v3.csv", PROJECT_DIR / "tabla_latencia_v2.csv")

    plot_boxplot(df, res)
    plot_barras(res)

    print("\n[OK] Script 07 v3 finalizado.")
