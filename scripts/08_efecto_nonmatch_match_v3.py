"""
==============================================================================
Script 08 v3: Efecto Nonmatch-Match
==============================================================================

Calcula:
    delta = |c240, S2 nomatch| - |c240, S2 match|

Uso:
    python .\\scripts\\08_efecto_nonmatch_match_v3.py
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
    PROJECT_DIR / "eeg_c240_extraido_v3.csv",
    OUTPUT_DIR / "eeg_c240_extraido_v2.csv",
], "CSV c240 con S2 nomatch y S2 match")

SALIDA_DELTA = OUTPUT_DIR / "eeg_delta_nonmatch_match_v3.csv"
SALIDA_TABLA = OUTPUT_DIR / "tabla_efecto_nonmatch_match_v3.csv"

def calcular_delta(df):
    nomatch = df[df["condicion"] == "S2 nomatch"][["sujeto", "grupo", "canal", "amplitud_abs_uV"]].rename(columns={"amplitud_abs_uV": "abs_nomatch"})
    match = df[df["condicion"] == "S2 match"][["sujeto", "grupo", "canal", "amplitud_abs_uV"]].rename(columns={"amplitud_abs_uV": "abs_match"})
    if nomatch.empty:
        raise ValueError("No hay condición S2 nomatch en el CSV c240.")
    if match.empty:
        raise ValueError("No hay condición S2 match en el CSV c240. Corré 03 v3 -> 04 v3 -> 05 v3.")
    merged = pd.merge(nomatch, match, on=["sujeto", "grupo", "canal"], how="inner")
    merged["delta"] = merged["abs_nomatch"] - merged["abs_match"]
    return merged

def contraste(df_delta):
    filas = []
    for canal in CANALES_INTERES:
        ctrl = df_delta[(df_delta["canal"] == canal) & (df_delta["grupo"] == "control")]["delta"].dropna().values
        alc = df_delta[(df_delta["canal"] == canal) & (df_delta["grupo"] == "alcoholic")]["delta"].dropna().values
        t, p = safe_ttest_ind(ctrl, alc)
        t_ctrl, p_ctrl = safe_ttest_1samp_greater(ctrl, 0.0)
        t_alc, p_alc = safe_ttest_1samp_greater(alc, 0.0)
        filas.append({
            "canal": canal,
            "n_control": len(ctrl),
            "n_alcoholic": len(alc),
            "delta_media_ctrl": float(np.mean(ctrl)) if len(ctrl) else np.nan,
            "delta_sd_ctrl": float(np.std(ctrl, ddof=1)) if len(ctrl) > 1 else np.nan,
            "delta_media_alc": float(np.mean(alc)) if len(alc) else np.nan,
            "delta_sd_alc": float(np.std(alc, ddof=1)) if len(alc) > 1 else np.nan,
            "t_entre_grupos": t,
            "p_entre_grupos": p,
            "cohen_d": cohen_d(ctrl, alc),
            "p_ctrl_delta_mayor_0": p_ctrl,
            "p_alc_delta_mayor_0": p_alc,
            "direccion_ok": "Si" if len(ctrl) and len(alc) and np.mean(ctrl) > np.mean(alc) else "No",
        })
    res = pd.DataFrame(filas)
    res["p_bonferroni"] = np.clip(res["p_entre_grupos"].values * len(res), 0, 1)
    res["p_fdr_bh"] = correccion_bh(res["p_entre_grupos"].values)
    return res

def imprimir_tabla(res):
    print("\n" + "=" * 100)
    print("EFECTO NONMATCH-MATCH: delta = |S2 nomatch| - |S2 match|")
    print("=" * 100)
    print(f"  {'Canal':<5} {'delta Ctrl':>18} {'delta Alc':>18} {'t':>8} {'p':>9} {'q(FDR)':>9} {'d':>7} {'dir':>5}")
    for _, f in res.iterrows():
        ctrl = f"{f['delta_media_ctrl']:+.3f}±{f['delta_sd_ctrl']:.3f}"
        alc = f"{f['delta_media_alc']:+.3f}±{f['delta_sd_alc']:.3f}"
        mark = "*" if pd.notna(f["p_fdr_bh"]) and f["p_fdr_bh"] < ALPHA else ""
        print(f"  {f['canal']:<5} {ctrl:>18} {alc:>18} {f['t_entre_grupos']:>8.3f} {f['p_entre_grupos']:>9.4f} {f['p_fdr_bh']:>9.4f} {f['cohen_d']:>7.2f} {f['direccion_ok']:>5} {mark}")
    print("\nOne-sided dentro de grupo: H1 delta > 0")
    print(f"  {'Canal':<5} {'p ctrl':>10} {'p alc':>10}")
    for _, f in res.iterrows():
        print(f"  {f['canal']:<5} {f['p_ctrl_delta_mayor_0']:>10.4f} {f['p_alc_delta_mayor_0']:>10.4f}")
    print("=" * 100)

def plot_efecto(df_delta, res):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, len(CANALES_INTERES), figsize=(5*len(CANALES_INTERES), 5), squeeze=False, sharey=True)
    axes = axes[0]
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    fig.suptitle("Efecto Nonmatch-Match: delta = |S2 nomatch| - |S2 match|", fontsize=13)

    for ax, canal in zip(axes, CANALES_INTERES):
        data, labels, cols = [], [], []
        for grupo in ["control", "alcoholic"]:
            vals = df_delta[(df_delta["canal"] == canal) & (df_delta["grupo"] == grupo)]["delta"].dropna().values
            data.append(vals)
            labels.append(f"{grupo}\n(n={len(vals)})")
            cols.append(colores[grupo])
        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col); patch.set_alpha(0.55)
        for k, vals in enumerate(data):
            ax.scatter(rng.normal(k+1, 0.06, len(vals)), vals, s=14, alpha=0.35, color=cols[k])
        rf = res[res["canal"] == canal].iloc[0]
        q = rf["p_fdr_bh"]; d = rf["cohen_d"]
        ax.set_title(f"{canal}\nq={q:.4f} | d={d:.2f}", fontsize=10,
                     color="red" if pd.notna(q) and q < ALPHA else "gray")
        ax.set_xticks([1,2]); ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel("delta |c240| (µV)")
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_efecto_nonmatch_match_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

if __name__ == "__main__":
    print("=" * 70)
    print("Script 08 v3: Efecto Nonmatch-Match")
    print("=" * 70)

    print(f"\nCargando: {ENTRADA}")
    df = pd.read_csv(ENTRADA)
    df_delta = calcular_delta(df)
    print(f"  Filas delta: {len(df_delta)}")

    res = contraste(df_delta)
    imprimir_tabla(res)

    save_csv_compat(df_delta, SALIDA_DELTA, PROJECT_DIR / "eeg_delta_nonmatch_match_v3.csv")
    save_csv_compat(res, SALIDA_TABLA, OUTPUT_DIR / "tabla_efecto_nonmatch_match_v2.csv", PROJECT_DIR / "tabla_efecto_nonmatch_match_v3.csv")
    plot_efecto(df_delta, res)

    print("\n[OK] Script 08 v3 finalizado.")
