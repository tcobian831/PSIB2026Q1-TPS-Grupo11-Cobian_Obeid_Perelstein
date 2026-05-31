"""
==============================================================================
Script 06 v3: Estadística entre grupos para |c240|
==============================================================================

Compara control vs alcoholic usando |amplitud c240| en S1 obj y S2 nomatch.
Guarda tabla y figuras v3, más copias v2 de compatibilidad.

Uso:
    python .\\scripts\\06_estadistica_v3.py
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


from scipy.stats import ttest_ind

ENTRADA = first_existing([
    OUTPUT_DIR / "eeg_c240_extraido_v3.csv",
    OUTPUT_DIR / "eeg_c240_extraido_v2.csv",
    PROJECT_DIR / "eeg_c240_extraido_v3.csv",
    PROJECT_DIR / "eeg_c240_extraido_v2.csv",
], "CSV c240")

ENTRADA_SENS = None
for p in [OUTPUT_DIR / "eeg_c240_extraido_v3_sens.csv", OUTPUT_DIR / "eeg_c240_extraido_v2_sens.csv", PROJECT_DIR / "eeg_c240_extraido_v3_sens.csv"]:
    if p.exists():
        ENTRADA_SENS = p
        break

SALIDA = OUTPUT_DIR / "tabla_estadistica_v3.csv"
SALIDA_SENS = OUTPUT_DIR / "tabla_estadistica_v3_sens.csv"

def calcular_estadisticas(df):
    filas = []
    for condicion in CONDICIONES_PRINCIPALES:
        for canal in CANALES_INTERES:
            sel = (df["canal"] == canal) & (df["condicion"] == condicion)
            ctrl_abs = df[sel & (df["grupo"] == "control")]["amplitud_abs_uV"].dropna().values
            alc_abs = df[sel & (df["grupo"] == "alcoholic")]["amplitud_abs_uV"].dropna().values
            ctrl_sig = df[sel & (df["grupo"] == "control")]["amplitud_uV"].dropna().values
            alc_sig = df[sel & (df["grupo"] == "alcoholic")]["amplitud_uV"].dropna().values

            t_stat, p_valor = safe_ttest_ind(ctrl_abs, alc_abs)
            filas.append({
                "canal": canal,
                "condicion": condicion,
                "n_control": len(ctrl_abs),
                "n_alcoholic": len(alc_abs),
                "media_abs_control": float(np.mean(ctrl_abs)) if len(ctrl_abs) else np.nan,
                "sd_abs_control": float(np.std(ctrl_abs, ddof=1)) if len(ctrl_abs) > 1 else np.nan,
                "media_abs_alcoholic": float(np.mean(alc_abs)) if len(alc_abs) else np.nan,
                "sd_abs_alcoholic": float(np.std(alc_abs, ddof=1)) if len(alc_abs) > 1 else np.nan,
                "media_signed_control": float(np.mean(ctrl_sig)) if len(ctrl_sig) else np.nan,
                "media_signed_alcoholic": float(np.mean(alc_sig)) if len(alc_sig) else np.nan,
                "t_estadistico": t_stat,
                "p_valor": p_valor,
                "cohen_d": cohen_d(ctrl_abs, alc_abs),
                "direccion_ok": "Si" if len(ctrl_abs) and len(alc_abs) and np.mean(alc_abs) < np.mean(ctrl_abs) else "No",
            })
    res = pd.DataFrame(filas)
    res["p_bonferroni"] = np.clip(res["p_valor"].values * len(res), 0, 1)
    res["p_fdr_bh"] = correccion_bh(res["p_valor"].values)
    res["sig_alpha"] = (res["p_valor"] < ALPHA).map({True: "Si", False: "No"})
    res["sig_bonferroni"] = (res["p_bonferroni"] < ALPHA).map({True: "Si", False: "No"})
    res["sig_fdr"] = (res["p_fdr_bh"] < ALPHA).map({True: "Si", False: "No"})
    return res

def imprimir_tabla(res, etiqueta):
    print("\n" + "=" * 95)
    print(f"ESTADÍSTICA |c240| — {etiqueta}")
    print("=" * 95)
    for condicion in CONDICIONES_PRINCIPALES:
        print(f"\nCondición: {condicion}")
        print(f"  {'Canal':<5} {'|Ctrl|':>16} {'|Alc|':>16} {'t':>8} {'p':>9} {'q(FDR)':>9} {'d':>7} {'dir':>5}")
        for _, f in res[res["condicion"] == condicion].iterrows():
            ctrl = f"{f['media_abs_control']:.3f}±{f['sd_abs_control']:.3f}"
            alc = f"{f['media_abs_alcoholic']:.3f}±{f['sd_abs_alcoholic']:.3f}"
            mark = "*" if pd.notna(f["p_fdr_bh"]) and f["p_fdr_bh"] < ALPHA else ""
            print(f"  {f['canal']:<5} {ctrl:>16} {alc:>16} {f['t_estadistico']:>8.3f} {f['p_valor']:>9.4f} {f['p_fdr_bh']:>9.4f} {f['cohen_d']:>7.2f} {f['direccion_ok']:>5} {mark}")
    print("=" * 95)

def plot_comparacion(df, res):
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(len(CONDICIONES_PRINCIPALES), len(CANALES_INTERES),
                             figsize=(5*len(CANALES_INTERES), 4.2*len(CONDICIONES_PRINCIPALES)),
                             squeeze=False)
    fig.suptitle("Comparación de |amplitud c240/VMP| entre grupos", fontsize=13)
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for i, condicion in enumerate(CONDICIONES_PRINCIPALES):
        for j, canal in enumerate(CANALES_INTERES):
            ax = axes[i, j]
            data, labels, cols = [], [], []
            for grupo in ["control", "alcoholic"]:
                vals = df[(df["canal"] == canal) & (df["condicion"] == condicion) & (df["grupo"] == grupo)]["amplitud_abs_uV"].dropna().values
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
            ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("|c240| (µV)")
            ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_estadistica_comparacion_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

def plot_barras(res):
    fig, axes = plt.subplots(1, len(CONDICIONES_PRINCIPALES), figsize=(7*len(CONDICIONES_PRINCIPALES), 5), squeeze=False, sharey=True)
    axes = axes[0]
    x = np.arange(len(CANALES_INTERES)); ancho = 0.35
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    fig.suptitle("Magnitud media de c240/VMP ± SD", fontsize=13)

    for ax, condicion in zip(axes, CONDICIONES_PRINCIPALES):
        sub = res[res["condicion"] == condicion]
        for i, grupo in enumerate(["control", "alcoholic"]):
            medias = [sub[sub["canal"] == c][f"media_abs_{grupo}"].values[0] for c in CANALES_INTERES]
            sds = [sub[sub["canal"] == c][f"sd_abs_{grupo}"].values[0] for c in CANALES_INTERES]
            ax.bar(x + i*ancho, medias, ancho, yerr=sds, capsize=4, label=grupo, color=colores[grupo], alpha=0.75)
        for j, canal in enumerate(CANALES_INTERES):
            q = sub[sub["canal"] == canal]["p_fdr_bh"].values[0]
            if pd.notna(q) and q < ALPHA:
                y = max(sub[sub["canal"] == canal]["media_abs_control"].values[0], sub[sub["canal"] == canal]["media_abs_alcoholic"].values[0])
                ax.text(j + ancho/2, y + 0.5, "*", ha="center", fontsize=14)
        ax.set_title(condicion)
        ax.set_xticks(x + ancho/2); ax.set_xticklabels(CANALES_INTERES)
        ax.set_ylabel("|c240| media (µV)")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_estadistica_barras_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

if __name__ == "__main__":
    print("=" * 70)
    print("Script 06 v3: Estadística |c240|")
    print("=" * 70)

    print(f"\nCargando: {ENTRADA}")
    df = pd.read_csv(ENTRADA)
    df = df[df["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()
    print(f"  Sujetos: {df['sujeto'].nunique()}")

    res = calcular_estadisticas(df)
    imprimir_tabla(res, "ventana principal 220-260 ms")

    save_csv_compat(res, SALIDA, OUTPUT_DIR / "tabla_estadistica_v2.csv", PROJECT_DIR / "tabla_estadistica_v3.csv", PROJECT_DIR / "tabla_estadistica_v2.csv")
    plot_comparacion(df, res)
    plot_barras(res)

    if ENTRADA_SENS is not None:
        print(f"\nCargando sensibilidad: {ENTRADA_SENS}")
        df_sens = pd.read_csv(ENTRADA_SENS)
        df_sens = df_sens[df_sens["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()
        res_sens = calcular_estadisticas(df_sens)
        imprimir_tabla(res_sens, "sensibilidad 200-280 ms")
        save_csv_compat(res_sens, SALIDA_SENS, OUTPUT_DIR / "tabla_estadistica_v2_sens.csv")

    print("\n[OK] Script 06 v3 finalizado.")
