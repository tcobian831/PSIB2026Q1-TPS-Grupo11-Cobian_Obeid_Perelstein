"""
==============================================================================
Script 09 v3: Validación de robustez
==============================================================================

Hace:
1. Jackknife inter-sujeto.
2. Split-half de trials.
3. Barrido de ventana temporal.

Uso:
    python .\\scripts\\09_robustez_v3.py
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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


ENTRADA_PREPROC = first_existing([
    OUTPUT_DIR / "eeg_data_preprocesado_v3.parquet",
    OUTPUT_DIR / "eeg_data_preprocesado_v2.parquet",
    PROJECT_DIR / "eeg_data_preprocesado_v3.parquet",
    PROJECT_DIR / "eeg_data_preprocesado_v2.parquet",
], "preprocesado")

ENTRADA_PE = first_existing([
    OUTPUT_DIR / "eeg_PE_individual_v3.parquet",
    OUTPUT_DIR / "eeg_PE_individual_v2.parquet",
    PROJECT_DIR / "eeg_PE_individual_v3.parquet",
    PROJECT_DIR / "eeg_PE_individual_v2.parquet",
], "PE individual")

ENTRADA_C240 = first_existing([
    OUTPUT_DIR / "eeg_c240_extraido_v3.csv",
    OUTPUT_DIR / "eeg_c240_extraido_v2.csv",
    PROJECT_DIR / "eeg_c240_extraido_v3.csv",
    PROJECT_DIR / "eeg_c240_extraido_v2.csv",
], "CSV c240")

VENTANAS_BARRIDO = [(200,260), (210,270), (220,260), (230,270), (210,280), (200,280)]

def extraer_pico_abs_arr(arr, m_ini, m_fin):
    ventana = np.asarray(arr)[m_ini:m_fin+1]
    if len(ventana) == 0:
        return np.nan
    return float(ventana[np.argmax(np.abs(ventana))])

def jackknife(df_c240):
    filas = []
    for cond in CONDICIONES_PRINCIPALES:
        for canal in CANALES_INTERES:
            sub = df_c240[(df_c240["canal"] == canal) & (df_c240["condicion"] == cond)].dropna(subset=["amplitud_abs_uV"])
            sujetos = sub["sujeto"].unique()
            ps = []
            for s in sujetos:
                tmp = sub[sub["sujeto"] != s]
                ctrl = tmp[tmp["grupo"] == "control"]["amplitud_abs_uV"].values
                alc = tmp[tmp["grupo"] == "alcoholic"]["amplitud_abs_uV"].values
                _, p = safe_ttest_ind(ctrl, alc)
                if not np.isnan(p):
                    ps.append(p)
            parr = np.asarray(ps)
            filas.append({
                "canal": canal,
                "condicion": cond,
                "n_iter": len(parr),
                "p_min": float(np.min(parr)) if len(parr) else np.nan,
                "p_max": float(np.max(parr)) if len(parr) else np.nan,
                "p_mediana": float(np.median(parr)) if len(parr) else np.nan,
                "pct_sig": float(100 * np.mean(parr < ALPHA)) if len(parr) else np.nan,
            })
    return pd.DataFrame(filas)

def split_half(df_preproc):
    filas = []
    m_ini, m_fin = ms_a_muestra(220), ms_a_muestra(260)

    for cond in CONDICIONES_PRINCIPALES:
        for canal in CANALES_INTERES:
            sub = df_preproc[(df_preproc["condicion"] == cond) & (df_preproc["canal"] == canal)]
            registros = []
            for (sujeto, grupo), df_s in sub.groupby(["sujeto", "grupo"], observed=True):
                trials = np.sort(df_s["trial_num"].unique())
                if len(trials) < 10:
                    continue
                mitad_a = trials[::2]
                mitad_b = trials[1::2]
                def pico_para(trials_sel):
                    avg = df_s[df_s["trial_num"].isin(trials_sel)].groupby("muestra")["valor_uV"].mean().sort_index().values
                    if len(avg) != N_SAMPLES:
                        return np.nan
                    return abs(extraer_pico_abs_arr(avg, m_ini, m_fin))
                registros.append({
                    "sujeto": sujeto,
                    "grupo": grupo,
                    "abs_half_a": pico_para(mitad_a),
                    "abs_half_b": pico_para(mitad_b),
                })
            reg = pd.DataFrame(registros).dropna()
            if len(reg) >= 5 and reg["abs_half_a"].std() > 0 and reg["abs_half_b"].std() > 0:
                r, p = pearsonr(reg["abs_half_a"], reg["abs_half_b"])
                r_sb = (2*r)/(1+r) if abs(1+r) > 1e-9 else np.nan
            else:
                r, p, r_sb = np.nan, np.nan, np.nan
            filas.append({
                "canal": canal,
                "condicion": cond,
                "n_sujetos": len(reg),
                "r_pearson": float(r) if not np.isnan(r) else np.nan,
                "p_corr": float(p) if not np.isnan(p) else np.nan,
                "r_sb": float(r_sb) if not np.isnan(r_sb) else np.nan,
            })
    return pd.DataFrame(filas)

def barrido_ventana(df_pe):
    filas = []
    for t_ini, t_fin in VENTANAS_BARRIDO:
        m_ini, m_fin = ms_a_muestra(t_ini), ms_a_muestra(t_fin)
        for cond in CONDICIONES_PRINCIPALES:
            for canal in CANALES_INTERES:
                sub = df_pe[(df_pe["canal"] == canal) & (df_pe["condicion"] == cond)]
                ctrl, alc = [], []
                for (sujeto, grupo), df_s in sub.groupby(["sujeto", "grupo"], observed=True):
                    ventana = df_s[df_s["muestra"].between(m_ini, m_fin)]
                    if ventana.empty:
                        continue
                    idx = ventana["PE_uV"].abs().idxmax()
                    val = abs(float(ventana.loc[idx, "PE_uV"]))
                    if grupo == "control":
                        ctrl.append(val)
                    elif grupo == "alcoholic":
                        alc.append(val)
                t, p = safe_ttest_ind(ctrl, alc)
                filas.append({
                    "ventana_ms": f"{t_ini}-{t_fin}",
                    "canal": canal,
                    "condicion": cond,
                    "n_control": len(ctrl),
                    "n_alcoholic": len(alc),
                    "media_ctrl": float(np.mean(ctrl)) if len(ctrl) else np.nan,
                    "media_alc": float(np.mean(alc)) if len(alc) else np.nan,
                    "t_estadistico": t,
                    "p_valor": p,
                    "sig": "Si" if pd.notna(p) and p < ALPHA else "No",
                })
    return pd.DataFrame(filas)

def plot_jackknife(df_jk):
    fig, axes = plt.subplots(1, len(CONDICIONES_PRINCIPALES), figsize=(7*len(CONDICIONES_PRINCIPALES), 4.5), squeeze=False, sharey=True)
    axes = axes[0]; x = np.arange(len(CANALES_INTERES))
    fig.suptitle("Robustez Jackknife: % de iteraciones con p<0.05", fontsize=12)
    for ax, cond in zip(axes, CONDICIONES_PRINCIPALES):
        sub = df_jk[df_jk["condicion"] == cond]
        vals = [sub[sub["canal"] == c]["pct_sig"].values[0] for c in CANALES_INTERES]
        ax.bar(x, vals, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(CANALES_INTERES)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% significativo")
        ax.set_title(cond)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, (0 if np.isnan(v) else v) + 1, "NA" if np.isnan(v) else f"{v:.0f}%", ha="center", fontsize=9)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_robustez_jackknife_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

def plot_split_half(df_sh):
    fig, axes = plt.subplots(1, len(CONDICIONES_PRINCIPALES), figsize=(7*len(CONDICIONES_PRINCIPALES), 4.5), squeeze=False, sharey=True)
    axes = axes[0]; x = np.arange(len(CANALES_INTERES))
    fig.suptitle("Confiabilidad Split-Half: r Spearman-Brown", fontsize=12)
    for ax, cond in zip(axes, CONDICIONES_PRINCIPALES):
        sub = df_sh[df_sh["condicion"] == cond]
        vals = [sub[sub["canal"] == c]["r_sb"].values[0] for c in CANALES_INTERES]
        ax.bar(x, vals, alpha=0.75)
        ax.axhline(0.7, color="orange", linestyle="--", linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(CANALES_INTERES)
        ax.set_ylim(-0.2, 1.05)
        ax.set_ylabel("r_SB")
        ax.set_title(cond)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, (0 if np.isnan(v) else v) + 0.03, "NA" if np.isnan(v) else f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_robustez_splithalf_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

def plot_barrido(df_vent):
    df = df_vent.copy()
    df["canal_cond"] = df["canal"] + "\n" + df["condicion"]
    pivot = df.pivot_table(index="ventana_ms", columns="canal_cond", values="p_valor", aggfunc="first")
    pivot = pivot.reindex([f"{a}-{b}" for a,b in VENTANAS_BARRIDO if f"{a}-{b}" in pivot.index])
    logp = -np.log10(pivot.values.astype(float) + 1e-12)

    fig, ax = plt.subplots(figsize=(max(9, 1.1*pivot.shape[1]), max(4, 0.7*pivot.shape[0] + 2)))
    im = ax.imshow(logp, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1])); ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0])); ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Barrido de ventana: -log10(p) del t-test |c240|", fontsize=11)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            p = pivot.values[i, j]
            if pd.isna(p):
                txt = "NA"
            else:
                txt = f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("-log10(p)")
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_robustez_ventana_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

if __name__ == "__main__":
    print("=" * 70)
    print("Script 09 v3: Robustez")
    print("=" * 70)

    print("\nCargando datos...")
    df_c240 = pd.read_csv(ENTRADA_C240)
    df_c240 = df_c240[df_c240["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()
    df_pre = pd.read_parquet(ENTRADA_PREPROC)
    df_pre = df_pre[df_pre["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()
    df_pe = pd.read_parquet(ENTRADA_PE)
    df_pe = df_pe[df_pe["condicion"].isin(CONDICIONES_PRINCIPALES)].copy()

    print("\n1) Jackknife...")
    df_jk = jackknife(df_c240)
    print(df_jk.to_string(index=False))
    save_csv_compat(df_jk, OUTPUT_DIR / "tabla_robustez_jackknife_v3.csv", OUTPUT_DIR / "tabla_robustez_jackknife_v2.csv")
    plot_jackknife(df_jk)

    print("\n2) Split-half...")
    df_sh = split_half(df_pre)
    print(df_sh.to_string(index=False))
    save_csv_compat(df_sh, OUTPUT_DIR / "tabla_robustez_splithalf_v3.csv", OUTPUT_DIR / "tabla_robustez_splithalf_v2.csv")
    plot_split_half(df_sh)

    print("\n3) Barrido de ventana...")
    df_vent = barrido_ventana(df_pe)
    print(df_vent.to_string(index=False))
    save_csv_compat(df_vent, OUTPUT_DIR / "tabla_robustez_ventana_v3.csv", OUTPUT_DIR / "tabla_robustez_ventana_v2.csv")
    plot_barrido(df_vent)

    print("\n[OK] Script 09 v3 finalizado.")
