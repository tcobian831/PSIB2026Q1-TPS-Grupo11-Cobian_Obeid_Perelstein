"""
==============================================================================
Script 10 v3: Especificidad regional con canales control
==============================================================================

Corrige el problema de mayúsculas/minúsculas: en UCI suelen venir CZ/FZ,
no Cz/Fz. El script resuelve canales de forma case-insensitive.

Uso:
    python .\\scripts\\10_especificidad_regional_v3.py
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


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


F_LOW, F_HIGH = 0.1, 30.0
ORDEN = 4
T_BASELINE_MS = 30
N_BASELINE = int(T_BASELINE_MS / 1000 * FS)
UMBRAL_UV = 100.0

CANALES_CONTROL_DESEADOS = ["Cz", "Fz"]
CONDICIONES = CONDICIONES_PRINCIPALES
M_INI = ms_a_muestra(T_C240_INI_MS)
M_FIN = ms_a_muestra(T_C240_FIN_MS)

ENTRADA_CRUDO = first_existing([
    OUTPUT_DIR / "eeg_data_cargado.parquet",
    PROJECT_DIR / "eeg_data_cargado.parquet",
], "eeg_data_cargado.parquet")

ENTRADA_INTERES = first_existing([
    OUTPUT_DIR / "tabla_estadistica_v3.csv",
    OUTPUT_DIR / "tabla_estadistica_v2.csv",
    PROJECT_DIR / "tabla_estadistica_v3.csv",
    PROJECT_DIR / "tabla_estadistica_v2.csv",
], "tabla estadística de canales de interés")

def resolver_canales_case_insensitive(disponibles, deseados):
    mapa = {c.lower(): c for c in disponibles}
    resueltos = []
    faltantes = []
    for c in deseados:
        real = mapa.get(c.lower())
        if real is None:
            faltantes.append(c)
        else:
            resueltos.append(real)
    return resueltos, faltantes

def cohen_stats_control(df_c240_ctrl, canales_control):
    filas = []
    for cond in CONDICIONES:
        for canal in canales_control:
            sel = (df_c240_ctrl["canal"] == canal) & (df_c240_ctrl["condicion"] == cond)
            ctrl = df_c240_ctrl[sel & (df_c240_ctrl["grupo"] == "control")]["amplitud_abs_uV"].dropna().values
            alc = df_c240_ctrl[sel & (df_c240_ctrl["grupo"] == "alcoholic")]["amplitud_abs_uV"].dropna().values
            t, p = safe_ttest_ind(ctrl, alc)
            filas.append({
                "canal": canal,
                "condicion": cond,
                "tipo": "control",
                "n_control": len(ctrl),
                "n_alcoholic": len(alc),
                "media_ctrl": float(np.mean(ctrl)) if len(ctrl) else np.nan,
                "media_alc": float(np.mean(alc)) if len(alc) else np.nan,
                "t_estadistico": t,
                "p_valor": p,
                "cohen_d": cohen_d(ctrl, alc),
            })
    return pd.DataFrame(filas)

def preprocesar_canales_control(df_crudo, canales_control):
    df = df_crudo[
        df_crudo["condicion"].isin(CONDICIONES)
        & df_crudo["canal"].isin(canales_control)
    ].copy()
    if df.empty:
        raise ValueError(f"No hay datos para canales control resueltos: {canales_control}")

    nyq = FS / 2
    b, a = butter(ORDEN, [F_LOW/nyq, F_HIGH/nyq], btype="band")

    resultados = []
    grupos = df.groupby(["sujeto", "grupo", "condicion", "trial_num", "canal"], observed=True)
    print(f"  Preprocesando {len(grupos):,} combinaciones sujeto-condición-trial-canal...")

    n_ok = 0
    n_rech = 0
    for idx, (_, g) in enumerate(grupos):
        if idx % 5000 == 0:
            print(f"    {idx:,}/{len(grupos):,}")
        g = g.sort_values("muestra").copy()
        if len(g) != N_SAMPLES:
            n_rech += 1
            continue
        senal = g["valor_uV"].to_numpy(dtype=float)
        senal = filtfilt(b, a, senal)
        senal = senal - np.mean(senal[:N_BASELINE])
        if np.any(np.abs(senal) > UMBRAL_UV):
            n_rech += 1
            continue
        g["valor_uV"] = senal
        resultados.append(g)
        n_ok += 1

    print(f"  OK: {n_ok:,} | Rechazados: {n_rech:,}")
    if not resultados:
        raise RuntimeError("No quedó ningún trial válido en canales control.")
    return pd.concat(resultados, ignore_index=True)

def calcular_c240_control(df_pre):
    pe = (
        df_pre.groupby(["sujeto", "grupo", "canal", "condicion", "muestra"], observed=True)["valor_uV"]
        .agg(PE_uV="mean", n_trials="count")
        .reset_index()
    )
    filas = []
    for (sujeto, grupo, canal, cond), df_s in pe.groupby(["sujeto", "grupo", "canal", "condicion"], observed=True):
        ventana = df_s[df_s["muestra"].between(M_INI, M_FIN)]
        if ventana.empty:
            continue
        idx = ventana["PE_uV"].abs().idxmax()
        amp = float(ventana.loc[idx, "PE_uV"])
        filas.append({
            "sujeto": sujeto,
            "grupo": grupo,
            "canal": canal,
            "condicion": cond,
            "amplitud_uV": amp,
            "amplitud_abs_uV": abs(amp),
            "latencia_ms": int(ventana.loc[idx, "muestra"]) / FS * 1000,
        })
    return pd.DataFrame(filas)

def preparar_interes(df_int):
    # tabla_estadistica_v3/v2 tiene media_abs_control/media_abs_alcoholic
    cols = ["canal", "condicion", "media_abs_control", "media_abs_alcoholic", "t_estadistico", "p_valor", "cohen_d"]
    faltantes = [c for c in cols if c not in df_int.columns]
    if faltantes:
        raise KeyError(f"La tabla de interés no tiene columnas esperadas: {faltantes}")
    df = df_int[cols].copy()
    df = df.rename(columns={"media_abs_control": "media_ctrl", "media_abs_alcoholic": "media_alc"})
    df["tipo"] = "interes"
    return df

def plot_especificidad(df_total):
    fig, axes = plt.subplots(1, len(CONDICIONES), figsize=(7*len(CONDICIONES), 5), squeeze=False, sharey=True)
    axes = axes[0]
    fig.suptitle("Especificidad regional: |Cohen's d| en canales de interés vs control", fontsize=12)

    for ax, cond in zip(axes, CONDICIONES):
        sub = df_total[df_total["condicion"] == cond].copy()
        sub["abs_d"] = sub["cohen_d"].abs()
        sub["orden"] = sub["tipo"].map({"interes": 0, "control": 1})
        sub = sub.sort_values(["orden", "canal"]).reset_index(drop=True)
        colors = ["#dc2626" if t == "interes" else "#94a3b8" for t in sub["tipo"]]
        x = np.arange(len(sub))
        ax.bar(x, sub["abs_d"], color=colors, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(sub["canal"])
        ax.set_ylabel("|Cohen's d|")
        ax.set_title(cond)
        ax.axhline(0.2, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0.5, color="orange", linestyle=":", linewidth=0.8)
        ax.axhline(0.8, color="red", linestyle=":", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)
        for i, d in enumerate(sub["abs_d"]):
            ax.text(i, d + 0.02, f"{d:.2f}", ha="center", fontsize=9)
        ax.text(0.02, 0.98, "Rojo = interés\nGris = control", transform=ax.transAxes,
                fontsize=8, va="top", bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_especificidad_regional_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")

if __name__ == "__main__":
    print("=" * 70)
    print("Script 10 v3: Especificidad regional")
    print("=" * 70)

    print(f"\nCargando crudo: {ENTRADA_CRUDO}")
    df_crudo = pd.read_parquet(ENTRADA_CRUDO)
    canales_disponibles = sorted(df_crudo["canal"].unique())
    canales_control, faltantes = resolver_canales_case_insensitive(canales_disponibles, CANALES_CONTROL_DESEADOS)

    print(f"  Canales control pedidos: {CANALES_CONTROL_DESEADOS}")
    print(f"  Canales control encontrados: {canales_control}")
    if faltantes:
        print(f"  [Aviso] No encontrados: {faltantes}")
        print(f"  Canales con z disponibles: {[c for c in canales_disponibles if 'z' in c.lower()]}")

    if not canales_control:
        raise ValueError("No se encontró ningún canal control. Editá CANALES_CONTROL_DESEADOS.")

    df_pre = preprocesar_canales_control(df_crudo, canales_control)
    df_c240_ctrl = calcular_c240_control(df_pre)
    save_csv_compat(df_c240_ctrl, OUTPUT_DIR / "eeg_c240_control_chans_v3.csv", OUTPUT_DIR / "eeg_c240_control_chans_v2.csv")

    df_ctrl = cohen_stats_control(df_c240_ctrl, canales_control)
    print("\nResultados canales control:")
    print(df_ctrl.to_string(index=False))

    print(f"\nCargando tabla de interés: {ENTRADA_INTERES}")
    df_int = preparar_interes(pd.read_csv(ENTRADA_INTERES))

    df_total = pd.concat([df_int, df_ctrl], ignore_index=True)
    save_csv_compat(df_total, OUTPUT_DIR / "tabla_especificidad_regional_v3.csv", OUTPUT_DIR / "tabla_especificidad_regional_v2.csv")
    plot_especificidad(df_total)

    print("\n[OK] Script 10 v3 finalizado.")
