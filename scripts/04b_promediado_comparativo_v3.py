"""
==============================================================================
Script 04b v3: Promediado homogéneo vs inhomogéneo (ponderado)
==============================================================================

Propósito
---------
Comparar dos estrategias de promediado de trials para obtener el PE por
sujeto/canal/condición:

  (A) Homogéneo: promedio aritmético clásico.
          x_bar(t) = (1/N) * sum_i x_i(t)
      Hipótesis implícita: todos los trials tienen amplitud de señal
      similar y varianza de ruido similar.

  (B) Inhomogéneo (weighted averaging tipo Davila & Mobin 1992):
          w_i = <x_i, s_hat> / ||s_hat||^2
          x_bar(t) = sum_i w_i * x_i(t) / sum_i w_i
      Donde s_hat es una estimación inicial de la señal evocada,
      tomada como el promedio homogéneo del sujeto. Esto modela
      amplitud variable trial-a-trial con varianza de ruido aprox.
      constante: a los trials con poca señal se les baja el peso
      sin "desinflar" el ruido.

Después compara ambas estrategias con tres métricas:
  1. SNR del ERP por sujeto: var(ventana de señal) / var(baseline temprano).
  2. Confiabilidad split-half (pares-impares) con corrección Spearman-Brown.
  3. Tamaño del efecto entre grupos (Cohen's d) en |c240| en cada canal y
     condición. Mayor |d| sugiere que el método de promediado conserva
     mejor la diferencia control vs alcoholic.

Entrada
-------
  outputs/eeg_data_preprocesado_v3.parquet  (con S1 obj, S2 nomatch, S2 match)

Salidas
-------
  outputs/eeg_PE_individual_homogeneo_v3.parquet
  outputs/eeg_PE_individual_ponderado_v3.parquet
  outputs/eeg_PE_grandaverage_homogeneo_v3.parquet
  outputs/eeg_PE_grandaverage_ponderado_v3.parquet
  outputs/tabla_comparacion_promediado_v3.csv
  outputs/figura_comparacion_promediado_v3.png
  outputs/figura_comparacion_GA_v3.png

Uso
---
  python .\\scripts\\04b_promediado_comparativo_v3.py
==============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS = 256
N_SAMPLES = 256
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES_PRINCIPALES = ["S1 obj", "S2 nomatch"]
CONDICIONES_TODAS = ["S1 obj", "S2 nomatch", "S2 match"]

# Ventana del c240
T_C240_INI_MS = 220
T_C240_FIN_MS = 260

# Para el SNR por sujeto
T_BASELINE_MS = 30           # baseline temprano (igual que script 03)
T_SEN_INI_MS = 100           # ventana de señal evocada amplia para SNR
T_SEN_FIN_MS = 400


def get_project_dirs():
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "scripts":
        project_dir = here.parent.parent
    else:
        project_dir = here.parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return project_dir, output_dir


PROJECT_DIR, OUTPUT_DIR = get_project_dirs()


def ms_a_muestra(t_ms):
    return int(t_ms / 1000 * FS)


def first_existing(paths, label):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    msg = "\n".join(f"  - {Path(p)}" for p in paths)
    raise FileNotFoundError(f"No se encontró {label}. Rutas:\n{msg}")


ENTRADA = first_existing(
    [
        OUTPUT_DIR / "eeg_data_preprocesado_v3.parquet",
        OUTPUT_DIR / "eeg_data_preprocesado_v2b.parquet",
        OUTPUT_DIR / "eeg_data_preprocesado_v2.parquet",
        PROJECT_DIR / "eeg_data_preprocesado_v3.parquet",
    ],
    "preprocesado",
)


# =============================================================================
# PROMEDIADOS
# =============================================================================

def trials_a_matriz(df_sce):
    """
    Toma el df de un (sujeto, canal, condicion) ya preprocesado y devuelve
    una matriz X de tamaño (n_trials, N_SAMPLES) con los trials válidos
    en orden de muestra.
    """
    # Asegura el orden y descarta trials que no tengan N_SAMPLES puntos.
    g = df_sce.sort_values(["trial_num", "muestra"])
    pivot = g.pivot_table(
        index="trial_num",
        columns="muestra",
        values="valor_uV",
        aggfunc="first",
    )
    # Solo trials con todas las muestras.
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[1] != N_SAMPLES:
        # Reindexar para forzar 0..N_SAMPLES-1 si faltan columnas.
        pivot = pivot.reindex(columns=np.arange(N_SAMPLES))
        pivot = pivot.dropna(axis=0, how="any")
    return pivot.values  # (n_trials, N_SAMPLES)


def promedio_homogeneo(X):
    """Promedio aritmético clásico."""
    return X.mean(axis=0)


def promedio_ponderado(X, s_hat=None, eps=1e-12):
    """
    Promedio inhomogéneo tipo Davila & Mobin (1992).

    Pasos:
      1. Si no se da s_hat, usar el promedio homogéneo como estimación
         inicial de la señal evocada.
      2. Calcular para cada trial i el coeficiente de proyección:
            a_i = <x_i, s_hat> / ||s_hat||^2
         Esto estima cuánto se "parece" el trial a la señal evocada.
      3. Los pesos son w_i = max(a_i, 0). Truncamos en cero para no
         restar trials con correlación negativa (eso introduciría
         signo contrario y degradaría el ERP).
      4. Si todos los pesos son cero o casi cero, caemos al homogéneo
         para no devolver NaN.
    """
    if s_hat is None:
        s_hat = promedio_homogeneo(X)

    den = float(np.dot(s_hat, s_hat))
    if den < eps:
        return promedio_homogeneo(X), np.ones(X.shape[0]) / X.shape[0]

    a = X @ s_hat / den               # coeficiente por trial
    w = np.clip(a, 0.0, None)         # truncar negativos

    s = w.sum()
    if s < eps:
        return promedio_homogeneo(X), np.ones(X.shape[0]) / X.shape[0]

    pe = (w[:, None] * X).sum(axis=0) / s
    return pe, w / s                  # pesos normalizados a sumar 1


# =============================================================================
# CÁLCULO DE PE INDIVIDUAL POR ESTRATEGIA
# =============================================================================

def calcular_pe_individual_comparativo(df):
    """
    Para cada (sujeto, grupo, canal, condicion):
      - arma matriz X de trials
      - calcula PE homogéneo
      - calcula PE ponderado (semilla = PE homogéneo)
      - guarda ambos en formato largo

    Devuelve dos DataFrames con columnas:
      sujeto, grupo, canal, condicion, muestra, PE_uV, n_trials, metodo
    """
    filas_hom = []
    filas_pon = []
    info_pesos = []

    grupos = df.groupby(
        ["sujeto", "grupo", "canal", "condicion"], observed=True
    )
    n_total = len(grupos)
    print(f"  Combinaciones sujeto×canal×condición: {n_total:,}")

    for idx, ((sujeto, grupo, canal, cond), sub) in enumerate(grupos):
        if idx % 500 == 0:
            print(f"    {idx:,}/{n_total:,}")

        X = trials_a_matriz(sub)
        n_trials = X.shape[0]
        if n_trials < 2:
            continue

        pe_hom = promedio_homogeneo(X)
        pe_pon, pesos = promedio_ponderado(X, s_hat=pe_hom)

        muestras = np.arange(N_SAMPLES)
        base = {
            "sujeto": sujeto,
            "grupo": grupo,
            "canal": canal,
            "condicion": cond,
            "n_trials": n_trials,
        }
        for m, v in zip(muestras, pe_hom):
            filas_hom.append({**base, "muestra": int(m), "PE_uV": float(v)})
        for m, v in zip(muestras, pe_pon):
            filas_pon.append({**base, "muestra": int(m), "PE_uV": float(v)})

        info_pesos.append({
            **base,
            "peso_min": float(pesos.min()),
            "peso_max": float(pesos.max()),
            "peso_efectivo": float(1.0 / np.sum(pesos ** 2)),  # n_eff
        })

    pe_hom_df = pd.DataFrame(filas_hom)
    pe_pon_df = pd.DataFrame(filas_pon)
    info_pesos_df = pd.DataFrame(info_pesos)
    return pe_hom_df, pe_pon_df, info_pesos_df


def calcular_grand_average(pe_ind):
    grand = (
        pe_ind.groupby(
            ["grupo", "canal", "condicion", "muestra"], observed=True
        )["PE_uV"]
        .agg(grand_avg_uV="mean", std_uV="std", n_sujetos="count")
        .reset_index()
    )
    grand["sem_uV"] = grand["std_uV"] / np.sqrt(grand["n_sujetos"].clip(lower=1))
    return grand


# =============================================================================
# MÉTRICAS DE COMPARACIÓN
# =============================================================================

def snr_por_sujeto(pe_ind, etiqueta):
    """
    SNR por sujeto/canal/condición:
        SNR = var(ventana señal) / var(ventana baseline)
    Más alto = mejor. Devuelve un DataFrame largo.
    """
    m_bl = ms_a_muestra(T_BASELINE_MS)
    m_si = ms_a_muestra(T_SEN_INI_MS)
    m_sf = ms_a_muestra(T_SEN_FIN_MS)

    filas = []
    for (sujeto, grupo, canal, cond), sub in pe_ind.groupby(
        ["sujeto", "grupo", "canal", "condicion"], observed=True
    ):
        s = sub.sort_values("muestra")["PE_uV"].values
        if len(s) != N_SAMPLES:
            continue
        var_bl = np.var(s[:m_bl])
        var_se = np.var(s[m_si:m_sf])
        snr = var_se / var_bl if var_bl > 0 else np.nan
        filas.append({
            "sujeto": sujeto,
            "grupo": grupo,
            "canal": canal,
            "condicion": cond,
            "var_baseline": float(var_bl),
            "var_senal": float(var_se),
            "snr": float(snr) if not np.isnan(snr) else np.nan,
            "metodo": etiqueta,
        })
    return pd.DataFrame(filas)


def split_half_por_sujeto(df_preproc, metodo):
    """
    Para cada sujeto: parte sus trials en pares e impares, calcula PE en
    cada mitad con el método indicado ('homogeneo' o 'ponderado') y mide
    el pico absoluto del c240 en cada mitad. Devuelve correlación entre
    ambas mitades (estabilidad del estimador) corregida con Spearman-Brown.
    """
    m_ini = ms_a_muestra(T_C240_INI_MS)
    m_fin = ms_a_muestra(T_C240_FIN_MS)

    def pico_de(X):
        if X.shape[0] < 1:
            return np.nan
        if metodo == "homogeneo":
            pe = promedio_homogeneo(X)
        else:
            pe = promedio_ponderado(X)[0]
        v = pe[m_ini:m_fin + 1]
        if len(v) == 0:
            return np.nan
        return float(abs(v[np.argmax(np.abs(v))]))

    filas = []
    for (canal, cond) in [(c, k) for c in CANALES_INTERES
                          for k in CONDICIONES_PRINCIPALES]:
        sub = df_preproc[
            (df_preproc["canal"] == canal) & (df_preproc["condicion"] == cond)
        ]
        reg = []
        for (sujeto, grupo), df_s in sub.groupby(
            ["sujeto", "grupo"], observed=True
        ):
            X = trials_a_matriz(df_s)
            if X.shape[0] < 10:
                continue
            mitad_a = X[0::2]
            mitad_b = X[1::2]
            reg.append({
                "sujeto": sujeto,
                "grupo": grupo,
                "pico_a": pico_de(mitad_a),
                "pico_b": pico_de(mitad_b),
            })
        reg = pd.DataFrame(reg).dropna()
        if len(reg) >= 5 and reg["pico_a"].std() > 0 and reg["pico_b"].std() > 0:
            r, p = pearsonr(reg["pico_a"], reg["pico_b"])
            r_sb = (2 * r) / (1 + r) if abs(1 + r) > 1e-9 else np.nan
        else:
            r, p, r_sb = np.nan, np.nan, np.nan
        filas.append({
            "canal": canal,
            "condicion": cond,
            "n_sujetos": len(reg),
            "r_pearson": float(r) if not np.isnan(r) else np.nan,
            "p_corr": float(p) if not np.isnan(p) else np.nan,
            "r_sb": float(r_sb) if not np.isnan(r_sb) else np.nan,
            "metodo": metodo,
        })
    return pd.DataFrame(filas)


def cohen_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    sp = np.sqrt(((len(x) - 1) * sx2 + (len(y) - 1) * sy2)
                 / (len(x) + len(y) - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    return float((np.mean(x) - np.mean(y)) / sp)


def cohen_d_c240(pe_ind, etiqueta):
    """
    Para cada (canal, condicion) extrae el |pico c240| por sujeto y calcula
    Cohen's d entre control y alcoholic.
    """
    m_ini = ms_a_muestra(T_C240_INI_MS)
    m_fin = ms_a_muestra(T_C240_FIN_MS)

    picos = []
    for (sujeto, grupo, canal, cond), sub in pe_ind.groupby(
        ["sujeto", "grupo", "canal", "condicion"], observed=True
    ):
        ventana = sub[sub["muestra"].between(m_ini, m_fin)]
        if ventana.empty:
            continue
        idx = ventana["PE_uV"].abs().idxmax()
        picos.append({
            "sujeto": sujeto,
            "grupo": grupo,
            "canal": canal,
            "condicion": cond,
            "abs_c240": float(abs(ventana.loc[idx, "PE_uV"])),
        })
    df_picos = pd.DataFrame(picos)

    filas = []
    for canal in CANALES_INTERES:
        for cond in CONDICIONES_PRINCIPALES:
            sub = df_picos[
                (df_picos["canal"] == canal) & (df_picos["condicion"] == cond)
            ]
            ctrl = sub[sub["grupo"] == "control"]["abs_c240"].values
            alc = sub[sub["grupo"] == "alcoholic"]["abs_c240"].values
            filas.append({
                "canal": canal,
                "condicion": cond,
                "n_control": len(ctrl),
                "n_alcoholic": len(alc),
                "media_ctrl": float(np.mean(ctrl)) if len(ctrl) else np.nan,
                "media_alc": float(np.mean(alc)) if len(alc) else np.nan,
                "cohen_d": cohen_d(ctrl, alc),
                "metodo": etiqueta,
            })
    return pd.DataFrame(filas)


# =============================================================================
# FIGURAS
# =============================================================================

def plot_grand_average_comparativo(ga_hom, ga_pon, condiciones):
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    n_rows = len(condiciones)
    n_cols = len(CANALES_INTERES)
    fig, axes = plt.subplots(
        n_rows * 2, n_cols,
        figsize=(4.5 * n_cols, 3 * n_rows * 2),
        sharex=True, sharey="row", squeeze=False,
    )
    fig.suptitle(
        "Grand Average: Homogéneo (filas pares) vs Ponderado (filas impares)",
        fontsize=12,
    )

    def plot_ga(ga, ax, canal, cond):
        sub = ga[(ga["canal"] == canal) & (ga["condicion"] == cond)]
        for grupo in ["control", "alcoholic"]:
            d = sub[sub["grupo"] == grupo].sort_values("muestra")
            if d.empty:
                continue
            t = d["muestra"].values / FS * 1000
            avg = d["grand_avg_uV"].values
            sem = d["sem_uV"].fillna(0).values
            ax.plot(t, avg, linewidth=1.6, color=colores[grupo], label=grupo)
            ax.fill_between(t, avg - sem, avg + sem,
                            color=colores[grupo], alpha=0.15)
        ax.axvspan(T_C240_INI_MS, T_C240_FIN_MS, alpha=0.12, color="orange")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.25)

    for i, cond in enumerate(condiciones):
        for j, canal in enumerate(CANALES_INTERES):
            ax_h = axes[2 * i, j]
            ax_p = axes[2 * i + 1, j]
            plot_ga(ga_hom, ax_h, canal, cond)
            plot_ga(ga_pon, ax_p, canal, cond)
            ax_h.set_title(f"{canal} — {cond}\nHomogéneo", fontsize=9)
            ax_p.set_title("Ponderado", fontsize=9)
            if i == 0 and j == 0:
                ax_h.legend(fontsize=8)
            if j == 0:
                ax_h.set_ylabel("µV")
                ax_p.set_ylabel("µV")
    for ax in axes[-1, :]:
        ax.set_xlabel("ms")
    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_comparacion_GA_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")


def plot_metricas(snr_df, sh_df, d_df):
    """
    Una figura con 3 paneles:
      (a) SNR por método (boxplot, todos los canales/condiciones)
      (b) r_SB split-half por canal y método
      (c) |Cohen's d| en c240 por canal y método
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparación: promediado homogéneo vs ponderado", fontsize=12)

    # (a) SNR
    ax = axes[0]
    data = []
    labels = []
    for met in ["homogeneo", "ponderado"]:
        vals = snr_df[snr_df["metodo"] == met]["snr"].dropna().values
        data.append(vals)
        labels.append(f"{met}\n(n={len(vals)})")
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, c in zip(bp["boxes"], ["#94a3b8", "#16a34a"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels(labels)
    ax.set_ylabel("SNR (var señal / var baseline)")
    ax.set_yscale("log")
    ax.set_title("(a) SNR por sujeto×canal×condición")
    ax.grid(True, axis="y", alpha=0.3)

    # (b) Split-half r_SB
    ax = axes[1]
    x = np.arange(len(CANALES_INTERES) * len(CONDICIONES_PRINCIPALES))
    ancho = 0.35
    etiquetas = [f"{c}\n{k}" for c in CANALES_INTERES
                 for k in CONDICIONES_PRINCIPALES]
    for i, met in enumerate(["homogeneo", "ponderado"]):
        vals = []
        for c in CANALES_INTERES:
            for k in CONDICIONES_PRINCIPALES:
                row = sh_df[(sh_df["metodo"] == met)
                            & (sh_df["canal"] == c)
                            & (sh_df["condicion"] == k)]
                vals.append(row["r_sb"].values[0] if len(row) else np.nan)
        ax.bar(x + i * ancho, vals, ancho, label=met,
               color=["#94a3b8", "#16a34a"][i], alpha=0.8)
    ax.set_xticks(x + ancho / 2)
    ax.set_xticklabels(etiquetas, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("r Spearman-Brown")
    ax.axhline(0.7, color="orange", linestyle="--", linewidth=0.9)
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("(b) Confiabilidad split-half")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # (c) |Cohen's d|
    ax = axes[2]
    for i, met in enumerate(["homogeneo", "ponderado"]):
        vals = []
        for c in CANALES_INTERES:
            for k in CONDICIONES_PRINCIPALES:
                row = d_df[(d_df["metodo"] == met)
                           & (d_df["canal"] == c)
                           & (d_df["condicion"] == k)]
                v = row["cohen_d"].values[0] if len(row) else np.nan
                vals.append(abs(v) if not np.isnan(v) else np.nan)
        ax.bar(x + i * ancho, vals, ancho, label=met,
               color=["#94a3b8", "#16a34a"][i], alpha=0.8)
    ax.set_xticks(x + ancho / 2)
    ax.set_xticklabels(etiquetas, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("|Cohen's d|")
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=0.7)
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=0.7)
    ax.axhline(0.8, color="red", linestyle=":", linewidth=0.7)
    ax.set_title("(c) Tamaño de efecto en |c240|")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_comparacion_promediado_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")


# =============================================================================
# MAIN
# =============================================================================

def filter_existing_conditions(df, desired):
    present = list(df["condicion"].dropna().unique())
    return [c for c in desired if c in present]


if __name__ == "__main__":
    print("=" * 70)
    print("Script 04b v3: Promediado homogéneo vs ponderado")
    print("=" * 70)

    print(f"\nCargando: {ENTRADA}")
    df = pd.read_parquet(ENTRADA)
    df = df[df["canal"].isin(CANALES_INTERES)].copy()
    condiciones = filter_existing_conditions(df, CONDICIONES_TODAS)
    df = df[df["condicion"].isin(condiciones)].copy()
    print(f"  Filas: {len(df):,}")
    print(f"  Sujetos: {df['sujeto'].nunique()}")
    print(f"  Condiciones: {condiciones}")

    print("\nCalculando PE individual con ambos métodos...")
    pe_hom, pe_pon, info_pesos = calcular_pe_individual_comparativo(df)
    pe_hom["metodo"] = "homogeneo"
    pe_pon["metodo"] = "ponderado"

    print("\nCalculando Grand Average...")
    ga_hom = calcular_grand_average(pe_hom)
    ga_pon = calcular_grand_average(pe_pon)

    print("\nGuardando PE individuales y GA...")
    pe_hom.to_parquet(OUTPUT_DIR / "eeg_PE_individual_homogeneo_v3.parquet",
                      index=False)
    pe_pon.to_parquet(OUTPUT_DIR / "eeg_PE_individual_ponderado_v3.parquet",
                      index=False)
    ga_hom.to_parquet(OUTPUT_DIR / "eeg_PE_grandaverage_homogeneo_v3.parquet",
                      index=False)
    ga_pon.to_parquet(OUTPUT_DIR / "eeg_PE_grandaverage_ponderado_v3.parquet",
                      index=False)
    info_pesos.to_csv(OUTPUT_DIR / "tabla_pesos_ponderado_v3.csv", index=False)
    print(f"  PE homogeneo: {OUTPUT_DIR / 'eeg_PE_individual_homogeneo_v3.parquet'}")
    print(f"  PE ponderado: {OUTPUT_DIR / 'eeg_PE_individual_ponderado_v3.parquet'}")

    print("\nMétrica 1: SNR por sujeto...")
    snr_hom = snr_por_sujeto(pe_hom, "homogeneo")
    snr_pon = snr_por_sujeto(pe_pon, "ponderado")
    snr_df = pd.concat([snr_hom, snr_pon], ignore_index=True)

    print("\nMétrica 2: Split-half (Spearman-Brown)...")
    sh_hom = split_half_por_sujeto(df, "homogeneo")
    sh_pon = split_half_por_sujeto(df, "ponderado")
    sh_df = pd.concat([sh_hom, sh_pon], ignore_index=True)

    print("\nMétrica 3: Cohen's d en |c240|...")
    d_hom = cohen_d_c240(pe_hom, "homogeneo")
    d_pon = cohen_d_c240(pe_pon, "ponderado")
    d_df = pd.concat([d_hom, d_pon], ignore_index=True)

    # ---------------------------------------------------------------
    # Resumen impreso
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)

    print("\nSNR (mediana por método sobre sujeto×canal×condición):")
    for met in ["homogeneo", "ponderado"]:
        v = snr_df[snr_df["metodo"] == met]["snr"].dropna()
        print(f"  {met:<10s}  mediana={v.median():.3f}  "
              f"media={v.mean():.3f}  n={len(v)}")

    print("\nSplit-half (r Spearman-Brown promedio por método):")
    for met in ["homogeneo", "ponderado"]:
        v = sh_df[sh_df["metodo"] == met]["r_sb"].dropna()
        print(f"  {met:<10s}  r_sb medio={v.mean():.3f}  "
              f"(canales×condiciones={len(v)})")

    print("\n|Cohen's d| en |c240| (promedio por método sobre canales×condiciones):")
    for met in ["homogeneo", "ponderado"]:
        v = d_df[d_df["metodo"] == met]["cohen_d"].dropna().abs()
        print(f"  {met:<10s}  |d| medio={v.mean():.3f}  "
              f"(canales×condiciones={len(v)})")

    # ---------------------------------------------------------------
    # Tabla unificada
    # ---------------------------------------------------------------
    tabla = []
    for met in ["homogeneo", "ponderado"]:
        snr_v = snr_df[snr_df["metodo"] == met]["snr"].dropna()
        sh_v = sh_df[sh_df["metodo"] == met]["r_sb"].dropna()
        d_v = d_df[d_df["metodo"] == met]["cohen_d"].dropna().abs()
        tabla.append({
            "metodo": met,
            "snr_mediana": float(snr_v.median()),
            "snr_media": float(snr_v.mean()),
            "r_sb_medio": float(sh_v.mean()),
            "abs_d_medio": float(d_v.mean()),
            "abs_d_max": float(d_v.max()),
        })
    tabla_df = pd.DataFrame(tabla)
    tabla_df.to_csv(OUTPUT_DIR / "tabla_comparacion_promediado_v3.csv",
                    index=False)
    print(f"\nTabla guardada: {OUTPUT_DIR / 'tabla_comparacion_promediado_v3.csv'}")

    # Tablas detalladas también
    snr_df.to_csv(OUTPUT_DIR / "tabla_comparacion_snr_v3.csv", index=False)
    sh_df.to_csv(OUTPUT_DIR / "tabla_comparacion_splithalf_v3.csv", index=False)
    d_df.to_csv(OUTPUT_DIR / "tabla_comparacion_cohend_v3.csv", index=False)

    # ---------------------------------------------------------------
    # Figuras
    # ---------------------------------------------------------------
    print("\nGenerando figuras...")
    plot_metricas(snr_df, sh_df, d_df)
    plot_grand_average_comparativo(ga_hom, ga_pon, CONDICIONES_PRINCIPALES)

    print("\n[OK] Script 04b v3 finalizado.")
    print("\nPróximo paso sugerido:")
    print("  Si el ponderado gana en SNR + split-half + |d|, podés usar")
    print("  eeg_PE_individual_ponderado_v3.parquet como entrada del 05.")
