"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 09: Validación de Robustez del Pipeline
==============================================================================

Propósito
---------
Demostrar que los hallazgos del Script 06 v2 (diferencias significativas
en |c240| entre control y alcoholic) NO dependen de elecciones
metodológicas puntuales. Se aplican tres pruebas de robustez:

1. JACKKNIFE INTER-SUJETO: para cada canal x condicion, se recalcula el
   t-test omitiendo un sujeto por vez. Si el efecto es robusto, ningún
   sujeto individual debería ser responsable del resultado significativo.
   Se reporta la distribución de p-valores resultante.

2. SPLIT-HALF DE TRIALS: para cada sujeto, se promedian la primera mitad
   y la segunda mitad de los trials por separado. Se calcula la
   correlación entre los |c240| obtenidos en cada mitad. Una correlación
   alta indica que la métrica es estable dentro del sujeto.

3. BARRIDO DE VENTANA: se extrae el c240 con distintas ventanas
   temporales (200-260, 210-270, 220-260, 230-270, 210-280) y se
   reporta el p-valor del t-test entre grupos en cada caso.

Entrada:  outputs/eeg_data_preprocesado_v2.parquet  (Script 03 v2)
          outputs/eeg_PE_individual_v2.parquet      (Script 04 v2)
          outputs/eeg_c240_extraido_v2.csv          (Script 05 v2)
Salida:   outputs/tabla_robustez_jackknife_v2.csv
          outputs/tabla_robustez_splithalf_v2.csv
          outputs/tabla_robustez_ventana_v2.csv
          outputs/figura_robustez_jackknife_v2.png
          outputs/figura_robustez_splithalf_v2.png
          outputs/figura_robustez_ventana_v2.png

Uso:
    python 09_robustez.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind, pearsonr

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS              = 256
ALPHA           = 0.05
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES     = ["S1 obj", "S2 nomatch"]

# Ventanas alternativas para el barrido (en ms)
VENTANAS_BARRIDO = [
    (200, 260),
    (210, 270),
    (220, 260),   # ventana principal del anteproyecto
    (230, 270),
    (210, 280),
    (200, 280),
]

ENTRADA_PREPROC = Path("../outputs/eeg_data_preprocesado_v2.parquet")
ENTRADA_PE_IND  = Path("../outputs/eeg_PE_individual_v2.parquet")
ENTRADA_C240    = Path("../outputs/eeg_c240_extraido_v2.csv")

SALIDA_JK   = Path("../outputs/tabla_robustez_jackknife_v2.csv")
SALIDA_SH   = Path("../outputs/tabla_robustez_splithalf_v2.csv")
SALIDA_VENT = Path("../outputs/tabla_robustez_ventana_v2.csv")


def extraer_pico_abs(senal_avg, m_ini, m_fin):
    """Pico por magnitud absoluta dentro de una ventana en muestras."""
    ventana = senal_avg[m_ini:m_fin + 1]
    if len(ventana) == 0:
        return np.nan
    idx_rel = int(np.argmax(np.abs(ventana)))
    return ventana[idx_rel]


# =============================================================================
# 1) JACKKNIFE INTER-SUJETO
# =============================================================================

def jackknife(df_c240):
    """
    Para cada canal x condicion, omite un sujeto por vez y recalcula el
    t-test entre grupos. Reporta min/max/mediana de p-valores y el % de
    iteraciones donde el resultado sigue siendo significativo.
    """
    print("\n--- 1) Jackknife inter-sujeto ---")
    filas = []
    for condicion in CONDICIONES:
        for canal in CANALES_INTERES:
            sub = df_c240[
                (df_c240["canal"] == canal) &
                (df_c240["condicion"] == condicion)
            ].dropna(subset=["amplitud_abs_uV"])
            sujetos = sub["sujeto"].unique()

            p_valores = []
            for s in sujetos:
                tmp = sub[sub["sujeto"] != s]
                ctrl = tmp[tmp["grupo"] == "control"]["amplitud_abs_uV"].values
                alc  = tmp[tmp["grupo"] == "alcoholic"]["amplitud_abs_uV"].values
                if len(ctrl) < 2 or len(alc) < 2:
                    continue
                _, p = ttest_ind(ctrl, alc, equal_var=False)
                p_valores.append(p)

            p_arr = np.array(p_valores)
            pct_sig = 100 * (p_arr < ALPHA).mean() if len(p_arr) else 0

            filas.append({
                "canal":      canal,
                "condicion":  condicion,
                "n_iter":     len(p_arr),
                "p_min":      float(p_arr.min())    if len(p_arr) else np.nan,
                "p_max":      float(p_arr.max())    if len(p_arr) else np.nan,
                "p_mediana":  float(np.median(p_arr)) if len(p_arr) else np.nan,
                "pct_sig":    float(pct_sig),
            })

    return pd.DataFrame(filas)


def graficar_jackknife(df_jk):
    """Barras con el % de iteraciones del jackknife que siguen significativas."""
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 4.5),
                             sharey=True)
    fig.suptitle(
        "Robustez Jackknife: % de iteraciones (omitiendo un sujeto) que\n"
        "mantienen p < 0.05 — Si es 100%, ningun sujeto sostiene el resultado solo",
        fontsize=12
    )
    x = np.arange(len(CANALES_INTERES))
    for ax, cond in zip(axes, CONDICIONES):
        sub = df_jk[df_jk["condicion"] == cond]
        pcts = [sub[sub["canal"] == c]["pct_sig"].values[0] for c in CANALES_INTERES]
        ax.bar(x, pcts, color="#0d9488", alpha=0.8)
        ax.axhline(100, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_ylim([0, 105])
        ax.set_ylabel("% iteraciones con p<0.05")
        ax.set_title(f"Condicion: {cond}", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        for i, p in enumerate(pcts):
            ax.text(i, p + 1.5, f"{p:.0f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("../outputs/figura_robustez_jackknife_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_robustez_jackknife_v2.png'")


# =============================================================================
# 2) SPLIT-HALF DE TRIALS
# =============================================================================

def split_half(df_preproc):
    """
    Para cada sujeto x canal x condicion, divide los trials en pares e
    impares, promedia cada mitad, extrae |c240| de cada una y calcula
    la correlacion de Pearson entre ambas mitades a nivel poblacional.
    """
    print("\n--- 2) Split-half de trials ---")
    M_INI = int(220 / 1000 * FS)
    M_FIN = int(260 / 1000 * FS)

    filas = []
    for condicion in CONDICIONES:
        df_cond = df_preproc[df_preproc["condicion"] == condicion]

        for canal in CANALES_INTERES:
            df_canal = df_cond[df_cond["canal"] == canal]
            registros = []

            for (sujeto, grupo), df_sub in df_canal.groupby(["sujeto", "grupo"]):
                trials = df_sub["trial_num"].unique()
                trials_sorted = np.sort(trials)
                pares   = trials_sorted[::2]
                impares = trials_sorted[1::2]
                if len(pares) < 5 or len(impares) < 5:
                    continue

                # Promediar cada mitad
                def avg_y_pico(df_subset):
                    avg = (df_subset.groupby("muestra")["valor_uV"].mean()
                                    .sort_index().values)
                    return abs(extraer_pico_abs(avg, M_INI, M_FIN))

                pico_p = avg_y_pico(df_sub[df_sub["trial_num"].isin(pares)])
                pico_i = avg_y_pico(df_sub[df_sub["trial_num"].isin(impares)])

                registros.append({
                    "sujeto": sujeto, "grupo": grupo,
                    "abs_pares": pico_p, "abs_impares": pico_i
                })

            if len(registros) < 5:
                continue

            df_reg = pd.DataFrame(registros).dropna()
            if len(df_reg) < 5:
                continue
            r, p_corr = pearsonr(df_reg["abs_pares"], df_reg["abs_impares"])
            r_sb = (2 * r) / (1 + r) if abs(1 + r) > 1e-9 else np.nan  # Spearman-Brown

            filas.append({
                "canal":     canal,
                "condicion": condicion,
                "n_sujetos": len(df_reg),
                "r_pearson": float(r),
                "p_corr":    float(p_corr),
                "r_sb":      float(r_sb),
            })

    return pd.DataFrame(filas)


def graficar_split_half(df_sh):
    """Barras de la correlacion split-half (Spearman-Brown corregida)."""
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 4.5),
                             sharey=True)
    fig.suptitle(
        "Confiabilidad Split-Half: correlacion entre |c240| obtenidas con\n"
        "trials pares e impares (corregida Spearman-Brown)",
        fontsize=12
    )
    x = np.arange(len(CANALES_INTERES))
    for ax, cond in zip(axes, CONDICIONES):
        sub = df_sh[df_sh["condicion"] == cond]
        rs = [sub[sub["canal"] == c]["r_sb"].values[0]
              if not sub[sub["canal"] == c].empty else 0
              for c in CANALES_INTERES]
        ax.bar(x, rs, color="#9333ea", alpha=0.75)
        ax.axhline(0.7, color="orange", linestyle="--", linewidth=0.9,
                   label="r=0.7 (buena confiabilidad)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_ylim([-0.2, 1.05])
        ax.set_ylabel("r (Spearman-Brown)")
        ax.set_title(f"Condicion: {cond}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for i, r in enumerate(rs):
            ax.text(i, r + 0.03, f"{r:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("../outputs/figura_robustez_splithalf_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_robustez_splithalf_v2.png'")


# =============================================================================
# 3) BARRIDO DE VENTANA TEMPORAL
# =============================================================================

def barrido_ventana(df_pe_ind):
    """
    Recalcula |c240| y t-test con distintas ventanas temporales.
    """
    print("\n--- 3) Barrido de ventana temporal ---")
    filas = []
    for t_ini, t_fin in VENTANAS_BARRIDO:
        m_ini = int(t_ini / 1000 * FS)
        m_fin = int(t_fin / 1000 * FS)

        for condicion in CONDICIONES:
            for canal in CANALES_INTERES:
                ctrl, alc = [], []
                df_filt = df_pe_ind[
                    (df_pe_ind["canal"] == canal) &
                    (df_pe_ind["condicion"] == condicion)
                ]
                for (sujeto, grupo), df_sub in df_filt.groupby(["sujeto", "grupo"]):
                    ventana = df_sub[
                        (df_sub["muestra"] >= m_ini) &
                        (df_sub["muestra"] <= m_fin)
                    ]
                    if ventana.empty:
                        continue
                    idx_pico = ventana["PE_uV"].abs().idxmax()
                    amp_abs  = abs(ventana.loc[idx_pico, "PE_uV"])
                    if grupo == "control":
                        ctrl.append(amp_abs)
                    else:
                        alc.append(amp_abs)

                if len(ctrl) < 2 or len(alc) < 2:
                    continue

                t_stat, p_valor = ttest_ind(ctrl, alc, equal_var=False)
                filas.append({
                    "ventana_ms":     f"{t_ini}-{t_fin}",
                    "canal":          canal,
                    "condicion":      condicion,
                    "n_control":      len(ctrl),
                    "n_alcoholic":    len(alc),
                    "media_ctrl":     float(np.mean(ctrl)),
                    "media_alc":      float(np.mean(alc)),
                    "t_estadistico":  float(t_stat),
                    "p_valor":        float(p_valor),
                    "sig":            "Si" if p_valor < ALPHA else "No",
                })
    return pd.DataFrame(filas)


def graficar_barrido(df_vent):
    """Heatmap: ventanas (filas) x canal_condicion (columnas), color = -log10(p)."""
    df_vent["canal_cond"] = df_vent["canal"] + "\n" + df_vent["condicion"]
    pivot = df_vent.pivot(index="ventana_ms", columns="canal_cond", values="p_valor")
    # Ordenar por ventanas naturales
    orden_ventanas = [f"{a}-{b}" for a, b in VENTANAS_BARRIDO]
    pivot = pivot.reindex([v for v in orden_ventanas if v in pivot.index])

    logp = -np.log10(pivot.values + 1e-12)
    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 2,
                                    0.7 * pivot.shape[0] + 2))
    im = ax.imshow(logp, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Barrido de ventana: -log10(p) del t-test |c240|\n"
                 "Mayor valor = mas significativo. Linea naranja = p=0.05",
                 fontsize=11)

    # Anotar p-valores
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            p = pivot.values[i, j]
            color = "white" if logp[i, j] < 4 else "black"
            ax.text(j, i, f"{p:.1e}" if p < 1e-3 else f"{p:.3f}",
                    ha="center", va="center", fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("-log10(p)")
    # Linea de referencia p=0.05 en el colorbar
    cbar.ax.axhline(-np.log10(0.05), color="orange", linewidth=2)

    plt.tight_layout()
    plt.savefig("../outputs/figura_robustez_ventana_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_robustez_ventana_v2.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Script 09: Validacion de Robustez")
    print("=" * 60)

    if not ENTRADA_C240.exists():
        raise FileNotFoundError(f"No se encontro '{ENTRADA_C240}'.")
    if not ENTRADA_PREPROC.exists():
        raise FileNotFoundError(f"No se encontro '{ENTRADA_PREPROC}'.")
    if not ENTRADA_PE_IND.exists():
        raise FileNotFoundError(f"No se encontro '{ENTRADA_PE_IND}'.")

    print("\nCargando datos...")
    df_c240    = pd.read_csv(ENTRADA_C240)
    df_preproc = pd.read_parquet(ENTRADA_PREPROC)
    df_pe_ind  = pd.read_parquet(ENTRADA_PE_IND)

    # 1) Jackknife
    df_jk = jackknife(df_c240)
    print(df_jk.to_string(index=False))
    df_jk.to_csv(SALIDA_JK, index=False)
    print(f"\nTabla guardada: {SALIDA_JK}")
    graficar_jackknife(df_jk)

    # 2) Split-half
    df_sh = split_half(df_preproc)
    print(df_sh.to_string(index=False))
    df_sh.to_csv(SALIDA_SH, index=False)
    print(f"\nTabla guardada: {SALIDA_SH}")
    graficar_split_half(df_sh)

    # 3) Barrido de ventana
    df_vent = barrido_ventana(df_pe_ind)
    print(df_vent.to_string(index=False))
    df_vent.to_csv(SALIDA_VENT, index=False)
    print(f"\nTabla guardada: {SALIDA_VENT}")
    graficar_barrido(df_vent)

    print("\n[OK] Script 09 finalizado.")
