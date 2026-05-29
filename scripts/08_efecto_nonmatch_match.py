"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 08: Efecto Nonmatch-Match (Análisis Secundario)
==============================================================================

Propósito
---------
Análisis secundario previsto en el anteproyecto: evaluar si la
diferenciación entre estímulos repetidos (S2 match) y estímulos nuevos
(S2 nomatch) está reducida o alterada en el grupo alcoholic.

Para cada sujeto se calcula:
    delta_A_c240 = |A_c240, S2 nomatch| - |A_c240, S2 match|

Esta delta mide CUÁNTO MÁS responde el cerebro a un estímulo nuevo
(que requiere comparar contra la representación en memoria) respecto
de un estímulo repetido. Bajo la hipótesis del anteproyecto, los
controles deberían mostrar una delta mayor que los alcohólicos
(memoria visual de trabajo más eficiente).

Contrastes
----------
1. Welch t-test entre grupos sobre delta_A_c240, por canal
2. Cohen's d, Bonferroni, FDR Benjamini-Hochberg (4 canales = 4 tests)
3. One-sample t-test dentro de cada grupo (delta > 0?) — verifica que
   el efecto nonmatch-match exista en cada grupo por separado

Pipeline
--------
Este script promedia los trials de S2 match desde el preprocesado v2b
(que SÍ incluye esa condición) y extrae |c240| con el mismo criterio
de magnitud absoluta del Script 05 v2. Luego junta los datos con los
de S2 nomatch ya extraídos en Script 05 v2.

Entrada:  outputs/eeg_data_preprocesado_v2b.parquet  (Script 03 v2b)
          outputs/eeg_c240_extraido_v2.csv          (Script 05 v2 — para S2 nomatch)
Salida:   outputs/eeg_c240_match_extraido_v2.csv
          outputs/tabla_efecto_nonmatch_match_v2.csv
          outputs/figura_efecto_nonmatch_match_v2.png

Uso:
    python 08_efecto_nonmatch_match.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind, ttest_1samp

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS               = 256
ALPHA            = 0.05
CANALES_INTERES  = ["P8", "PO8", "T8", "TP8"]
N_TESTS          = len(CANALES_INTERES)  # 4 canales -> 4 tests

# Ventana c240 (consistente con Script 05 v2)
T_INI_MS, T_FIN_MS = 220, 260
M_INI = int(T_INI_MS / 1000 * FS)
M_FIN = int(T_FIN_MS / 1000 * FS)

ENTRADA_PREPROC = Path("../outputs/eeg_data_preprocesado_v2b.parquet")
ENTRADA_NOMATCH = Path("../outputs/eeg_c240_extraido_v2.csv")
SALIDA_MATCH    = Path("../outputs/eeg_c240_match_extraido_v2.csv")
SALIDA_TABLA    = Path("../outputs/tabla_efecto_nonmatch_match_v2.csv")

# =============================================================================
# FUNCIONES
# =============================================================================

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if s_pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / s_pooled


def correccion_bh(p_valores):
    p = np.asarray(p_valores, dtype=float)
    n = len(p)
    orden = np.argsort(p)
    p_sorted = p[orden]
    q_sorted = p_sorted * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[orden] = q_sorted
    return q


def calcular_PE_y_extraer_match(df_preproc):
    """
    Para cada sujeto x canal, promedia trials de S2 match y extrae
    el pico c240 por magnitud absoluta dentro de 220-260 ms.
    Devuelve un DataFrame con columnas: sujeto, grupo, canal,
    amplitud_uV, amplitud_abs_uV, latencia_ms, polaridad.
    """
    df_match = df_preproc[df_preproc["condicion"] == "S2 match"]
    if df_match.empty:
        raise ValueError("No hay datos de S2 match en el preprocesado v2b.")

    # PE individual
    PE = (
        df_match.groupby(["sujeto", "grupo", "canal", "muestra"])
        ["valor_uV"]
        .agg(PE_uV="mean", n_trials="count")
        .reset_index()
    )

    # Extraer c240
    filas = []
    for (sujeto, grupo, canal), df_sub in PE.groupby(["sujeto", "grupo", "canal"]):
        ventana = df_sub[
            (df_sub["muestra"] >= M_INI) &
            (df_sub["muestra"] <= M_FIN)
        ].sort_values("muestra")
        if ventana.empty:
            continue
        idx_pico     = ventana["PE_uV"].abs().idxmax()
        amplitud     = ventana.loc[idx_pico, "PE_uV"]
        muestra_pico = ventana.loc[idx_pico, "muestra"]
        filas.append({
            "sujeto":           sujeto,
            "grupo":            grupo,
            "canal":            canal,
            "condicion":        "S2 match",
            "amplitud_uV":      amplitud,
            "amplitud_abs_uV":  abs(amplitud),
            "latencia_ms":      muestra_pico / FS * 1000,
            "polaridad":        "+" if amplitud >= 0 else "-",
            "n_trials":         ventana["n_trials"].iloc[0],
        })

    return pd.DataFrame(filas)


def calcular_efecto(df_match, df_nomatch_csv):
    """
    Cruza match y nomatch por sujeto x canal, calcula delta y aplica
    contrastes.
    """
    # Cargar S2 nomatch del CSV de Script 05 v2 (filtrar solo la condicion)
    df_nomatch = df_nomatch_csv[df_nomatch_csv["condicion"] == "S2 nomatch"].copy()
    df_nomatch = df_nomatch[["sujeto", "grupo", "canal", "amplitud_abs_uV"]]
    df_nomatch = df_nomatch.rename(columns={"amplitud_abs_uV": "abs_nomatch"})

    df_m = df_match[["sujeto", "grupo", "canal", "amplitud_abs_uV"]].copy()
    df_m = df_m.rename(columns={"amplitud_abs_uV": "abs_match"})

    df_merge = pd.merge(df_nomatch, df_m, on=["sujeto", "grupo", "canal"], how="inner")
    df_merge["delta"] = df_merge["abs_nomatch"] - df_merge["abs_match"]

    return df_merge


def contraste_estadistico(df_merge):
    """Welch t-test entre grupos sobre delta, por canal."""
    filas = []
    for canal in CANALES_INTERES:
        ctrl = df_merge[(df_merge["canal"] == canal) &
                        (df_merge["grupo"] == "control")]["delta"].dropna().values
        alc  = df_merge[(df_merge["canal"] == canal) &
                        (df_merge["grupo"] == "alcoholic")]["delta"].dropna().values

        t_stat, p_valor = ttest_ind(ctrl, alc, equal_var=False)
        d = cohen_d(ctrl, alc)

        # One-sample t-test: ¿delta > 0 dentro de cada grupo?
        t_ctrl, p_ctrl = ttest_1samp(ctrl, 0.0)
        t_alc,  p_alc  = ttest_1samp(alc, 0.0)

        filas.append({
            "canal":             canal,
            "n_control":         len(ctrl),
            "n_alcoholic":       len(alc),
            "delta_media_ctrl":  float(np.mean(ctrl)),
            "delta_sd_ctrl":     float(np.std(ctrl, ddof=1)),
            "delta_media_alc":   float(np.mean(alc)),
            "delta_sd_alc":      float(np.std(alc, ddof=1)),
            "t_entre_grupos":    float(t_stat),
            "p_entre_grupos":    float(p_valor),
            "cohen_d":           float(d),
            "p_ctrl_vs_0":       float(p_ctrl),
            "p_alc_vs_0":        float(p_alc),
        })

    resultados = pd.DataFrame(filas)
    p = resultados["p_entre_grupos"].values
    resultados["p_bonferroni"] = np.clip(p * N_TESTS, 0, 1)
    resultados["p_fdr_bh"]     = correccion_bh(p)
    return resultados


def imprimir_tabla(resultados):
    print("\n" + "=" * 100)
    print("EFECTO NONMATCH-MATCH:  delta = |c240, S2 nomatch| - |c240, S2 match|")
    print("=" * 100)
    print(f"  {'Canal':<5} {'delta Ctrl (uV)':>20} {'delta Alc (uV)':>20} "
          f"{'t':>7} {'p':>9} {'p_BH':>8} {'d':>6}")
    print(f"  {'-'*5} {'-'*20} {'-'*20} {'-'*7} {'-'*9} {'-'*8} {'-'*6}")
    for _, f in resultados.iterrows():
        ctrl_str = f"{f['delta_media_ctrl']:+.3f}+/-{f['delta_sd_ctrl']:.3f}"
        alc_str  = f"{f['delta_media_alc']:+.3f}+/-{f['delta_sd_alc']:.3f}"
        marca = "*" if f["p_fdr_bh"] < ALPHA else " "
        print(f"  {f['canal']:<5} {ctrl_str:>20} {alc_str:>20} "
              f"{f['t_entre_grupos']:>7.3f} {f['p_entre_grupos']:>9.4f} "
              f"{f['p_fdr_bh']:>8.4f} {f['cohen_d']:>6.2f} {marca}")
    print("\n  One-sample t-test (¿delta > 0 dentro de cada grupo?)")
    print(f"  {'Canal':<5} {'p_ctrl_vs_0':>14} {'p_alc_vs_0':>14}")
    for _, f in resultados.iterrows():
        print(f"  {f['canal']:<5} {f['p_ctrl_vs_0']:>14.4f} {f['p_alc_vs_0']:>14.4f}")
    print("=" * 100)


def graficar_efecto(df_merge, resultados):
    fig, axes = plt.subplots(1, len(CANALES_INTERES),
                             figsize=(5 * len(CANALES_INTERES), 5),
                             sharey=True)
    fig.suptitle(
        "Efecto Nonmatch-Match — delta = |c240, S2 nomatch| - |c240, S2 match|\n"
        "Valores positivos indican mayor respuesta a estimulos nuevos",
        fontsize=12
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for col, canal in enumerate(CANALES_INTERES):
        ax = axes[col]
        datos_plot, etiquetas, colores_bp = [], [], []

        for grupo in ["control", "alcoholic"]:
            vals = df_merge[
                (df_merge["canal"] == canal) &
                (df_merge["grupo"] == grupo)
            ]["delta"].dropna().values
            datos_plot.append(vals)
            n     = len(vals)
            media = np.mean(vals)
            sd    = np.std(vals, ddof=1)
            etiquetas.append(f"{grupo.capitalize()}\nn={n}\nd={media:+.2f}+/-{sd:.2f}")
            colores_bp.append(colores[grupo])

        bp = ax.boxplot(datos_plot, patch_artist=True,
                       medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colores_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        for i, (vals, color) in enumerate(zip(datos_plot, colores_bp)):
            x_jitter = np.random.normal(i + 1, 0.07, size=len(vals))
            ax.scatter(x_jitter, vals, alpha=0.4, color=color, s=12, zorder=3)

        res_fila = resultados[resultados["canal"] == canal].iloc[0]
        q = res_fila["p_fdr_bh"]
        d = res_fila["cohen_d"]
        anotacion = f"q = {q:.4f}  |  d = {d:.2f}"
        color_tit = "red" if q < ALPHA else "gray"

        ax.set_title(f"Canal: {canal}\n{anotacion}", fontsize=10, color=color_tit)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(etiquetas, fontsize=8)
        ax.set_ylabel("delta amplitud (uV)")
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_efecto_nonmatch_match_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_efecto_nonmatch_match_v2.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Script 08: Efecto Nonmatch-Match")
    print("=" * 60)

    if not ENTRADA_PREPROC.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA_PREPROC}'.\n"
            "Asegurate de haber corrido primero el Script 03 v2b."
        )
    if not ENTRADA_NOMATCH.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA_NOMATCH}'.\n"
            "Asegurate de haber corrido primero el Script 05 v2."
        )

    print(f"\nCargando preprocesado v2b...")
    df_preproc = pd.read_parquet(ENTRADA_PREPROC)
    print(f"  {df_preproc['sujeto'].nunique()} sujetos")

    print(f"Cargando S2 nomatch ya extraido...")
    df_nomatch = pd.read_csv(ENTRADA_NOMATCH)

    print("\nPromediando trials S2 match y extrayendo c240...")
    df_match = calcular_PE_y_extraer_match(df_preproc)
    df_match.to_csv(SALIDA_MATCH, index=False)
    print(f"  Guardado en '{SALIDA_MATCH}'")

    print("\nCruzando datos match y nomatch...")
    df_merge = calcular_efecto(df_match, df_nomatch)
    print(f"  {len(df_merge)} filas (sujetos x canales con datos en ambas condiciones)")

    print("\nContrastes estadisticos...")
    resultados = contraste_estadistico(df_merge)
    imprimir_tabla(resultados)

    resultados.to_csv(SALIDA_TABLA, index=False)
    print(f"\nTabla guardada en '{SALIDA_TABLA}'")

    print("\nGenerando grafico...")
    np.random.seed(42)
    graficar_efecto(df_merge, resultados)

    print("\n[OK] Script 08 finalizado.")
