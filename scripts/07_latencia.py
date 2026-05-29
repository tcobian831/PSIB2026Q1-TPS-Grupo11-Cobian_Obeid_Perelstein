"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 07: Análisis de Latencia del Componente c240/VMP
==============================================================================

Propósito
---------
Análisis secundario previsto en el anteproyecto: comparar la LATENCIA del
pico c240/VMP entre grupos. Mientras la amplitud refleja "cuán evocado
está el componente", la latencia refleja "cuán rápido aparece" — un
índice de eficiencia del procesamiento visual de memoria de trabajo.

La columna `latencia_ms` ya viene calculada por el Script 05 v2; este
script solo aplica los contrastes estadísticos y genera las figuras.

Análisis
--------
1. Estadística descriptiva de latencia por grupo, canal y condición
2. t-test de Welch para latencia entre control y alcoholic
3. Cohen's d, Bonferroni y FDR Benjamini-Hochberg
4. Boxplots y barras
5. Diagnóstico: descartar latencias en bordes de la ventana (señalan
   pico fuera de la ventana y por lo tanto no son latencias válidas)

Entrada:  outputs/eeg_c240_extraido_v2.csv
Salida:   outputs/tabla_latencia_v2.csv
          outputs/figura_latencia_boxplot_v2.png
          outputs/figura_latencia_barras_v2.png

Uso:
    python 07_latencia.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

ALPHA           = 0.05
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES     = ["S1 obj", "S2 nomatch"]
N_TESTS         = len(CANALES_INTERES) * len(CONDICIONES)

# Bordes de la ventana del Script 05 v2 (220-260 ms a 256 Hz)
LATENCIA_INI_BORDE = 222   # ms — picos por debajo se consideran "borde inferior"
LATENCIA_FIN_BORDE = 257   # ms — picos por encima se consideran "borde superior"

ENTRADA = Path("../outputs/eeg_c240_extraido_v2.csv")
SALIDA  = Path("../outputs/tabla_latencia_v2.csv")

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
    ranks = np.empty(n, dtype=int)
    ranks[orden] = np.arange(1, n + 1)
    p_sorted = p[orden]
    q_sorted = p_sorted * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[orden] = q_sorted
    return q


def calcular_estadisticas_latencia(df):
    """t-test de Welch comparando latencia control vs alcoholic."""
    filas = []
    for condicion in CONDICIONES:
        for canal in CANALES_INTERES:
            sel = (df["canal"] == canal) & (df["condicion"] == condicion)
            ctrl = df[sel & (df["grupo"] == "control")]["latencia_ms"].dropna().values
            alc  = df[sel & (df["grupo"] == "alcoholic")]["latencia_ms"].dropna().values

            t_stat, p_valor = ttest_ind(ctrl, alc, equal_var=False)
            d = cohen_d(ctrl, alc)

            filas.append({
                "canal":              canal,
                "condicion":          condicion,
                "n_control":          len(ctrl),
                "n_alcoholic":        len(alc),
                "media_lat_control":  float(np.mean(ctrl)),
                "sd_lat_control":     float(np.std(ctrl, ddof=1)),
                "media_lat_alcoholic": float(np.mean(alc)),
                "sd_lat_alcoholic":    float(np.std(alc, ddof=1)),
                "t_estadistico":      float(t_stat),
                "p_valor":            float(p_valor),
                "cohen_d":            float(d),
            })

    resultados = pd.DataFrame(filas)
    p = resultados["p_valor"].values
    resultados["p_bonferroni"] = np.clip(p * N_TESTS, 0, 1)
    resultados["p_fdr_bh"]     = correccion_bh(p)
    resultados["sig_fdr"] = (resultados["p_fdr_bh"] < ALPHA).map({True:"Si", False:"No"})
    return resultados


def imprimir_tabla(resultados):
    print("\n" + "=" * 90)
    print("LATENCIA c240/VMP (media +/- SD) + t-test")
    print("=" * 90)
    for condicion in CONDICIONES:
        print(f"\nCondicion: {condicion}")
        print(f"  {'Canal':<5} {'Ctrl (ms)':>16} {'Alc (ms)':>16} "
              f"{'t':>7} {'p':>9} {'p_BH':>8} {'d':>6}")
        sub = resultados[resultados["condicion"] == condicion]
        for _, f in sub.iterrows():
            ctrl_str = f"{f['media_lat_control']:.1f}+/-{f['sd_lat_control']:.1f}"
            alc_str  = f"{f['media_lat_alcoholic']:.1f}+/-{f['sd_lat_alcoholic']:.1f}"
            marca = "*" if f["p_fdr_bh"] < ALPHA else " "
            print(f"  {f['canal']:<5} {ctrl_str:>16} {alc_str:>16} "
                  f"{f['t_estadistico']:>7.3f} {f['p_valor']:>9.4f} "
                  f"{f['p_fdr_bh']:>8.4f} {f['cohen_d']:>6.2f} {marca}")
    print(f"\n  * = significativo tras FDR (BH) con alpha={ALPHA}")
    print("=" * 90)


def diagnostico_bordes(df):
    """
    Reporta cuántas latencias caen en los bordes de la ventana, lo que
    indica que el pico real probablemente esté FUERA de la ventana y
    estos datos sean ruido.
    """
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO: latencias en bordes de la ventana 220-260 ms")
    print("=" * 70)
    print("Si >30% de las latencias caen en borde, considerar ampliar la ventana")
    for cond in CONDICIONES:
        print(f"\nCondicion: {cond}")
        for canal in CANALES_INTERES:
            for grupo in ["control", "alcoholic"]:
                sub = df[
                    (df["canal"] == canal) &
                    (df["condicion"] == cond) &
                    (df["grupo"] == grupo)
                ]["latencia_ms"].dropna()
                if len(sub) == 0:
                    continue
                n_inf = (sub < LATENCIA_INI_BORDE).sum()
                n_sup = (sub > LATENCIA_FIN_BORDE).sum()
                pct_borde = 100 * (n_inf + n_sup) / len(sub)
                marca = " <-- ALTO" if pct_borde > 30 else ""
                print(f"  {canal:<5} {grupo:<11} n={len(sub):3d}  "
                      f"borde_inf:{n_inf:3d}  borde_sup:{n_sup:3d}  "
                      f"({pct_borde:5.1f}%){marca}")
    print("=" * 70)


def graficar_boxplot(df, resultados):
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharey=True
    )
    fig.suptitle(
        "Latencia del pico c240/VMP por grupo\n"
        "Ventana 220-260 ms (pico por magnitud absoluta)",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for fila, condicion in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]
            datos_plot, etiquetas, colores_bp = [], [], []

            for grupo in ["control", "alcoholic"]:
                vals = df[
                    (df["canal"] == canal) &
                    (df["condicion"] == condicion) &
                    (df["grupo"] == grupo)
                ]["latencia_ms"].dropna().values
                datos_plot.append(vals)
                n     = len(vals)
                media = np.mean(vals)
                sd    = np.std(vals, ddof=1)
                etiquetas.append(f"{grupo.capitalize()}\nn={n}\n{media:.1f}+/-{sd:.1f}ms")
                colores_bp.append(colores[grupo])

            bp = ax.boxplot(datos_plot, patch_artist=True,
                           medianprops={"color": "black", "linewidth": 2})
            for patch, color in zip(bp["boxes"], colores_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            for i, (vals, color) in enumerate(zip(datos_plot, colores_bp)):
                x_jitter = np.random.normal(i + 1, 0.07, size=len(vals))
                ax.scatter(x_jitter, vals, alpha=0.4, color=color, s=12, zorder=3)

            res_fila = resultados[
                (resultados["canal"] == canal) &
                (resultados["condicion"] == condicion)
            ].iloc[0]
            q = res_fila["p_fdr_bh"]
            d = res_fila["cohen_d"]
            anotacion = f"q = {q:.4f}  |  d = {d:.2f}"
            color_tit = "red" if q < ALPHA else "gray"

            ax.set_title(f"Canal: {canal}\n{condicion}\n{anotacion}",
                        fontsize=9, color=color_tit)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(etiquetas, fontsize=8)
            ax.set_ylabel("Latencia (ms)")
            ax.axhline(240, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_latencia_boxplot_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_latencia_boxplot_v2.png'")


def graficar_barras(resultados):
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5),
                             sharey=True)
    fig.suptitle(
        "Latencia media del c240/VMP +/- SD\nControl vs Alcoholico",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    x      = np.arange(len(CANALES_INTERES))
    ancho  = 0.35

    for ax, condicion in zip(axes, CONDICIONES):
        sub = resultados[resultados["condicion"] == condicion]
        for i, grupo in enumerate(["control", "alcoholic"]):
            medias = [sub[sub["canal"]==c][f"media_lat_{grupo}"].values[0] for c in CANALES_INTERES]
            sds    = [sub[sub["canal"]==c][f"sd_lat_{grupo}"].values[0]    for c in CANALES_INTERES]
            ax.bar(x + i * ancho, medias, ancho, yerr=sds, capsize=4,
                  label=grupo.capitalize(), color=colores[grupo], alpha=0.75)

        for j, canal in enumerate(CANALES_INTERES):
            q = sub[sub["canal"] == canal]["p_fdr_bh"].values[0]
            if q < ALPHA:
                ax.text(j + ancho/2, 265, "*", ha="center", fontsize=14)

        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condicion: {condicion}", fontsize=11)
        ax.set_ylabel("Latencia (ms)")
        ax.set_ylim([215, 275])
        ax.axhline(240, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_latencia_barras_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_latencia_barras_v2.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 07: Analisis de Latencia c240/VMP")
    print("=" * 60)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 05 v2."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_csv(ENTRADA)
    print(f"  {df['sujeto'].nunique()} sujetos cargados")

    diagnostico_bordes(df)

    print("\nCalculando estadisticas de latencia...")
    resultados = calcular_estadisticas_latencia(df)
    imprimir_tabla(resultados)

    resultados.to_csv(SALIDA, index=False)
    print(f"\nTabla guardada en '{SALIDA}'")

    print("\nGenerando graficos...")
    np.random.seed(42)
    graficar_boxplot(df, resultados)
    graficar_barras(resultados)

    print("\n[OK] Script 07 finalizado.")
