"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 06 v2: Comparación Estadística entre Grupos
==============================================================================

CAMBIOS RESPECTO DE v1
----------------------
1. Variable de comparación: ahora se usa amplitud_abs_uV (magnitud), no
   amplitud_uV con signo. Esto hace el análisis robusto a inversiones
   de polaridad por la referencia, manteniendo el criterio consistente
   para todos los sujetos (anteproyecto).

2. Se agrega corrección por comparaciones múltiples Bonferroni y Benjamini-
   Hochberg (FDR), porque se realizan 4 canales × 2 condiciones = 8 tests.

3. Se agrega tamaño del efecto (Cohen's d con corrección de Welch) para
   complementar el p-valor con una medida de magnitud práctica.

4. Se corre también sobre el CSV de sensibilidad (ventana 200–280 ms),
   generando una tabla paralela que permite verificar robustez.

Entrada:  outputs/eeg_c240_extraido_v2.csv       (Script 05 v2, ventana 220–260)
          outputs/eeg_c240_extraido_v2_sens.csv  (Script 05 v2, ventana 200–280)
Salida:   outputs/tabla_estadistica_v2.csv
          outputs/tabla_estadistica_v2_sens.csv
          outputs/figura_estadistica_comparacion_v2.png
          outputs/figura_estadistica_barras_v2.png

Uso:
    python 06_estadistica_v2.py
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
N_TESTS         = len(CANALES_INTERES) * len(CONDICIONES)  # = 8

ENTRADA       = Path("../outputs/eeg_c240_extraido_v2.csv")
ENTRADA_SENS  = Path("../outputs/eeg_c240_extraido_v2_sens.csv")
SALIDA        = Path("../outputs/tabla_estadistica_v2.csv")
SALIDA_SENS   = Path("../outputs/tabla_estadistica_v2_sens.csv")

# =============================================================================
# FUNCIONES
# =============================================================================

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Tamaño del efecto Cohen's d para dos grupos independientes con
    varianzas posiblemente desiguales (pooled SD).

    Interpretación canónica (Cohen, 1988):
        |d| ≈ 0.2  → efecto pequeño
        |d| ≈ 0.5  → efecto medio
        |d| ≈ 0.8  → efecto grande
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if s_pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / s_pooled


def correccion_bh(p_valores: np.ndarray) -> np.ndarray:
    """
    Corrección de Benjamini-Hochberg para control de la False Discovery
    Rate (FDR). Devuelve los p-valores ajustados (q-values).

    Más permisiva que Bonferroni: controla la proporción esperada de
    falsos positivos entre los rechazos, en lugar de la probabilidad
    de cometer al menos un falso positivo.
    """
    p = np.asarray(p_valores, dtype=float)
    n = len(p)
    orden = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[orden] = np.arange(1, n + 1)

    # p_adj[i] = min_{k >= rank_i} ( p_sorted[k] * n / k )
    p_sorted = p[orden]
    q_sorted = p_sorted * n / np.arange(1, n + 1)
    # Asegurar monotonía (de derecha a izquierda)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q = np.empty(n)
    q[orden] = q_sorted
    return q


def calcular_estadisticas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compara la MAGNITUD absoluta del c240 entre control y alcoholic,
    para cada canal × condición.

    Métricas reportadas:
    - Media ± SD de |amplitud| por grupo (variable comparada)
    - Media ± SD de la amplitud con signo (para inspección)
    - t-estadístico y p-valor del test de Welch (varianzas desiguales)
    - p_bonferroni : p × N_TESTS (corte conservador)
    - p_fdr        : Benjamini-Hochberg
    - cohen_d      : tamaño del efecto
    - signif_*     : flags Sí/No para cada criterio

    Hipótesis del anteproyecto:
        H1: |A_c240| en alcoholic < |A_c240| en control,
            tanto para S1 obj como para S2 nomatch.

    El test usado es bilateral (ttest_ind). Reportamos los signos a
    través del estadístico t y de las medias, para verificar que la
    diferencia siga la dirección hipotetizada.
    """
    filas = []

    for condicion in CONDICIONES:
        for canal in CANALES_INTERES:

            sel = (
                (df["canal"] == canal) &
                (df["condicion"] == condicion)
            )

            ctrl_abs = df[sel & (df["grupo"] == "control")
                          ]["amplitud_abs_uV"].dropna().values
            alc_abs  = df[sel & (df["grupo"] == "alcoholic")
                          ]["amplitud_abs_uV"].dropna().values

            ctrl_sig = df[sel & (df["grupo"] == "control")
                          ]["amplitud_uV"].dropna().values
            alc_sig  = df[sel & (df["grupo"] == "alcoholic")
                          ]["amplitud_uV"].dropna().values

            t_stat, p_valor = ttest_ind(ctrl_abs, alc_abs, equal_var=False)
            d = cohen_d(ctrl_abs, alc_abs)

            filas.append({
                "canal":           canal,
                "condicion":       condicion,
                "n_control":       len(ctrl_abs),
                "n_alcoholic":     len(alc_abs),
                # Variable de comparación (magnitud)
                "media_abs_control":   float(np.mean(ctrl_abs)),
                "sd_abs_control":      float(np.std(ctrl_abs, ddof=1)),
                "media_abs_alcoholic": float(np.mean(alc_abs)),
                "sd_abs_alcoholic":    float(np.std(alc_abs, ddof=1)),
                # Inspección: amplitud con signo
                "media_signed_control":   float(np.mean(ctrl_sig)),
                "media_signed_alcoholic": float(np.mean(alc_sig)),
                # Test
                "t_estadistico":   float(t_stat),
                "p_valor":         float(p_valor),
                "cohen_d":         float(d),
                "direccion_ok":    "Si" if np.mean(alc_abs) < np.mean(ctrl_abs) else "No",
            })

    resultados = pd.DataFrame(filas)

    # Correcciones por comparaciones múltiples
    p = resultados["p_valor"].values
    resultados["p_bonferroni"] = np.clip(p * N_TESTS, 0, 1)
    resultados["p_fdr_bh"]     = correccion_bh(p)

    resultados["sig_alpha"]       = (resultados["p_valor"]      < ALPHA).map({True:"Si", False:"No"})
    resultados["sig_bonferroni"]  = (resultados["p_bonferroni"] < ALPHA).map({True:"Si", False:"No"})
    resultados["sig_fdr"]         = (resultados["p_fdr_bh"]     < ALPHA).map({True:"Si", False:"No"})

    return resultados


def imprimir_tabla(resultados: pd.DataFrame, etiqueta: str = ""):
    print("\n" + "=" * 95)
    print(f"TABLA DE RESULTADOS — |Amplitud c240/VMP| (media ± SD) + t-test {etiqueta}")
    print("=" * 95)

    for condicion in CONDICIONES:
        print(f"\nCondicion: {condicion}")
        print(f"  {'Canal':<5} "
              f"{'|Ctrl|':>16} {'|Alc|':>16} "
              f"{'t':>7} {'p':>9} {'p_BH':>8} {'p_Bonf':>9} "
              f"{'d':>6} {'dir':>4}")
        print(f"  {'-'*5} {'-'*16} {'-'*16} {'-'*7} {'-'*9} {'-'*8} {'-'*9} "
              f"{'-'*6} {'-'*4}")

        sub = resultados[resultados["condicion"] == condicion]
        for _, fila in sub.iterrows():
            ctrl_str = f"{fila['media_abs_control']:.3f}±{fila['sd_abs_control']:.3f}"
            alc_str  = f"{fila['media_abs_alcoholic']:.3f}±{fila['sd_abs_alcoholic']:.3f}"
            marca = "*" if fila["p_fdr_bh"] < ALPHA else " "
            print(f"  {fila['canal']:<5} {ctrl_str:>16} {alc_str:>16} "
                  f"{fila['t_estadistico']:>7.3f} "
                  f"{fila['p_valor']:>9.4f} "
                  f"{fila['p_fdr_bh']:>8.4f} "
                  f"{fila['p_bonferroni']:>9.4f} "
                  f"{fila['cohen_d']:>6.2f} "
                  f"{fila['direccion_ok']:>4} {marca}")

    print(f"\n  N_tests = {N_TESTS}  |  * = significativo tras FDR (BH) con alpha={ALPHA}")
    print(f"  'dir' = Si la magnitud media en alcoholic es menor que en control "
          f"(hipotesis del anteproyecto)")
    print("=" * 95)


def graficar_comparacion(df: pd.DataFrame, resultados: pd.DataFrame):
    """Boxplots de la MAGNITUD c240 con anotación de p y q-values."""
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 5 * len(CONDICIONES)),
        sharey=False
    )

    fig.suptitle(
        "Comparacion de |amplitud c240/VMP| entre grupos\n"
        "Control vs Alcoholico — Ventana 220-260 ms (pico por magnitud absoluta)",
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
                ]["amplitud_abs_uV"].dropna().values

                datos_plot.append(vals)
                n     = len(vals)
                media = np.mean(vals)
                sd    = np.std(vals, ddof=1)
                etiquetas.append(
                    f"{grupo.capitalize()}\n"
                    f"n={n}\n"
                    f"{media:.2f} ± {sd:.2f} uV"
                )
                colores_bp.append(colores[grupo])

            bp = ax.boxplot(
                datos_plot, patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.4}
            )
            for patch, color in zip(bp["boxes"], colores_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            for i, (vals, color) in enumerate(zip(datos_plot, colores_bp)):
                x_jitter = np.random.normal(i + 1, 0.07, size=len(vals))
                ax.scatter(x_jitter, vals, alpha=0.35,
                          color=color, s=12, zorder=3)

            ax.axhline(0, color="black", linewidth=0.6,
                      linestyle="--", alpha=0.5)

            res_fila = resultados[
                (resultados["canal"] == canal) &
                (resultados["condicion"] == condicion)
            ].iloc[0]

            p  = res_fila["p_valor"]
            q  = res_fila["p_fdr_bh"]
            d  = res_fila["cohen_d"]
            p_str = f"p = {p:.4f}" if p >= 1e-4 else "p < 0.0001"
            q_str = f"q = {q:.4f}" if q >= 1e-4 else "q < 0.0001"
            marca = " *" if q < ALPHA else ""
            anotacion = f"{p_str}  |  {q_str}{marca}  |  d = {d:.2f}"

            ax.set_title(f"Canal: {canal}\n{condicion}\n{anotacion}",
                fontsize=9,
                color="red" if q < ALPHA else "gray")

            ax.set_xticks([1, 2])
            ax.set_xticklabels(etiquetas, fontsize=8)
            ax.set_ylabel("|Amplitud pico c240| (uV)")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_estadistica_comparacion_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_estadistica_comparacion_v2.png'")


def graficar_barras(resultados: pd.DataFrame):
    """Barras de media ± SD por grupo y canal."""
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5),
                             sharey=True)

    fig.suptitle(
        "Magnitud media del componente c240/VMP ± SD\n"
        "Control vs Alcoholico (q = FDR Benjamini-Hochberg)",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    x      = np.arange(len(CANALES_INTERES))
    ancho  = 0.35

    for ax, condicion in zip(axes, CONDICIONES):
        sub = resultados[resultados["condicion"] == condicion]

        for i, grupo in enumerate(["control", "alcoholic"]):
            medias = [
                sub[sub["canal"] == c][f"media_abs_{grupo}"].values[0]
                for c in CANALES_INTERES
            ]
            sds = [
                sub[sub["canal"] == c][f"sd_abs_{grupo}"].values[0]
                for c in CANALES_INTERES
            ]
            ax.bar(x + i * ancho, medias, ancho,
                  yerr=sds, capsize=4,
                  label=grupo.capitalize(),
                  color=colores[grupo], alpha=0.75,
                  error_kw={"linewidth": 1.2})

        # Marcas de significancia (basadas en FDR)
        for j, canal in enumerate(CANALES_INTERES):
            q = sub[sub["canal"] == canal]["p_fdr_bh"].values[0]
            if q < ALPHA:
                y_ctrl  = sub[sub["canal"] == canal]["media_abs_control"].values[0]
                y_alc   = sub[sub["canal"] == canal]["media_abs_alcoholic"].values[0]
                sd_ctrl = sub[sub["canal"] == canal]["sd_abs_control"].values[0]
                y_pos = max(y_ctrl, y_alc) + sd_ctrl + 0.5
                ax.text(j + ancho / 2, y_pos, "*",
                       ha="center", fontsize=14, color="black")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condicion: {condicion}", fontsize=11)
        ax.set_ylabel("|Amplitud media c240| (uV)")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.text(0.98, 0.02, f"* q < {ALPHA} (FDR-BH)",
               transform=ax.transAxes,
               ha="right", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig("../outputs/figura_estadistica_barras_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_estadistica_barras_v2.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 06 v2: Comparacion Estadistica (magnitud + FDR + d)")
    print("=" * 60)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 05 v2."
        )

    # -------------------------------------------------------------------------
    # ANÁLISIS PRINCIPAL — ventana 220–260 ms
    # -------------------------------------------------------------------------
    print(f"\nCargando '{ENTRADA}' (ventana 220–260 ms)...")
    df = pd.read_csv(ENTRADA)
    print(f"  {df['sujeto'].nunique()} sujetos cargados")

    print("\nCalculando estadisticas (Welch t-test + Bonferroni + FDR + Cohen's d)...")
    resultados = calcular_estadisticas(df)

    imprimir_tabla(resultados, etiqueta="(VENTANA PRINCIPAL 220–260 ms)")
    resultados.to_csv(SALIDA, index=False)
    print(f"\nTabla guardada en '{SALIDA}'")

    print("\nGenerando graficos...")
    np.random.seed(42)
    graficar_comparacion(df, resultados)
    graficar_barras(resultados)

    # -------------------------------------------------------------------------
    # ANÁLISIS DE SENSIBILIDAD — ventana 200–280 ms
    # -------------------------------------------------------------------------
    if ENTRADA_SENS.exists():
        print(f"\nCargando '{ENTRADA_SENS}' (ventana 200–280 ms)...")
        df_sens = pd.read_csv(ENTRADA_SENS)
        resultados_sens = calcular_estadisticas(df_sens)
        imprimir_tabla(resultados_sens, etiqueta="(SENSIBILIDAD 200–280 ms)")
        resultados_sens.to_csv(SALIDA_SENS, index=False)
        print(f"Tabla guardada en '{SALIDA_SENS}'")
    else:
        print(f"\n[!] {ENTRADA_SENS} no encontrado — se omite analisis de sensibilidad.")

    print("\n[OK] Script 06 v2 finalizado.")
