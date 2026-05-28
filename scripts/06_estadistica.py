"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 06: Comparación Estadística entre Grupos
==============================================================================

Propósito:
    Comparar la amplitud del componente c240/VMP entre el grupo control
    y el grupo alcohólico, para cada canal y condición de interés.

Análisis realizados:
    1. Tabla de media ± desvío estándar por grupo
    2. T-test de Student para comparación entre grupos
    3. Gráficos de distribución (boxplots + barras con error)

Entrada:  outputs/eeg_c240_extraido.csv  (generado por Script 05)
Salida:   outputs/tabla_estadistica.csv
          outputs/figura_estadistica_comparacion.png
          outputs/figura_estadistica_barras.png

Uso:
    Correr desde la carpeta scripts/
    python 06_estadistica.py
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

ENTRADA = Path("../outputs/eeg_c240_extraido.csv")
SALIDA  = Path("../outputs/tabla_estadistica.csv")

# =============================================================================
# FUNCIONES
# =============================================================================

def calcular_estadisticas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula media y desvío estándar de la amplitud c240 por grupo,
    canal y condición. También aplica el T-test de Student entre grupos.

    El T-test de Student compara las medias de dos grupos independientes.
    H0: las medias de ambos grupos son iguales.
    Si p < 0.05 -> rechazamos H0 -> la diferencia entre grupos es
    estadísticamente significativa.

    Usamos equal_var=False (versión de Welch) porque los grupos tienen
    distinto tamaño (n=45 vs n=77) y no podemos asumir varianzas iguales.
    """
    resultados = []

    for condicion in CONDICIONES:
        for canal in CANALES_INTERES:

            ctrl = df[
                (df["canal"] == canal) &
                (df["condicion"] == condicion) &
                (df["grupo"] == "control")
            ]["amplitud_uV"].dropna().values

            alc = df[
                (df["canal"] == canal) &
                (df["condicion"] == condicion) &
                (df["grupo"] == "alcoholic")
            ]["amplitud_uV"].dropna().values

            t_stat, p_valor = ttest_ind(ctrl, alc, equal_var=False)

            resultados.append({
                "canal":           canal,
                "condicion":       condicion,
                "n_control":       len(ctrl),
                "media_control":   np.mean(ctrl),
                "sd_control":      np.std(ctrl, ddof=1),
                "n_alcoholic":     len(alc),
                "media_alcoholic": np.mean(alc),
                "sd_alcoholic":    np.std(alc, ddof=1),
                "t_estadistico":   t_stat,
                "p_valor":         p_valor,
                "significativo":   "Si" if p_valor < ALPHA else "No"
            })

    return pd.DataFrame(resultados)


def imprimir_tabla(resultados: pd.DataFrame):
    """Imprime la tabla resumen de resultados estadísticos."""
    print("\n" + "=" * 75)
    print("TABLA DE RESULTADOS — Amplitud c240/VMP (media +/- SD) y T-test")
    print("=" * 75)

    for condicion in CONDICIONES:
        print(f"\nCondicion: {condicion}")
        print(f"  {'Canal':<5} {'Control (uV)':>18} {'Alcoholic (uV)':>18} "
              f"{'t':>7} {'p-valor':>9} {'Sig':>5}")
        print(f"  {'-'*5} {'-'*18} {'-'*18} {'-'*7} {'-'*9} {'-'*5}")

        sub = resultados[resultados["condicion"] == condicion]
        for _, fila in sub.iterrows():
            ctrl_str = f"{fila['media_control']:+.3f} +/- {fila['sd_control']:.3f}"
            alc_str  = f"{fila['media_alcoholic']:+.3f} +/- {fila['sd_alcoholic']:.3f}"
            sig = "*" if fila["p_valor"] < ALPHA else " "
            print(f"  {fila['canal']:<5} {ctrl_str:>18} {alc_str:>18} "
                  f"{fila['t_estadistico']:>7.3f} "
                  f"{fila['p_valor']:>9.4f} {sig:>5}")

    print(f"\n  * = p < {ALPHA} (diferencia estadisticamente significativa)")
    print("=" * 75)


def graficar_comparacion(df: pd.DataFrame, resultados: pd.DataFrame):
    """
    Boxplots con puntos individuales y p-valor anotado.
    Cubre el requisito de graficos de distribucion del anteproyecto.
    """
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 5 * len(CONDICIONES)),
        sharey=False
    )

    fig.suptitle(
        "Comparacion de amplitud c240/VMP entre grupos\n"
        "Control vs Alcoholico — Ventana 220-260 ms post-estimulo",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for fila, condicion in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            datos_plot = []
            etiquetas  = []
            colores_bp = []

            for grupo in ["control", "alcoholic"]:
                vals = df[
                    (df["canal"] == canal) &
                    (df["condicion"] == condicion) &
                    (df["grupo"] == grupo)
                ]["amplitud_uV"].dropna().values

                datos_plot.append(vals)
                n     = len(vals)
                media = np.mean(vals)
                sd    = np.std(vals, ddof=1)
                etiquetas.append(
                    f"{grupo.capitalize()}\n"
                    f"n={n}\n"
                    f"{media:+.2f} +/- {sd:.2f} uV"
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

            p = res_fila["p_valor"]
            sig_texto = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"
            if p < ALPHA:
                sig_texto += " *"

            ax.set_title(f"Canal: {canal}\n{condicion}\n{sig_texto}",
            fontsize=9,
            color="red" if p < ALPHA else "gray")

            ax.set_xticks([1, 2])
            ax.set_xticklabels(etiquetas, fontsize=8)
            ax.set_ylabel("Amplitud pico c240 (uV)")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_estadistica_comparacion.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_estadistica_comparacion.png'")


def graficar_barras(resultados: pd.DataFrame):
    """
    Grafico de barras con media +/- SD por grupo, canal y condicion.
    """
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5))

    fig.suptitle(
        "Amplitud media del componente c240/VMP +/- SD\n"
        "Control vs Alcoholico",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    x = np.arange(len(CANALES_INTERES))
    ancho = 0.35

    for ax, condicion in zip(axes, CONDICIONES):
        for i, grupo in enumerate(["control", "alcoholic"]):
            sub = resultados[resultados["condicion"] == condicion]
            medias = [
                sub[sub["canal"] == c][f"media_{grupo}"].values[0]
                for c in CANALES_INTERES
            ]
            sds = [
                sub[sub["canal"] == c][f"sd_{grupo}"].values[0]
                for c in CANALES_INTERES
            ]
            ax.bar(x + i * ancho, medias, ancho,
                  yerr=sds, capsize=4,
                  label=grupo.capitalize(),
                  color=colores[grupo], alpha=0.75,
                  error_kw={"linewidth": 1.2})

        sub = resultados[resultados["condicion"] == condicion]
        for j, canal in enumerate(CANALES_INTERES):
            p = sub[sub["canal"] == canal]["p_valor"].values[0]
            if p < ALPHA:
                y_ctrl = sub[sub["canal"] == canal]["media_control"].values[0]
                y_alc  = sub[sub["canal"] == canal]["media_alcoholic"].values[0]
                sd_ctrl = sub[sub["canal"] == canal]["sd_control"].values[0]
                y_pos = max(y_ctrl, y_alc) + sd_ctrl + 0.5
                ax.text(j + ancho / 2, y_pos, "*",
                       ha="center", fontsize=14, color="black")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condicion: {condicion}", fontsize=11)
        ax.set_ylabel("Amplitud media c240 (uV)")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.text(0.98, 0.02, "* p < 0.05",
               transform=ax.transAxes,
               ha="right", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig("../outputs/figura_estadistica_barras.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_estadistica_barras.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 06: Comparacion Estadistica entre Grupos")
    print("=" * 60)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 05."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_csv(ENTRADA)
    print(f"  {df['sujeto'].nunique()} sujetos cargados")

    print("\nCalculando estadisticas (media, SD, T-test)...")
    resultados = calcular_estadisticas(df)

    imprimir_tabla(resultados)

    resultados.to_csv(SALIDA, index=False)
    print(f"\nTabla guardada en '{SALIDA}'")

    print("\nGenerando graficos...")
    np.random.seed(42)
    graficar_comparacion(df, resultados)
    graficar_barras(resultados)

    print("\n[OK] Script 06 finalizado.")
