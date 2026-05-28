"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 05: Extracción del Componente c240/VMP
==============================================================================

Propósito:
    Para cada sujeto, extraer un único valor numérico que represente
    el componente c240: la amplitud pico dentro de la ventana 220-260 ms
    y la latencia (en qué ms exacto ocurre ese pico).

    Estos valores son los que se compararán estadísticamente en el Script 06.

Pasos:
    1. Para cada sujeto × canal × condición, tomar su PE individual
    2. Dentro de la ventana 220-260 ms, encontrar el valor máximo (pico)
    3. Registrar la amplitud y la latencia de ese pico
    4. Generar visualizaciones de la distribución de amplitudes por grupo

Entrada:  outputs/eeg_PE_individual.parquet  (generado por Script 04)
Salida:   outputs/eeg_c240_extraido.csv      (un valor por sujeto×canal×condición)
          outputs/figura_c240_boxplot.png
          outputs/figura_c240_latencias.png

Uso:
    Correr desde la carpeta scripts/
    python 05_extraccion_c240.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS = 256  # Hz

# Ventana del componente c240/VMP (ms post-estímulo)
T_C240_INI_MS = 220
T_C240_FIN_MS = 260

# Convertir ventana a muestras
M_C240_INI = int(T_C240_INI_MS / 1000 * FS)  # = 56
M_C240_FIN = int(T_C240_FIN_MS / 1000 * FS)  # = 66

CANALES_INTERES  = ["P8", "PO8", "T8", "TP8"]
CONDICIONES      = ["S1 obj", "S2 nomatch"]

ENTRADA  = Path("../outputs/eeg_PE_individual.parquet")
SALIDA   = Path("../outputs/eeg_c240_extraido.csv")

# =============================================================================
# FUNCIONES
# =============================================================================

def extraer_c240_sujeto(df_sujeto: pd.DataFrame) -> dict:
    """
    Extrae la amplitud y latencia del componente c240 para un sujeto,
    canal y condición dados.

    Busca el valor MÁXIMO dentro de la ventana 220-260 ms del PE
    individual de ese sujeto. Usamos el máximo (pico positivo) porque
    el c240/VMP es un componente de polaridad positiva.

    Args:
        df_sujeto: DataFrame con el PE individual de un sujeto×canal×condición
                   columnas: muestra, PE_uV, n_trials

    Retorna:
        dict con:
            amplitud_uV: valor máximo en la ventana (µV)
            latencia_ms: tiempo en ms donde ocurre ese máximo
            n_trials:    cantidad de trials que se promediaron
    """
    ventana = df_sujeto[
        (df_sujeto["muestra"] >= M_C240_INI) &
        (df_sujeto["muestra"] <= M_C240_FIN)
    ]

    if ventana.empty:
        return {"amplitud_uV": np.nan, "latencia_ms": np.nan, "n_trials": 0}

    idx_max      = ventana["PE_uV"].idxmax()
    amplitud     = ventana.loc[idx_max, "PE_uV"]
    muestra_pico = ventana.loc[idx_max, "muestra"]
    latencia_ms  = muestra_pico / FS * 1000
    n_trials     = ventana["n_trials"].iloc[0]

    return {
        "amplitud_uV": amplitud,
        "latencia_ms": latencia_ms,
        "n_trials":    n_trials
    }


def extraer_todos(erp_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica extraer_c240_sujeto() a cada combinación sujeto×canal×condición.

    Retorna un DataFrame con una fila por combinación y columnas:
        sujeto, grupo, canal, condicion, amplitud_uV, latencia_ms, n_trials
    """
    resultados = []
    grupos = erp_ind.groupby(["sujeto", "grupo", "canal", "condicion"])
    n_total = len(grupos)

    print(f"  Extrayendo c240 de {n_total} combinaciones sujeto×canal×condición...")

    for (sujeto, grupo, canal, condicion), df_sub in grupos:
        vals = extraer_c240_sujeto(df_sub.sort_values("muestra"))
        resultados.append({
            "sujeto":       sujeto,
            "grupo":        grupo,
            "canal":        canal,
            "condicion":    condicion,
            "amplitud_uV":  vals["amplitud_uV"],
            "latencia_ms":  vals["latencia_ms"],
            "n_trials":     vals["n_trials"]
        })

    return pd.DataFrame(resultados)


def resumen_c240(df_c240: pd.DataFrame):
    """
    Imprime tabla resumen de amplitud y latencia media del c240
    por grupo, canal y condición.
    """
    print("\n" + "=" * 70)
    print("RESUMEN DEL COMPONENTE c240/VMP — Ventana 220–260 ms")
    print("=" * 70)

    for condicion in CONDICIONES:
        print(f"\nCondición: {condicion}")
        print(f"  {'Canal':<6} {'Grupo':<12} {'N':>4} "
              f"{'Amplitud media':>16} {'±SD':>8} {'Latencia media':>16}")
        print(f"  {'-'*6} {'-'*12} {'-'*4} {'-'*16} {'-'*8} {'-'*16}")

        for canal in CANALES_INTERES:
            for grupo in ["control", "alcoholic"]:
                sub = df_c240[
                    (df_c240["canal"] == canal) &
                    (df_c240["condicion"] == condicion) &
                    (df_c240["grupo"] == grupo)
                ]["amplitud_uV"].dropna()

                lat = df_c240[
                    (df_c240["canal"] == canal) &
                    (df_c240["condicion"] == condicion) &
                    (df_c240["grupo"] == grupo)
                ]["latencia_ms"].dropna()

                print(f"  {canal:<6} {grupo:<12} {len(sub):>4} "
                      f"{sub.mean():>14.3f} µV "
                      f"{sub.std():>6.3f} "
                      f"{lat.mean():>14.1f} ms")

    print("=" * 70)


def graficar_boxplot(df_c240: pd.DataFrame):
    """
    Grafica boxplots de la amplitud del c240 por grupo,
    para cada canal y condición.

    El boxplot muestra:
        - Línea central: mediana
        - Caja: rango intercuartil (IQR, percentiles 25–75)
        - Bigotes: 1.5 × IQR
        - Puntos: valores atípicos (outliers)

    Permite comparar visualmente la distribución de amplitudes
    entre control y alcohólico antes del análisis estadístico.
    """
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharey=False
    )

    fig.suptitle(
        "Amplitud del componente c240/VMP por sujeto\n"
        "Ventana 220–260 ms post-estímulo",
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
                vals = df_c240[
                    (df_c240["canal"] == canal) &
                    (df_c240["condicion"] == condicion) &
                    (df_c240["grupo"] == grupo)
                ]["amplitud_uV"].dropna().values

                datos_plot.append(vals)
                n = len(vals)
                etiquetas.append(f"{grupo.capitalize()}\n(n={n})")
                colores_bp.append(colores[grupo])

            bp = ax.boxplot(datos_plot, patch_artist=True,
                           medianprops={"color": "black", "linewidth": 2},
                           whiskerprops={"linewidth": 1.2},
                           capprops={"linewidth": 1.2},
                           flierprops={"marker": "o", "markersize": 4,
                                      "alpha": 0.5})

            for patch, color in zip(bp["boxes"], colores_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Superponer puntos individuales (jitter)
            for i, (vals, color) in enumerate(zip(datos_plot, colores_bp)):
                x_jitter = np.random.normal(i + 1, 0.06, size=len(vals))
                ax.scatter(x_jitter, vals, alpha=0.4, color=color,
                          s=15, zorder=3)

            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(etiquetas, fontsize=9)
            ax.set_title(f"Canal: {canal}\n{condicion}", fontsize=10)
            ax.set_ylabel("Amplitud pico c240 (µV)")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_c240_boxplot.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_c240_boxplot.png'")


def graficar_latencias(df_c240: pd.DataFrame):
    """
    Grafica la distribución de latencias del pico c240 por grupo.

    La latencia es en qué milisegundo exacto ocurre el pico dentro
    de la ventana 220-260 ms. Si el componente es robusto, debería
    concentrarse alrededor de los 240 ms en ambos grupos.
    """
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharey=False, sharex=True
    )

    fig.suptitle(
        "Distribución de latencias del pico c240 por grupo\n"
        "Ventana 220–260 ms post-estímulo",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    bins = np.linspace(T_C240_INI_MS, T_C240_FIN_MS, 12)

    for fila, condicion in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            for grupo in ["control", "alcoholic"]:
                lats = df_c240[
                    (df_c240["canal"] == canal) &
                    (df_c240["condicion"] == condicion) &
                    (df_c240["grupo"] == grupo)
                ]["latencia_ms"].dropna().values

                ax.hist(lats, bins=bins, alpha=0.5,
                       color=colores[grupo],
                       label=grupo.capitalize(),
                       density=True)

            ax.axvline(240, color="orange", linestyle="--",
                      linewidth=1.2, label="240 ms")
            ax.set_title(f"Canal: {canal}\n{condicion}", fontsize=10)
            ax.set_xlabel("Latencia (ms)")
            ax.set_ylabel("Densidad")
            ax.grid(True, alpha=0.3)

            if fila == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("../outputs/figura_c240_latencias.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_c240_latencias.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 05: Extracción del Componente c240/VMP")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Cargar PEs individuales
    # -------------------------------------------------------------------------
    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 04."
        )

    print(f"\nCargando '{ENTRADA}'...")
    erp_ind = pd.read_parquet(ENTRADA)
    print(f"  {erp_ind['sujeto'].nunique()} sujetos cargados")
    print(f"  Ventana c240: muestras {M_C240_INI}–{M_C240_FIN} "
          f"({T_C240_INI_MS}–{T_C240_FIN_MS} ms)")

    # -------------------------------------------------------------------------
    # Extracción del c240
    # -------------------------------------------------------------------------
    print("\nExtrayendo componente c240...")
    df_c240 = extraer_todos(erp_ind)

    # -------------------------------------------------------------------------
    # Resumen
    # -------------------------------------------------------------------------
    resumen_c240(df_c240)

    # -------------------------------------------------------------------------
    # Guardar
    # -------------------------------------------------------------------------
    df_c240.to_csv(SALIDA, index=False)
    print(f"\nDatos guardados en '{SALIDA}'")
    print(f"  {len(df_c240)} filas "
          f"({df_c240['sujeto'].nunique()} sujetos × "
          f"{df_c240['canal'].nunique()} canales × "
          f"{df_c240['condicion'].nunique()} condiciones)")

    # -------------------------------------------------------------------------
    # Gráficos
    # -------------------------------------------------------------------------
    print("\nGenerando gráficos...")
    np.random.seed(42)
    graficar_boxplot(df_c240)
    graficar_latencias(df_c240)

    print("\n[OK] Script 05 finalizado.")
