"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 05 v2: Extracción del Componente c240/VMP
==============================================================================

CAMBIOS RESPECTO DE v1
----------------------
v1 buscaba el MÁXIMO POSITIVO dentro de la ventana 220-260 ms.
Esto introducía un sesgo metodológico: en muchos sujetos alcohólicos
la señal en esa ventana es globalmente negativa (polaridad invertida
por la referencia o ausencia del componente), y v1 devolvía como "pico"
el valor menos negativo, que típicamente caía en el borde superior de
la ventana (latencia ≈ 258 ms). Eso no es un pico real.

v2 implementa lo que pide el anteproyecto:
    "Si la polaridad del componente aparece invertida por la referencia
     utilizada, se tomará el pico correspondiente dentro de esa ventana
     temporal, manteniendo el mismo criterio para todos los sujetos."

Para todos los sujetos, en lugar de buscar el máximo positivo, buscamos
el pico de mayor MAGNITUD ABSOLUTA dentro de la ventana, preservando
el signo en la columna `amplitud_uV`. Para los análisis estadísticos se
usa la magnitud (|amplitud_uV|), que es la variable que refleja "cuán
evocado está el componente" independientemente del signo de la
referencia.

Se agregan dos columnas nuevas:
    - amplitud_abs_uV : magnitud del pico (siempre >= 0)
    - polaridad       : "+" o "-", para auditar la distribución por grupo

Análisis de sensibilidad
------------------------
Adicionalmente se corre la extracción con una ventana ampliada (200–280 ms)
para confirmar que los resultados no dependen del recorte estricto. Se
guarda en un CSV aparte.

Entrada:  outputs/eeg_PE_individual.parquet  (generado por Script 04)
Salida:   outputs/eeg_c240_extraido_v2.csv          (ventana 220–260 ms)
          outputs/eeg_c240_extraido_v2_sens.csv     (ventana 200–280 ms)
          outputs/figura_c240_boxplot_v2.png
          outputs/figura_c240_latencias_v2.png
          outputs/figura_c240_polaridad_v2.png

Uso:
    python 05_extraccion_c240_v2.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS = 256  # Hz

# Ventana principal del c240/VMP según anteproyecto
T_C240_INI_MS = 220
T_C240_FIN_MS = 260

# Ventana ampliada para análisis de sensibilidad
T_C240_INI_SENS_MS = 200
T_C240_FIN_SENS_MS = 280

CANALES_INTERES  = ["P8", "PO8", "T8", "TP8"]
CONDICIONES      = ["S1 obj", "S2 nomatch"]

ENTRADA      = Path("../outputs/eeg_PE_individual_v2.parquet")
SALIDA       = Path("../outputs/eeg_c240_extraido_v2.csv")
SALIDA_SENS  = Path("../outputs/eeg_c240_extraido_v2_sens.csv")


def ms_a_muestra(t_ms: float) -> int:
    """Convierte milisegundos a índice de muestra a FS=256 Hz."""
    return int(t_ms / 1000 * FS)


# =============================================================================
# FUNCIONES
# =============================================================================

def extraer_c240_sujeto(df_sujeto: pd.DataFrame,
                        m_ini: int, m_fin: int) -> dict:
    """
    Extrae el componente c240 para un sujeto×canal×condición.

    LÓGICA v2 — POLARIDAD AGNÓSTICA
    --------------------------------
    Dentro de la ventana temporal, busca el punto de mayor MAGNITUD
    ABSOLUTA y devuelve su valor con signo. Esto cubre los dos casos:

    - Si el componente aparece como pico positivo (típico en occipitales
      con referencia Cz), |valor_max_positivo| > |valor_min_negativo|,
      y el resultado es positivo.

    - Si la referencia invierte la polaridad y el componente aparece
      como pico negativo, |valor_min| > |valor_max|, y el resultado
      es negativo.

    Para los análisis estadísticos se compara la magnitud absoluta,
    que es invariante al signo de la referencia.

    Args:
        df_sujeto: DataFrame con columnas (muestra, PE_uV, n_trials)
        m_ini, m_fin: límites de la ventana en muestras (inclusive)

    Retorna:
        dict con amplitud_uV (con signo), amplitud_abs_uV, latencia_ms,
        polaridad, n_trials
    """
    ventana = df_sujeto[
        (df_sujeto["muestra"] >= m_ini) &
        (df_sujeto["muestra"] <= m_fin)
    ]

    if ventana.empty:
        return {
            "amplitud_uV":      np.nan,
            "amplitud_abs_uV":  np.nan,
            "latencia_ms":      np.nan,
            "polaridad":        "n/a",
            "n_trials":         0,
        }

    # Pico por magnitud absoluta, preservando el signo
    idx_pico     = ventana["PE_uV"].abs().idxmax()
    amplitud     = ventana.loc[idx_pico, "PE_uV"]
    muestra_pico = ventana.loc[idx_pico, "muestra"]
    latencia_ms  = muestra_pico / FS * 1000
    n_trials     = ventana["n_trials"].iloc[0]

    return {
        "amplitud_uV":      amplitud,
        "amplitud_abs_uV":  abs(amplitud),
        "latencia_ms":      latencia_ms,
        "polaridad":        "+" if amplitud >= 0 else "-",
        "n_trials":         n_trials,
    }


def extraer_todos(erp_ind: pd.DataFrame,
                  m_ini: int, m_fin: int,
                  etiqueta_ventana: str) -> pd.DataFrame:
    """
    Aplica extraer_c240_sujeto() a cada combinación sujeto×canal×condición
    para la ventana indicada.
    """
    resultados = []
    grupos     = erp_ind.groupby(["sujeto", "grupo", "canal", "condicion"])
    n_total    = len(grupos)

    print(f"  Extrayendo c240 [{etiqueta_ventana}] "
          f"de {n_total} combinaciones sujeto×canal×condición...")

    for (sujeto, grupo, canal, condicion), df_sub in grupos:
        vals = extraer_c240_sujeto(
            df_sub.sort_values("muestra"), m_ini, m_fin
        )
        resultados.append({
            "sujeto":           sujeto,
            "grupo":            grupo,
            "canal":            canal,
            "condicion":        condicion,
            "amplitud_uV":      vals["amplitud_uV"],
            "amplitud_abs_uV":  vals["amplitud_abs_uV"],
            "latencia_ms":      vals["latencia_ms"],
            "polaridad":        vals["polaridad"],
            "n_trials":         vals["n_trials"],
        })

    return pd.DataFrame(resultados)


def resumen_c240(df_c240: pd.DataFrame, etiqueta: str = ""):
    """
    Tabla resumen: media y SD de la AMPLITUD ABSOLUTA por grupo, canal y
    condición. También reporta la proporción de polaridad positiva por
    grupo (auditoría del sesgo de polaridad).
    """
    print("\n" + "=" * 78)
    print(f"RESUMEN c240/VMP — {etiqueta}")
    print("=" * 78)

    for condicion in CONDICIONES:
        print(f"\nCondición: {condicion}")
        print(f"  {'Canal':<6} {'Grupo':<11} {'N':>4} "
              f"{'|Amp| media':>14} {'±SD':>8} "
              f"{'Lat. media':>14} {'% pos':>7}")
        print(f"  {'-'*6} {'-'*11} {'-'*4} "
              f"{'-'*14} {'-'*8} {'-'*14} {'-'*7}")

        for canal in CANALES_INTERES:
            for grupo in ["control", "alcoholic"]:
                sub = df_c240[
                    (df_c240["canal"] == canal) &
                    (df_c240["condicion"] == condicion) &
                    (df_c240["grupo"] == grupo)
                ]

                amp = sub["amplitud_abs_uV"].dropna()
                lat = sub["latencia_ms"].dropna()
                pos = (sub["polaridad"] == "+").sum()
                pct_pos = 100 * pos / len(sub) if len(sub) else 0

                print(f"  {canal:<6} {grupo:<11} {len(amp):>4} "
                      f"{amp.mean():>12.3f} µV "
                      f"{amp.std():>6.3f} "
                      f"{lat.mean():>12.1f} ms "
                      f"{pct_pos:>6.1f}%")

    print("=" * 78)


def graficar_boxplot(df_c240: pd.DataFrame):
    """
    Boxplots de la MAGNITUD ABSOLUTA del c240, que es la variable que
    se compara estadísticamente entre grupos.
    """
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharey=False
    )

    fig.suptitle(
        "Magnitud del componente c240/VMP por sujeto\n"
        "Ventana 220–260 ms post-estímulo — |amplitud pico|",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for fila, condicion in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            datos_plot, etiquetas, colores_bp = [], [], []

            for grupo in ["control", "alcoholic"]:
                vals = df_c240[
                    (df_c240["canal"] == canal) &
                    (df_c240["condicion"] == condicion) &
                    (df_c240["grupo"] == grupo)
                ]["amplitud_abs_uV"].dropna().values

                datos_plot.append(vals)
                n = len(vals)
                etiquetas.append(f"{grupo.capitalize()}\n(n={n})")
                colores_bp.append(colores[grupo])

            bp = ax.boxplot(
                datos_plot, patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
            )
            for patch, color in zip(bp["boxes"], colores_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Jitter
            for i, (vals, color) in enumerate(zip(datos_plot, colores_bp)):
                x_jitter = np.random.normal(i + 1, 0.06, size=len(vals))
                ax.scatter(x_jitter, vals, alpha=0.4, color=color,
                          s=15, zorder=3)

            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(etiquetas, fontsize=9)
            ax.set_title(f"Canal: {canal}\n{condicion}", fontsize=10)
            ax.set_ylabel("|Amplitud pico c240| (µV)")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_c240_boxplot_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_c240_boxplot_v2.png'")


def graficar_latencias(df_c240: pd.DataFrame):
    """Distribución de latencias del pico c240 por grupo."""
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharey=False, sharex=True
    )

    fig.suptitle(
        "Distribución de latencias del pico c240 por grupo\n"
        "Ventana 220–260 ms post-estímulo (pico por magnitud absoluta)",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    bins    = np.linspace(T_C240_INI_MS, T_C240_FIN_MS, 12)

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
    plt.savefig("../outputs/figura_c240_latencias_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_c240_latencias_v2.png'")


def graficar_polaridad(df_c240: pd.DataFrame):
    """
    Gráfico de auditoría: proporción de picos positivos vs negativos
    por grupo, canal y condición. Justifica documentalmente la elección
    de magnitud absoluta como variable de comparación.
    """
    fig, axes = plt.subplots(
        1, len(CONDICIONES),
        figsize=(7 * len(CONDICIONES), 4.5),
        sharey=True
    )

    fig.suptitle(
        "Distribución de polaridad del pico c240 dentro de la ventana 220–260 ms\n"
        "Justifica el uso de magnitud absoluta como variable de comparación",
        fontsize=12
    )

    x      = np.arange(len(CANALES_INTERES))
    ancho  = 0.35

    for ax, condicion in zip(axes, CONDICIONES):
        for i, grupo in enumerate(["control", "alcoholic"]):
            pct_neg = []
            for canal in CANALES_INTERES:
                sub = df_c240[
                    (df_c240["canal"] == canal) &
                    (df_c240["condicion"] == condicion) &
                    (df_c240["grupo"] == grupo)
                ]
                if len(sub) == 0:
                    pct_neg.append(0)
                else:
                    pct_neg.append(100 * (sub["polaridad"] == "-").mean())

            color = "#2563eb" if grupo == "control" else "#dc2626"
            ax.bar(x + i * ancho, pct_neg, ancho,
                  label=grupo.capitalize(), color=color, alpha=0.75)

        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condición: {condicion}", fontsize=11)
        ax.set_ylabel("% de sujetos con pico negativo")
        ax.set_ylim([0, 100])
        ax.axhline(50, color="gray", linestyle=":", linewidth=0.8)
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_c240_polaridad_v2.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_c240_polaridad_v2.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 05 v2: Extracción c240/VMP (polaridad agnóstica)")
    print("=" * 60)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 04."
        )

    print(f"\nCargando '{ENTRADA}'...")
    erp_ind = pd.read_parquet(ENTRADA)
    print(f"  {erp_ind['sujeto'].nunique()} sujetos cargados")

    # -------------------------------------------------------------------------
    # Extracción principal: ventana 220–260 ms (anteproyecto)
    # -------------------------------------------------------------------------
    m_ini      = ms_a_muestra(T_C240_INI_MS)
    m_fin      = ms_a_muestra(T_C240_FIN_MS)
    print(f"\nVentana principal: muestras {m_ini}–{m_fin} "
          f"({T_C240_INI_MS}–{T_C240_FIN_MS} ms)")

    df_c240 = extraer_todos(erp_ind, m_ini, m_fin, "220–260 ms")
    resumen_c240(df_c240, etiqueta="Ventana principal 220–260 ms")
    df_c240.to_csv(SALIDA, index=False)
    print(f"\nDatos guardados en '{SALIDA}'")

    # -------------------------------------------------------------------------
    # Análisis de sensibilidad: ventana 200–280 ms
    # -------------------------------------------------------------------------
    m_ini_s   = ms_a_muestra(T_C240_INI_SENS_MS)
    m_fin_s   = ms_a_muestra(T_C240_FIN_SENS_MS)
    print(f"\n[Análisis de sensibilidad] Ventana ampliada: "
          f"muestras {m_ini_s}–{m_fin_s} "
          f"({T_C240_INI_SENS_MS}–{T_C240_FIN_SENS_MS} ms)")

    df_c240_sens = extraer_todos(erp_ind, m_ini_s, m_fin_s, "200–280 ms")
    resumen_c240(df_c240_sens, etiqueta="Sensibilidad 200–280 ms")
    df_c240_sens.to_csv(SALIDA_SENS, index=False)
    print(f"\nDatos guardados en '{SALIDA_SENS}'")

    # -------------------------------------------------------------------------
    # Gráficos
    # -------------------------------------------------------------------------
    print("\nGenerando gráficos...")
    np.random.seed(42)
    graficar_boxplot(df_c240)
    graficar_latencias(df_c240)
    graficar_polaridad(df_c240)

    print("\n[OK] Script 05 v2 finalizado.")
