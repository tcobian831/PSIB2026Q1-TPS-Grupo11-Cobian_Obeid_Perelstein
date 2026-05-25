"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 04: Promediado y Cálculo del Potencial Evocado
==============================================================================

Pasos:
    1. Promediar trials por sujeto × canal × condición  →   PE individual
    2. Promediar PEs individuales por grupo             →  Grand Average
    3. Graficar PEs de ambos grupos superpuestos
    4. Guardar PEs para el Script 04

Entrada:  eeg_data_preprocesado.parquet  (generado por Script 02)
Salida:   eeg_PE_individual.parquet     (PE por sujeto)
          eeg_PE_grandaverage.parquet   (Grand Average por grupo)

Uso:
    Correr desde la carpeta scripts/
    python 04_promediado_PotencialesEvocados.py
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
N_SAMPLES = 256

CANALES_INTERES   = ["P8", "PO8", "T8", "TP8"]
CONDICIONES       = ["S1 obj", "S2 nomatch"]

# Ventana del componente c240/VMP (ms)
T_C240_INI = 220
T_C240_FIN = 260

ENTRADA         = Path("eeg_data_preprocesado.parquet")
SALIDA_IND      = Path("eeg_PE_individual.parquet")
SALIDA_GRAND    = Path("eeg_PE_grandaverage.parquet")

# =============================================================================
# FUNCIONES
# =============================================================================

def calcular_PE_individual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada combinación (sujeto, canal, condición) promedia el valor_uV
    de todos los trials en cada muestra. El resultado es una curva de 256
    puntos que representa la respuesta promedio del cerebro de ese sujeto
    a ese estímulo en ese canal.

    Args:
        df: DataFrame preprocesado con columnas
            sujeto, canal, condicion, trial_num, muestra, valor_uV, grupo

    Retorna:
        DataFrame con columnas:
            sujeto, grupo, canal, condicion, muestra, PE_uV, n_trials
    """
    print("Calculando PE individual (promedio por sujeto × canal × condición)...")

    PE = (
        df.groupby(["sujeto", "grupo", "canal", "condicion", "muestra"])
        ["valor_uV"]
        .agg(PE_uV="mean", n_trials="count")
        .reset_index()
    )

    return PE


def calcular_grand_average(PE_ind: pd.DataFrame) -> pd.DataFrame:
    """
    El Grand Average es el promedio de los PE de todos los sujetos de un
    mismo grupo. También calcula el error estándar (SEM) para graficar el
    intervalo de confianza alrededor de la curva media.

    SEM = std / sqrt(n_sujetos)  →  refleja la variabilidad entre sujetos

    Args:
        PE_ind: DataFrame con PE individuales (salida de calcular_PE_individual)

    Retorna:
        DataFrame con columnas:
            grupo, canal, condicion, muestra, grand_avg_uV, sem_uV, n_sujetos
    """
    print("  Calculando Grand Average (promedio entre sujetos por grupo)...")

    grand = (
        PE_ind.groupby(["grupo", "canal", "condicion", "muestra"])
        ["PE_uV"]
        .agg(
            grand_avg_uV="mean",
            std_uV="std",
            n_sujetos="count"
        )
        .reset_index()
    )

    grand["sem_uV"] = grand["std_uV"] / np.sqrt(grand["n_sujetos"])

    return grand


def tiempo_ms(muestra: np.ndarray) -> np.ndarray:
    """Convierte índices de muestra a tiempo en milisegundos."""
    return muestra / FS * 1000


def graficar_grand_average(grand: pd.DataFrame):
    """
    Grafica el Grand Average de ambos grupos superpuestos,
    para cada canal y condición de interés.

    Cada panel muestra:
    - Curva sólida: Grand Average del grupo
    - Banda sombreada: ± 1 SEM (variabilidad entre sujetos)
    - Franja naranja: ventana del componente c240 (220–260 ms)
    - Línea punteada vertical: onset del estímulo (t = 0)
    """
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    n_canales    = len(CANALES_INTERES)
    n_condiciones = len(CONDICIONES)

    fig, axes = plt.subplots(
        n_condiciones, n_canales,
        figsize=(5 * n_canales, 4 * n_condiciones),
        sharex=True, sharey=True
    )

    fig.suptitle(
        "Grand Average PE — Alcohólicos vs Controles\n"
        "Banda sombreada: ± 1 SEM entre sujetos",
        fontsize=13, y=1.01
    )

    for fila, condicion in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            subset = grand[
                (grand["canal"] == canal) &
                (grand["condicion"] == condicion)
            ]

            if subset.empty:
                ax.set_title(f"{canal} / {condicion}\n(sin datos)")
                continue

            for grupo in ["control", "alcoholic"]:
                datos_grupo = subset[subset["grupo"] == grupo].sort_values("muestra")
                if datos_grupo.empty:
                    continue

                t   = tiempo_ms(datos_grupo["muestra"].values)
                avg = datos_grupo["grand_avg_uV"].values
                sem = datos_grupo["sem_uV"].values
                n   = datos_grupo["n_sujetos"].iloc[0]
                color = colores[grupo]

                ax.plot(t, avg, color=color, linewidth=1.8,
                        label=f"{grupo.capitalize()} (n={n})")
                ax.fill_between(t, avg - sem, avg + sem,
                                color=color, alpha=0.15)

            # Ventana c240
            ax.axvspan(T_C240_INI, T_C240_FIN, alpha=0.12,
                       color="orange", label="Ventana c240")

            # Onset del estímulo
            ax.axvline(0, color="gray", linestyle="--",
                       linewidth=0.8, label="Onset")
            ax.axhline(0, color="black", linewidth=0.5)

            ax.set_title(f"Canal: {canal}\nCondición: {condicion}", fontsize=10)
            ax.set_xlabel("Tiempo (ms)")
            ax.set_ylabel("Amplitud (µV)")
            ax.grid(True, alpha=0.25)

            if fila == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig("figura_grand_average.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada como 'figura_grand_average.png'")


def graficar_PE_por_canal(grand: pd.DataFrame):
    """
    Grafica el Grand Average de ambas condiciones superpuestas,
    separado por canal y grupo. Útil para comparar S1 obj vs S2 nomatch
    dentro de cada grupo.
    """
    colores_cond = {"S1 obj": "#16a34a", "S2 nomatch": "#9333ea"}
    grupos = ["control", "alcoholic"]

    fig, axes = plt.subplots(
        len(grupos), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(grupos)),
        sharex=True, sharey=True
    )

    fig.suptitle(
        "Grand Average PE — S1 obj vs S2 nomatch por grupo y canal",
        fontsize=13, y=1.01
    )

    for fila, grupo in enumerate(grupos):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            for condicion in CONDICIONES:
                datos = grand[
                    (grand["grupo"] == grupo) &
                    (grand["canal"] == canal) &
                    (grand["condicion"] == condicion)
                ].sort_values("muestra")

                if datos.empty:
                    continue

                t   = tiempo_ms(datos["muestra"].values)
                avg = datos["grand_avg_uV"].values
                sem = datos["sem_uV"].values
                color = colores_cond[condicion]

                ax.plot(t, avg, color=color, linewidth=1.8, label=condicion)
                ax.fill_between(t, avg - sem, avg + sem,
                                color=color, alpha=0.15)

            ax.axvspan(T_C240_INI, T_C240_FIN, alpha=0.12, color="orange")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(
                f"{grupo.capitalize()} — Canal: {canal}", fontsize=10
            )
            ax.set_xlabel("Tiempo (ms)")
            ax.set_ylabel("Amplitud (µV)")
            ax.grid(True, alpha=0.25)

            if fila == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figura_PE_por_condicion.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada como 'figura_PE_por_condicion.png'")


def resumen_PE(grand: pd.DataFrame):
    """
    Imprime un resumen de la amplitud pico en la ventana c240
    para cada grupo, canal y condición.
    """
    print("\n" + "=" * 60)
    print("AMPLITUD PICO EN VENTANA c240 (220–260 ms)")
    print("=" * 60)

    # Muestras correspondientes a la ventana c240
    m_ini = int(T_C240_INI / 1000 * FS)
    m_fin = int(T_C240_FIN / 1000 * FS)

    for condicion in CONDICIONES:
        print(f"\nCondición: {condicion}")
        print(f"  {'Canal':<8} {'Control (µV)':>14} {'Alcoholic (µV)':>16}")
        print(f"  {'-'*8} {'-'*14} {'-'*16}")

        for canal in CANALES_INTERES:
            fila = []
            for grupo in ["control", "alcoholic"]:
                datos = grand[
                    (grand["grupo"] == grupo) &
                    (grand["canal"] == canal) &
                    (grand["condicion"] == condicion) &
                    (grand["muestra"] >= m_ini) &
                    (grand["muestra"] <= m_fin)
                ]
                if datos.empty:
                    fila.append(float("nan"))
                else:
                    fila.append(datos["grand_avg_uV"].max())

            print(f"  {canal:<8} {fila[0]:>14.3f} {fila[1]:>16.3f}")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 03: Promediado y Cálculo del PE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Cargar datos preprocesados
    # -------------------------------------------------------------------------
    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 02."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas")
    print(f"  Sujetos: {df['sujeto'].nunique()} "
          f"({df[df['grupo']=='alcoholic']['sujeto'].nunique()} alcohólicos, "
          f"{df[df['grupo']=='control']['sujeto'].nunique()} controles)")

    # -------------------------------------------------------------------------
    # Paso 1: PE individual
    # -------------------------------------------------------------------------
    print("\nPaso 1: PE individual")
    PE_ind = calcular_PE_individual(df)
    print(f"  {len(PE_ind):,} curvas PE individuales calculadas")

    PE_ind.to_parquet(SALIDA_IND, index=False)
    print(f"  Guardado en '{SALIDA_IND}'")

    # -------------------------------------------------------------------------
    # Paso 2: Grand Average
    # -------------------------------------------------------------------------
    print("\nPaso 2: Grand Average")
    grand = calcular_grand_average(PE_ind)
    print(f"  {len(grand):,} puntos del Grand Average calculados")

    grand.to_parquet(SALIDA_GRAND, index=False)
    print(f"  Guardado en '{SALIDA_GRAND}'")

    # -------------------------------------------------------------------------
    # Paso 3: Resumen de amplitudes en ventana c240
    # -------------------------------------------------------------------------
    resumen_PE(grand)

    # -------------------------------------------------------------------------
    # Paso 4: Gráficos
    # -------------------------------------------------------------------------
    print("\nGenerando gráficos...")
    graficar_grand_average(grand)
    graficar_PE_por_canal(grand)

    print("\n[OK] Script 04 finalizado.")
