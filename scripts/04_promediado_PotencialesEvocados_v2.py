"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 04 v2: Promediado y Cálculo del Potencial Evocado
==============================================================================

CAMBIOS RESPECTO DE v1
----------------------
- Lee 'eeg_data_preprocesado_v2.parquet' (baseline corregido a 30 ms)
- Escribe 'eeg_PE_individual_v2.parquet' y 'eeg_PE_grandaverage_v2.parquet'
- Sin cambios en la lógica de promediado

Entrada:  eeg_data_preprocesado_v2.parquet  (generado por Script 03 v2)
Salida:   eeg_PE_individual_v2.parquet
          eeg_PE_grandaverage_v2.parquet

Uso:
    python 04_promediado_PotencialesEvocados_v2.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS = 256
N_SAMPLES = 256

CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES     = ["S1 obj", "S2 nomatch"]

T_C240_INI = 220
T_C240_FIN = 260

ENTRADA      = Path("../outputs/eeg_data_preprocesado_v2.parquet")
SALIDA_IND   = Path("../outputs/eeg_PE_individual_v2.parquet")
SALIDA_GRAND = Path("../outputs/eeg_PE_grandaverage_v2.parquet")

# =============================================================================
# FUNCIONES
# =============================================================================

def calcular_PE_individual(df):
    print("Calculando PE individual (promedio por sujeto x canal x condicion)...")
    PE = (
        df.groupby(["sujeto", "grupo", "canal", "condicion", "muestra"])
        ["valor_uV"]
        .agg(PE_uV="mean", n_trials="count")
        .reset_index()
    )
    return PE


def calcular_grand_average(PE_ind):
    print("  Calculando Grand Average (promedio entre sujetos por grupo)...")
    grand = (
        PE_ind.groupby(["grupo", "canal", "condicion", "muestra"])
        ["PE_uV"]
        .agg(grand_avg_uV="mean", std_uV="std", n_sujetos="count")
        .reset_index()
    )
    grand["sem_uV"] = grand["std_uV"] / np.sqrt(grand["n_sujetos"])
    return grand


def tiempo_ms(muestra):
    return muestra / FS * 1000


def graficar_grand_average(grand):
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    n_canales     = len(CANALES_INTERES)
    n_condiciones = len(CONDICIONES)

    fig, axes = plt.subplots(
        n_condiciones, n_canales,
        figsize=(5 * n_canales, 4 * n_condiciones),
        sharex=True, sharey=True
    )
    fig.suptitle(
        "Grand Average PE v2 — Alcoholicos vs Controles\n"
        "Banda sombreada: +/- 1 SEM entre sujetos",
        fontsize=13, y=1.01
    )

    for fila, condicion in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col] if n_condiciones > 1 else axes[col]

            subset = grand[
                (grand["canal"] == canal) &
                (grand["condicion"] == condicion)
            ]
            if subset.empty:
                ax.set_title(f"{canal} / {condicion}\n(sin datos)")
                continue

            for grupo in ["control", "alcoholic"]:
                datos = subset[subset["grupo"] == grupo].sort_values("muestra")
                if datos.empty:
                    continue
                t   = tiempo_ms(datos["muestra"].values)
                avg = datos["grand_avg_uV"].values
                sem = datos["sem_uV"].values
                n   = datos["n_sujetos"].iloc[0]
                color = colores[grupo]
                ax.plot(t, avg, color=color, linewidth=1.8,
                        label=f"{grupo.capitalize()} (n={n})")
                ax.fill_between(t, avg - sem, avg + sem,
                                color=color, alpha=0.15)

            ax.axvspan(T_C240_INI, T_C240_FIN, alpha=0.12,
                       color="orange", label="Ventana c240")
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"Canal: {canal}\nCondicion: {condicion}", fontsize=10)
            ax.set_xlabel("Tiempo (ms)")
            ax.set_ylabel("Amplitud (uV)")
            ax.grid(True, alpha=0.25)

            if fila == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig("../outputs/figura_grand_average_v2.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_grand_average_v2.png'")


def resumen_PE(grand):
    print("\n" + "=" * 60)
    print("AMPLITUD PICO EN VENTANA c240 (220-260 ms) — v2")
    print("=" * 60)
    m_ini = int(T_C240_INI / 1000 * FS)
    m_fin = int(T_C240_FIN / 1000 * FS)
    for condicion in CONDICIONES:
        print(f"\nCondicion: {condicion}")
        print(f"  {'Canal':<8} {'Control (uV)':>14} {'Alcoholic (uV)':>16}")
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
                fila.append(datos["grand_avg_uV"].max() if not datos.empty else float("nan"))
            print(f"  {canal:<8} {fila[0]:>14.3f} {fila[1]:>16.3f}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 04 v2: Promediado y Calculo del PE")
    print("=" * 60)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 03 v2."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas")
    print(f"  Sujetos: {df['sujeto'].nunique()}")

    print("\nPaso 1: PE individual")
    PE_ind = calcular_PE_individual(df)
    print(f"  {len(PE_ind):,} curvas PE individuales calculadas")
    PE_ind.to_parquet(SALIDA_IND, index=False)
    print(f"  Guardado en '{SALIDA_IND}'")

    print("\nPaso 2: Grand Average")
    grand = calcular_grand_average(PE_ind)
    print(f"  {len(grand):,} puntos calculados")
    grand.to_parquet(SALIDA_GRAND, index=False)
    print(f"  Guardado en '{SALIDA_GRAND}'")

    resumen_PE(grand)

    print("\nGenerando grafico Grand Average...")
    graficar_grand_average(grand)

    print("\n[OK] Script 04 v2 finalizado.")
