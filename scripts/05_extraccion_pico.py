"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 05: Extracción del Pico Global y Jerarquía de Promedios
==============================================================================

Propósito:
    Para cada sujeto, canal y condición, extraer la amplitud máxima del PE
    y el instante en que ocurre (latencia), SIN restricción de ventana fija.

    Luego construir la jerarquía de promedios propuesta:
        A(c,p,k) — amplitud máxima por canal c, sujeto p, condición k
             ↓ promediar sobre sujetos p
        A(c,k)   — amplitud representativa por canal y condición
             ↓ promediar sobre canales c
        A(k)     — amplitud representativa por grupo y condición

    Lo mismo para la latencia t(c,p,k) → t(c,k) → t(k).

Decisiones de diseño:
    - Se usa el PE homogéneo (demostrado superior en Script 04)
    - Se busca el MÁXIMO POSITIVO a partir de los 30 ms post-estímulo
      para evitar confundir componentes tempranos (P30, N70, P100)
      con el componente de interés (c240/VMP)
    - Se reporta también la latencia para analizar si el pico ocurre
      dentro o fuera de la ventana clásica 220-260 ms del paper

Entrada:  outputs/eeg_PE_homogeneo.parquet  (generado por Script 04)
Salida:   outputs/eeg_pico_global.csv
          outputs/figura_pico_amplitud.png
          outputs/figura_pico_latencia.png
          outputs/figura_pico_jerarquia.png

Uso:
    Correr desde la carpeta scripts/
    python 05_extraccion_pico.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS        = 256   # Hz
N_SAMPLES = 256   # muestras por trial

# Tiempo mínimo para buscar el pico (ms)
# Evitamos los primeros 150 ms para no confundir con componentes tempranos
T_MIN_MS  = 150  # excluye P1 y N1
T_MAX_MS  = 350  # excluye P300

# Canales y condiciones
CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES     = ["S1 obj", "S2 nomatch"]

# Muestra mínima para buscar el pico
M_MIN = int(T_MIN_MS / 1000 * FS)  # = 7

ENTRADA     = Path("../outputs/eeg_PE_homogeneo.parquet")
SALIDA_CSV  = Path("../outputs/eeg_pico_global.csv")

# =============================================================================
# FUNCIONES DE EXTRACCIÓN
# =============================================================================

def extraer_pico(señal: np.ndarray) -> tuple:
    """
    Extrae la amplitud máxima positiva y su latencia de un PE individual,
    buscando solo a partir de la muestra M_MIN.

    Buscamos el MÁXIMO POSITIVO porque:
    - El componente c240/VMP es de polaridad positiva
    - Los primeros 30 ms corresponden a componentes tempranos que no
      son de nuestro interés

    Args:
        señal: array 1D de N_SAMPLES puntos (PE individual)

    Retorna:
        (amplitud_uV, latencia_ms)
        amplitud_uV: valor máximo positivo en la señal post 30 ms
        latencia_ms: tiempo en ms donde ocurre ese máximo
    """
    M_MAX      = int(T_MAX_MS / 1000 * FS)  # = 89
    señal_vent = señal[M_MIN:M_MAX]
    idx_max    = np.argmax(señal_vent)
    amplitud   = float(señal_vent[idx_max])
    latencia   = float((idx_max + M_MIN) / FS * 1000)
    return amplitud, latencia


def extraer_todos_picos(pe_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica extraer_pico() a cada combinación sujeto×canal×condición.

    Retorna DataFrame con columnas:
        sujeto, grupo, canal, condicion, amplitud_uV, latencia_ms, n_trials
    """
    resultados = []
    grupos = pe_ind.groupby(["sujeto", "grupo", "canal", "condicion"])
    n_total = len(grupos)
    print(f"  Extrayendo pico de {n_total:,} combinaciones...")

    for (sujeto, grupo, canal, cond), sub in grupos:
        s = sub.sort_values("muestra")["PE_uV"].values
        if len(s) != N_SAMPLES:
            continue

        amp, lat = extraer_pico(s)
        n_trials = sub["n_trials"].iloc[0] if "n_trials" in sub.columns else np.nan

        resultados.append({
            "sujeto":      sujeto,
            "grupo":       grupo,
            "canal":       canal,
            "condicion":   cond,
            "amplitud_uV": amp,
            "latencia_ms": lat,
            "n_trials":    n_trials,
        })

    return pd.DataFrame(resultados)


# =============================================================================
# JERARQUÍA DE PROMEDIOS
# =============================================================================

def calcular_jerarquia(df_pico: pd.DataFrame) -> tuple:
    """
    Construye la jerarquía de promedios:

        Nivel 1: A(c,p,k) — ya está en df_pico (un valor por sujeto×canal×condición)

        Nivel 2: A(c,k) — promedio de A(c,p,k) sobre todos los sujetos p
                           del mismo grupo, para cada canal c y condición k

        Nivel 3: A(k)   — promedio de A(c,k) sobre todos los canales c
                           para cada grupo y condición k

    Lo mismo para la latencia t(c,p,k) → t(c,k) → t(k).

    Retorna:
        nivel2: DataFrame con A(c,k) y t(c,k)
        nivel3: DataFrame con A(k) y t(k)
    """
    # Nivel 2: promediar sobre sujetos
    nivel2 = (
        df_pico
        .groupby(["grupo", "canal", "condicion"])
        .agg(
            A_media=("amplitud_uV", "mean"),
            A_sd=("amplitud_uV", "std"),
            t_media=("latencia_ms", "mean"),
            t_sd=("latencia_ms", "std"),
            n_sujetos=("sujeto", "count")
        )
        .reset_index()
    )

    # Nivel 3: promediar sobre canales
    nivel3 = (
        nivel2
        .groupby(["grupo", "condicion"])
        .agg(
            A_global=("A_media", "mean"),
            A_global_sd=("A_media", "std"),
            t_global=("t_media", "mean"),
            t_global_sd=("t_media", "std"),
        )
        .reset_index()
    )

    return nivel2, nivel3


def imprimir_jerarquia(nivel2: pd.DataFrame, nivel3: pd.DataFrame):
    """Imprime la tabla de la jerarquía de promedios."""
    print("\n" + "=" * 70)
    print("JERARQUÍA DE PROMEDIOS — Amplitud y Latencia del pico global")
    print("=" * 70)

    print("\n--- NIVEL 2: A(c,k) — promedio por canal y condición ---")
    print(f"  {'Grupo':<12} {'Canal':<6} {'Condición':<12} "
          f"{'Amplitud (µV)':>16} {'Latencia (ms)':>16} {'N':>5}")
    print(f"  {'-'*12} {'-'*6} {'-'*12} {'-'*16} {'-'*16} {'-'*5}")

    for _, r in nivel2.sort_values(
        ["condicion", "grupo", "canal"]
    ).iterrows():
        print(f"  {r['grupo']:<12} {r['canal']:<6} {r['condicion']:<12} "
              f"{r['A_media']:>+10.3f} ± {r['A_sd']:>5.3f}  "
              f"{r['t_media']:>10.1f} ± {r['t_sd']:>5.1f}  "
              f"{int(r['n_sujetos']):>5}")

    print("\n--- NIVEL 3: A(k) — amplitud representativa por grupo ---")
    print(f"  {'Grupo':<12} {'Condición':<12} "
          f"{'A global (µV)':>16} {'t global (ms)':>16}")
    print(f"  {'-'*12} {'-'*12} {'-'*16} {'-'*16}")

    for _, r in nivel3.sort_values(["condicion", "grupo"]).iterrows():
        print(f"  {r['grupo']:<12} {r['condicion']:<12} "
              f"{r['A_global']:>+10.3f} ± {r['A_global_sd']:>5.3f}  "
              f"{r['t_global']:>10.1f} ± {r['t_global_sd']:>5.1f}")

    print("=" * 70)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def graficar_amplitud_pico(df_pico: pd.DataFrame):
    """
    Boxplots de la amplitud del pico global por grupo,
    para cada canal y condición.
    """
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharey=False
    )
    fig.suptitle(
        "Amplitud del pico máximo del PE por sujeto\n"
        "Búsqueda en toda la señal (>150 ms post-estímulo)",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    np.random.seed(42)

    for fila, cond in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]
            datos = []
            etiquetas = []
            cols_bp = []

            for grupo in ["control", "alcoholic"]:
                vals = df_pico[
                    (df_pico["canal"] == canal) &
                    (df_pico["condicion"] == cond) &
                    (df_pico["grupo"] == grupo)
                ]["amplitud_uV"].dropna().values

                datos.append(vals)
                n = len(vals)
                m = np.mean(vals)
                s = np.std(vals, ddof=1)
                etiquetas.append(f"{grupo.capitalize()}\nn={n}\n"
                                f"{m:+.2f}±{s:.2f} µV")
                cols_bp.append(colores[grupo])

            bp = ax.boxplot(
                datos, patch_artist=True,
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.4}
            )
            for patch, color in zip(bp["boxes"], cols_bp):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

            for i, (vals, color) in enumerate(zip(datos, cols_bp)):
                jitter = np.random.normal(i + 1, 0.07, size=len(vals))
                ax.scatter(jitter, vals, alpha=0.35,
                          color=color, s=12, zorder=3)

            ax.axhline(0, color="black", linewidth=0.6,
                      linestyle="--", alpha=0.5)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(etiquetas, fontsize=8)
            ax.set_title(f"Canal: {canal}\n{cond}", fontsize=10)
            ax.set_ylabel("Amplitud pico (µV)")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_pico_amplitud.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_pico_amplitud.png'")


def graficar_latencia_pico(df_pico: pd.DataFrame):
    """
    Histogramas de la latencia del pico por grupo y canal.
    Permite verificar si el pico ocurre alrededor de los 240 ms
    o en otro momento.
    """
    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharex=True
    )
    fig.suptitle(
        "Distribución de latencias del pico máximo del PE\n",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    for fila, cond in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            for grupo in ["control", "alcoholic"]:
                lats = df_pico[
                    (df_pico["canal"] == canal) &
                    (df_pico["condicion"] == cond) &
                    (df_pico["grupo"] == grupo)
                ]["latencia_ms"].dropna().values

                ax.hist(lats, bins=20, alpha=0.5,
                       color=colores[grupo],
                       label=grupo.capitalize(),
                       density=True)

            ax.axvline(220, color="orange", linestyle="--",
                        linewidth=1.5, label="Ventana paper (220-260 ms)")
            ax.axvline(260, color="orange", linestyle="--",
                        linewidth=1.5)
            ax.axvspan(220, 260, alpha=0.15, color="orange")
            ax.set_title(f"Canal: {canal}\n{cond}", fontsize=10)
            ax.set_xlabel("Latencia (ms)")
            ax.set_ylabel("Densidad")
            ax.grid(True, alpha=0.3)
            if fila == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("../outputs/figura_pico_latencia.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_pico_latencia.png'")


def graficar_jerarquia(nivel2: pd.DataFrame, nivel3: pd.DataFrame):
    """
    Grafica la jerarquía de promedios:
    - Nivel 2: A(c,k) — barras por canal
    - Nivel 3: A(k)   — valor único por grupo y condición anotado
    """
    fig, axes = plt.subplots(
        1, len(CONDICIONES),
        figsize=(7 * len(CONDICIONES), 5)
    )
    fig.suptitle(
        "Jerarquía de promedios — Amplitud media del pico\n"
        "Barras: A(c,k) por canal | Línea punteada: A(k) global por grupo",
        fontsize=13
    )

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    x = np.arange(len(CANALES_INTERES))
    ancho = 0.35

    for ax, cond in zip(axes, CONDICIONES):
        for i, grupo in enumerate(["control", "alcoholic"]):
            sub2 = nivel2[
                (nivel2["condicion"] == cond) &
                (nivel2["grupo"] == grupo)
            ].set_index("canal").reindex(CANALES_INTERES)

            medias = sub2["A_media"].values
            sds    = sub2["A_sd"].values

            ax.bar(x + i * ancho, medias, ancho,
                  yerr=sds, capsize=4,
                  label=f"{grupo.capitalize()}",
                  color=colores[grupo], alpha=0.75,
                  error_kw={"linewidth": 1.2})

            # Línea del nivel 3 (A global)
            sub3 = nivel3[
                (nivel3["condicion"] == cond) &
                (nivel3["grupo"] == grupo)
            ]
            if not sub3.empty:
                a_global = sub3["A_global"].values[0]
                ax.axhline(
                    a_global,
                    color=colores[grupo],
                    linestyle="--", linewidth=1.5,
                    alpha=0.7,
                    label=f"A global {grupo.capitalize()} = {a_global:.2f} µV"
                )

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condición: {cond}", fontsize=11)
        ax.set_ylabel("Amplitud media del pico (µV)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_pico_jerarquia.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_pico_jerarquia.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 05: Extracción del Pico Global")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Cargar PE homogéneo
    # -------------------------------------------------------------------------
    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 04."
        )

    print(f"\nCargando '{ENTRADA}'...")
    pe_ind = pd.read_parquet(ENTRADA)
    print(f"  {pe_ind['sujeto'].nunique()} sujetos cargados")
    print(f"  Buscando pico máximo a partir de t = {T_MIN_MS} ms "
          f"(muestra {M_MIN})")

    # -------------------------------------------------------------------------
    # Extracción del pico
    # -------------------------------------------------------------------------
    print("\nExtrayendo pico global por sujeto×canal×condición...")
    df_pico = extraer_todos_picos(pe_ind)
    print(f"  {len(df_pico)} combinaciones procesadas")

    # -------------------------------------------------------------------------
    # Jerarquía de promedios
    # -------------------------------------------------------------------------
    print("\nCalculando jerarquía de promedios...")
    nivel2, nivel3 = calcular_jerarquia(df_pico)
    imprimir_jerarquia(nivel2, nivel3)

    # -------------------------------------------------------------------------
    # Guardar
    # -------------------------------------------------------------------------
    df_pico.to_csv(SALIDA_CSV, index=False)
    print(f"\nDatos guardados en '{SALIDA_CSV}'")

    # -------------------------------------------------------------------------
    # Gráficos
    # -------------------------------------------------------------------------
    print("\nGenerando gráficos...")
    np.random.seed(42)
    graficar_amplitud_pico(df_pico)
    graficar_latencia_pico(df_pico)
    graficar_jerarquia(nivel2, nivel3)

    print("\n[OK] Script 05 finalizado.")
