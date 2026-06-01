"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 04: Promediado de Trials — Homogéneo vs Inhomogéneo
==============================================================================

Propósito:
    Obtener el Potencial Evocado (PE) individual por sujeto, canal y
    condición mediante dos estrategias de promediado:

    (A) HOMOGÉNEO: promedio aritmético clásico. Todos los trials pesan igual.
            PE(t) = (1/N) * Σ x_i(t)

    (B) INHOMOGÉNEO (amplitud variable, ruido constante):
        Modelo: x_i(t) = a_i * s(t) + n_i(t)
        Donde a_i es la amplitud relativa del trial i (varía entre trials)
        y n_i(t) es ruido con varianza aproximadamente constante.
        El estimador óptimo pondera cada trial según su similitud con el PE:
            a_i_hat = <x_i, s_hat> / ||s_hat||²
            PE(t)   = Σ(a_i_hat * x_i) / Σ(a_i_hat)
        Se usa el promedio homogéneo como estimación inicial de s_hat.
        Los pesos negativos se truncan en cero.

    Justificación del modelo inhomogéneo:
        - No requiere muestras pre-estímulo (que no tenemos en este dataset)
        - Los trials con mayor amplitud de señal contribuyen más al PE
        - Es el modelo estándar cuando no se puede estimar la varianza del
          ruido por trial (Davila & Mobin, 1992)

    Comparación entre métodos:
        - Grand Average de ambos métodos superpuestos
        - SNR por sujeto: var(ventana señal 100-400ms) / var(ventana baseline 0-30ms)

Entrada:  outputs/eeg_data_preprocesado.parquet  (generado por Script 03)
Salida:   outputs/eeg_PE_homogeneo.parquet
          outputs/eeg_PE_inhomogeneo.parquet
          outputs/eeg_GA_homogeneo.parquet
          outputs/eeg_GA_inhomogeneo.parquet
          outputs/tabla_snr_comparacion.csv
          outputs/figura_GA_comparacion.png
          outputs/figura_snr_comparacion.png

Uso:
    Correr desde la carpeta scripts/
    python 04_promediado.py
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

CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES     = ["S1 obj", "S2 nomatch"]

# Ventana de señal para calcular SNR (ms)
T_SENAL_INI_MS  = 100
T_SENAL_FIN_MS  = 400

# Ventana de baseline para calcular SNR (ms)
# Usamos los primeros 30 ms como aproximación al baseline
T_BASELINE_MS = 30

ENTRADA          = Path("../outputs/eeg_data_preprocesado.parquet")
SALIDA_HOM       = Path("../outputs/eeg_PE_homogeneo.parquet")
SALIDA_INH       = Path("../outputs/eeg_PE_inhomogeneo.parquet")
SALIDA_GA_HOM    = Path("../outputs/eeg_GA_homogeneo.parquet")
SALIDA_GA_INH    = Path("../outputs/eeg_GA_inhomogeneo.parquet")
SALIDA_SNR       = Path("../outputs/tabla_snr_comparacion.csv")

# =============================================================================
# FUNCIONES DE PROMEDIADO
# =============================================================================

def trials_a_matriz(df_grupo: pd.DataFrame) -> np.ndarray:
    """
    Convierte el DataFrame de un grupo (sujeto×canal×condición) en una
    matriz X de shape (n_trials, N_SAMPLES).

    Solo incluye trials con exactamente N_SAMPLES muestras (trials completos).

    Args:
        df_grupo: DataFrame con columnas trial_num, muestra, valor_uV

    Retorna:
        matriz numpy (n_trials, N_SAMPLES) o None si no hay trials válidos
    """
    pivot = (
        df_grupo
        .sort_values(["trial_num", "muestra"])
        .pivot_table(index="trial_num", columns="muestra",
                     values="valor_uV", aggfunc="first")
        .reindex(columns=np.arange(N_SAMPLES))
        .dropna(axis=0, how="any")
    )
    if pivot.empty or pivot.shape[0] < 2:
        return None
    return pivot.values  # (n_trials, N_SAMPLES)


def promedio_homogeneo(X: np.ndarray) -> np.ndarray:
    """
    Promedio aritmético clásico: PE = (1/N) * Σ x_i

    Todos los trials tienen el mismo peso.

    Args:
        X: matriz (n_trials, N_SAMPLES)

    Retorna:
        array 1D (N_SAMPLES,) con el PE promediado
    """
    return X.mean(axis=0)


def promedio_inhomogeneo(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Promedio inhomogéneo — caso amplitud variable, ruido de varianza constante.

    Modelo: x_i[n] = a_i·s[n] + v_i[n],  var(v_i) = σ² constante.
    Estimador LS:  ŝ = Σ aᵢ·xᵢ / Σ aᵢ²

    Pasos (según la teórica):
      1. ŝ inicial = promedio ordinario del ensamble.
      2. aᵢ = <xᵢ, ŝ> / <ŝ, ŝ>  → proyección de cada trial sobre ŝ
         (quedan centradas en 1; mantienen la escala en µV).
      3. ŝ = Σ aᵢ·xᵢ / Σ aᵢ²    → promedio ponderado.

    No se truncan amplitudes negativas (la teórica no lo indica y truncar sesga).
    """
    s_hat = promedio_homogeneo(X)              # paso 1

    den = float(np.dot(s_hat, s_hat))
    if den < eps:                              # ŝ ≈ 0 → no hay señal estimable
        return s_hat

    a = X @ s_hat / den                        # paso 2: amplitudes (media ≈ 1)

    sse = float(np.dot(a, a))                  # Σ aᵢ²
    if sse < eps:
        return s_hat

    return (a[:, None] * X).sum(axis=0) / sse  # paso 3


# =============================================================================
# CÁLCULO DE PE INDIVIDUAL
# =============================================================================

def calcular_PE_individual(df: pd.DataFrame) -> tuple:
    """
    Para cada combinación (sujeto, grupo, canal, condición):
        1. Arma la matriz de trials
        2. Calcula PE homogéneo e inhomogéneo
        3. Guarda ambos en formato largo (una fila por muestra)

    Retorna:
        pe_hom_df: DataFrame con PE homogéneo
        pe_inh_df: DataFrame con PE inhomogéneo
    """
    grupos = df.groupby(["sujeto", "grupo", "canal", "condicion"])
    n_total = len(grupos)
    print(f"  Combinaciones sujeto×canal×condición: {n_total:,}")

    filas_hom = []
    filas_inh = []

    for idx, ((sujeto, grupo, canal, cond), sub) in enumerate(grupos):
        if idx % 2000 == 0:
            print(f"  Progreso: {idx:,}/{n_total:,}...")

        X = trials_a_matriz(sub)
        if X is None:
            continue

        n_trials  = X.shape[0]
        pe_hom    = promedio_homogeneo(X)
        pe_inh    = promedio_inhomogeneo(X)
        muestras  = np.arange(N_SAMPLES)

        base = {
            "sujeto":   sujeto,
            "grupo":    grupo,
            "canal":    canal,
            "condicion": cond,
            "n_trials": n_trials,
        }

        for m, v_h, v_i in zip(muestras, pe_hom, pe_inh):
            filas_hom.append({**base, "muestra": int(m), "PE_uV": float(v_h)})
            filas_inh.append({**base, "muestra": int(m), "PE_uV": float(v_i)})

    return pd.DataFrame(filas_hom), pd.DataFrame(filas_inh)


# =============================================================================
# GRAND AVERAGE
# =============================================================================

def calcular_grand_average(pe_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Promedia los PE individuales de todos los sujetos del mismo grupo.
    También calcula el error estándar (SEM) para graficar bandas de
    variabilidad entre sujetos.

    Retorna DataFrame con columnas:
        grupo, canal, condicion, muestra, grand_avg_uV, sem_uV, n_sujetos
    """
    grand = (
        pe_ind
        .groupby(["grupo", "canal", "condicion", "muestra"])["PE_uV"]
        .agg(grand_avg_uV="mean", std_uV="std", n_sujetos="count")
        .reset_index()
    )
    grand["sem_uV"] = grand["std_uV"] / np.sqrt(grand["n_sujetos"].clip(lower=1))
    return grand


# =============================================================================
# COMPARACIÓN DE MÉTODOS — SNR
# =============================================================================

def calcular_snr(pe_ind: pd.DataFrame, metodo: str) -> pd.DataFrame:
    """
    Calcula el SNR por sujeto×canal×condición.

    SNR = var(ventana de señal) / var(ventana de baseline)

    Ventana de señal:   100–400 ms (donde ocurren los principales PE)
    Ventana de baseline: 0–30 ms  (antes de que llegue la respuesta neural)

    Un SNR mayor indica que el PE está mejor definido respecto al ruido.

    Args:
        pe_ind:  DataFrame con PE individuales
        metodo:  etiqueta del método ("homogeneo" o "inhomogeneo")

    Retorna:
        DataFrame con columnas: sujeto, grupo, canal, condicion, snr, metodo
    """
    m_bl  = int(T_BASELINE_MS / 1000 * FS)
    m_si  = int(T_SENAL_INI_MS / 1000 * FS)
    m_sf  = int(T_SENAL_FIN_MS / 1000 * FS)

    filas = []
    for (sujeto, grupo, canal, cond), sub in pe_ind.groupby(
        ["sujeto", "grupo", "canal", "condicion"]
    ):
        s = sub.sort_values("muestra")["PE_uV"].values
        if len(s) != N_SAMPLES:
            continue

        var_bl = np.var(s[:m_bl])
        var_se = np.var(s[m_si:m_sf])
        snr    = var_se / var_bl if var_bl > 1e-12 else np.nan

        filas.append({
            "sujeto":   sujeto,
            "grupo":    grupo,
            "canal":    canal,
            "condicion": cond,
            "snr":      float(snr) if not np.isnan(snr) else np.nan,
            "metodo":   metodo,
        })
    return pd.DataFrame(filas)


def comparar_snr(snr_hom: pd.DataFrame, snr_inh: pd.DataFrame):
    """Imprime resumen comparativo de SNR entre métodos."""
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE SNR: Homogéneo vs Inhomogéneo")
    print("(SNR = var señal 100-400ms / var baseline 0-30ms)")
    print("=" * 60)

    for metodo, df_snr in [("Homogéneo", snr_hom),
                            ("Inhomogéneo", snr_inh)]:
        vals = df_snr["snr"].dropna()
        print(f"\n{metodo}:")
        print(f"  Mediana SNR: {vals.median():.3f}")
        print(f"  Media SNR:   {vals.mean():.3f}")
        print(f"  N:           {len(vals)}")

    # Comparar por canal y condición
    print(f"\n{'Canal':<5} {'Condición':<12} "
          f"{'SNR Hom':>10} {'SNR Inh':>10} {'Ganador':>10}")
    print("-" * 50)

    snr_hom_g = snr_hom.groupby(["canal", "condicion"])["snr"].median()
    snr_inh_g = snr_inh.groupby(["canal", "condicion"])["snr"].median()

    for canal in CANALES_INTERES:
        for cond in CONDICIONES:
            sh = snr_hom_g.get((canal, cond), np.nan)
            si = snr_inh_g.get((canal, cond), np.nan)
            ganador = "Inh" if si > sh else "Hom"
            print(f"{canal:<5} {cond:<12} {sh:>10.3f} {si:>10.3f} "
                  f"{ganador:>10}")
    print("=" * 60)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def graficar_grand_average_comparativo(ga_hom: pd.DataFrame,
                                        ga_inh: pd.DataFrame):
    """
    Grafica el Grand Average de ambos métodos superpuestos para cada
    canal y condición. Permite comparar visualmente si el método
    inhomogéneo genera un PE más claro.
    """
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    estilos = {"homogeneo": "-", "inhomogeneo": "--"}

    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharex=True, sharey=True
    )

    fig.suptitle(
        "Grand Average — Homogéneo (línea sólida) vs Inhomogéneo (línea punteada)\n"
        "Control (azul) vs Alcohólico (rojo)",
        fontsize=13
    )

    tiempo_ms = np.arange(N_SAMPLES) / FS * 1000

    for fila, cond in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            for grupo in ["control", "alcoholic"]:
                color = colores[grupo]

                # Homogéneo
                d_hom = ga_hom[
                    (ga_hom["grupo"] == grupo) &
                    (ga_hom["canal"] == canal) &
                    (ga_hom["condicion"] == cond)
                ].sort_values("muestra")

                if not d_hom.empty:
                    ax.plot(tiempo_ms, d_hom["grand_avg_uV"].values,
                           color=color, linewidth=1.8,
                           linestyle="-",
                           label=f"{grupo.capitalize()} Hom" if col == 0 and fila == 0 else "")
                    ax.fill_between(
                        tiempo_ms,
                        d_hom["grand_avg_uV"] - d_hom["sem_uV"],
                        d_hom["grand_avg_uV"] + d_hom["sem_uV"],
                        color=color, alpha=0.1
                    )

                # Inhomogéneo
                d_inh = ga_inh[
                    (ga_inh["grupo"] == grupo) &
                    (ga_inh["canal"] == canal) &
                    (ga_inh["condicion"] == cond)
                ].sort_values("muestra")

                if not d_inh.empty:
                    ax.plot(tiempo_ms, d_inh["grand_avg_uV"].values,
                           color=color, linewidth=1.8,
                           linestyle="--",
                           label=f"{grupo.capitalize()} Inh" if col == 0 and fila == 0 else "")

            ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_title(f"Canal: {canal}\nCondición: {cond}", fontsize=10)
            ax.set_xlabel("Tiempo (ms)")
            ax.set_ylabel("Amplitud (µV)")
            ax.grid(True, alpha=0.25)

            if fila == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig("../outputs/figura_GA_comparacion.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_GA_comparacion.png'")


def graficar_snr_comparacion(snr_hom: pd.DataFrame,
                              snr_inh: pd.DataFrame):
    """
    Boxplots del SNR por método, agrupados por canal y condición.
    """
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5))

    fig.suptitle(
        "Comparación de SNR: Homogéneo vs Inhomogéneo\n"
        "SNR = var(señal 100-400ms) / var(baseline 0-30ms)",
        fontsize=13
    )

    colores = {"homogeneo": "#94a3b8", "inhomogeneo": "#16a34a"}
    x = np.arange(len(CANALES_INTERES))
    ancho = 0.35

    for ax, cond in zip(axes, CONDICIONES):
        for i, (metodo, df_snr) in enumerate([("homogeneo", snr_hom),
                                               ("inhomogeneo", snr_inh)]):
            medianas = []
            for canal in CANALES_INTERES:
                vals = df_snr[
                    (df_snr["canal"] == canal) &
                    (df_snr["condicion"] == cond)
                ]["snr"].dropna().values
                medianas.append(np.median(vals) if len(vals) else np.nan)

            ax.bar(x + i * ancho, medianas, ancho,
                  label=metodo.capitalize(),
                  color=colores[metodo], alpha=0.8)

        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condición: {cond}", fontsize=11)
        ax.set_ylabel("SNR mediano (var señal / var baseline)")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_snr_comparacion.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_snr_comparacion.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 04: Promediado Homogéneo vs Inhomogéneo")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Cargar datos preprocesados
    # -------------------------------------------------------------------------
    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'.\n"
            "Asegurate de haber corrido primero el Script 03."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas")
    print(f"  Sujetos: {df['sujeto'].nunique()} "
          f"({df[df['grupo']=='alcoholic']['sujeto'].nunique()} alcohólicos, "
          f"{df[df['grupo']=='control']['sujeto'].nunique()} controles)")

    # -------------------------------------------------------------------------
    # Calcular PE individuales con ambos métodos
    # -------------------------------------------------------------------------
    print("\nCalculando PE individuales (homogéneo e inhomogéneo)...")
    pe_hom, pe_inh = calcular_PE_individual(df)

    print(f"\n  PE homogéneo:    {len(pe_hom):,} filas")
    print(f"  PE inhomogéneo:  {len(pe_inh):,} filas")

    # -------------------------------------------------------------------------
    # Grand Average
    # -------------------------------------------------------------------------
    print("\nCalculando Grand Average...")
    ga_hom = calcular_grand_average(pe_hom)
    ga_inh = calcular_grand_average(pe_inh)

    # -------------------------------------------------------------------------
    # Guardar
    # -------------------------------------------------------------------------
    print("\nGuardando archivos...")
    pe_hom.to_parquet(SALIDA_HOM, index=False)
    pe_inh.to_parquet(SALIDA_INH, index=False)
    ga_hom.to_parquet(SALIDA_GA_HOM, index=False)
    ga_inh.to_parquet(SALIDA_GA_INH, index=False)
    print(f"  PE homogéneo → '{SALIDA_HOM}'")
    print(f"  PE inhomogéneo → '{SALIDA_INH}'")

    # -------------------------------------------------------------------------
    # Comparación de SNR
    # -------------------------------------------------------------------------
    print("\nCalculando SNR por sujeto×canal×condición...")
    snr_hom = calcular_snr(pe_hom, "homogeneo")
    snr_inh = calcular_snr(pe_inh, "inhomogeneo")

    comparar_snr(snr_hom, snr_inh)

    snr_total = pd.concat([snr_hom, snr_inh], ignore_index=True)
    snr_total.to_csv(SALIDA_SNR, index=False)
    print(f"\nTabla SNR guardada en '{SALIDA_SNR}'")

    # -------------------------------------------------------------------------
    # Gráficos
    # -------------------------------------------------------------------------
    print("\nGenerando gráficos...")
    graficar_grand_average_comparativo(ga_hom, ga_inh)
    graficar_snr_comparacion(snr_hom, snr_inh)

    print("\n[OK] Script 04 finalizado.")
    print("\nPróximo paso:")
    print("  Mirá la tabla de SNR y la figura comparativa.")
    print("  El método con mayor SNR se usará como entrada del Script 05.")
