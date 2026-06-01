"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 04: Promediado de Trials — Homogéneo vs Inhomogéneo
==============================================================================

Propósito:
    Obtener el Potencial Evocado (PE) individual por sujeto, canal y
    condición mediante dos estrategias de promediado, y comparar cuál
    estima mejor el PE.

    (A) HOMOGÉNEO: promedio aritmético clásico. Todos los trials pesan igual.
            PE(t) = (1/M) * Σ x_i(t)

    (B) INHOMOGÉNEO (caso s[n] de amplitud variable, ruido de varianza
        constante — visto en clase):
            Modelo:  X = s·aᵀ + V,   con R_v = σ²·I  (ruido homocedástico)
                     cada trial:  x_i = a_i·s + v_i
            El estimador del promedio ponderado es insesgado (E[ŝ_w]=s) si
            los pesos son:
                     w = a / (aᵀa)          (diapositivas 9-10)
            donde a es el vector de amplitudes de cada trial.

        Receta (diapositiva 12):
            1. Promedio ordinario del ensamble  x̄ = (1/M) Σ x_i
            2. Estimar las amplitudes a_i proyectando cada trial sobre el
               promedio ordinario.  La diapositiva 11 asume sᵀs = 1, por lo
               que la proyección se hace sobre el promedio NORMALIZADO; en la
               práctica esto equivale a:
                     a_i = <x_i, x̄> / <x̄, x̄>
               con lo que las a_i quedan adimensionales y centradas en 1
               (amplitudes RELATIVAS), y el promedio ponderado conserva la
               escala en µV (comparable con el homogéneo).
            3. Pesos:  w = a / (aᵀa)         (aᵀa = Σ a_i²)
            4. Promedio ponderado:  ŝ_w = Σ w_i · x_i

        Nota de implementación: la proyección se normaliza por <x̄,x̄> y NO se
        truncan amplitudes negativas (la teórica no lo indica; truncar sesga
        el estimador). Si se omite la normalización, ŝ_w sale dividido por
        ||x̄||² y la escala se colapsa sobre datos reales en µV.

Comparación entre métodos (SNR por método ±, "plus-minus reference"):
    No tenemos baseline pre-estímulo confiable, así que NO usamos
    var(señal)/var(baseline). En su lugar estimamos el ruido residual del
    promedio con el método ±:
        - promedio normal  →  señal + ruido residual
        - promedio con signos alternados  →  la señal se cancela, queda ruido
        SNR = var(promedio) / var(promedio con signos alternados)
    Es una métrica justa entre métodos porque mide el ruido que realmente
    queda en cada promedio, no la varianza de la propia señal.

Entrada:  outputs/eeg_data_preprocesado.parquet   (TRIALS crudos, generado
          por Script 03; columnas: sujeto, grupo, canal, condicion,
          trial_num, muestra, valor_uV)
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

EPS = 1e-12       # estabilidad numérica

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

    Retorna:
        matriz numpy (n_trials, N_SAMPLES) o None si no hay >= 2 trials válidos
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
    Promedio aritmético clásico: PE = (1/M) * Σ x_i

    Args:
        X: matriz (n_trials, N_SAMPLES)
    Retorna:
        array 1D (N_SAMPLES,) con el PE promediado
    """
    return X.mean(axis=0)


def estimar_amplitudes(X: np.ndarray) -> np.ndarray:
    """
    Estima el vector de amplitudes relativas a_i de cada trial proyectando
    cada realización sobre el promedio ordinario del ensamble (diapositiva 11).

        x̄   = (1/M) Σ x_i               (promedio ordinario)
        a_i = <x_i, x̄> / <x̄, x̄>        (proyección normalizada)

    La normalización por <x̄,x̄> hace que las a_i sean adimensionales y que su
    media sea ≈ 1 (amplitudes RELATIVAS al trial típico). Un trial parecido al
    PE promedio → a_i alto; un trial dominado por ruido → a_i bajo.

    Retorna:
        a: array (n_trials,) con las amplitudes relativas, o None si x̄ ≈ 0
    """
    x_bar = promedio_homogeneo(X)
    den   = float(np.dot(x_bar, x_bar))
    if den < EPS:
        return None
    return (X @ x_bar) / den


def promedio_inhomogeneo(X: np.ndarray) -> np.ndarray:
    """
    Promedio inhomogéneo — caso s[n] de amplitud variable, ruido constante.

    Pasos (diapositivas 9-12):
        1. x̄ = promedio ordinario.
        2. a_i = <x_i, x̄> / <x̄, x̄>          (amplitudes relativas)
        3. w   = a / (aᵀa),   con  aᵀa = Σ a_i²
        4. ŝ_w = Σ w_i · x_i

    El resultado queda en µV, en la misma escala que el promedio homogéneo,
    por lo que ambos picos son directamente comparables en el Script 05.

    Args:
        X: matriz (n_trials, N_SAMPLES)
    Retorna:
        array 1D (N_SAMPLES,) con el PE ponderado
    """
    a = estimar_amplitudes(X)
    if a is None:
        return promedio_homogeneo(X)          # x̄ ≈ 0 → no hay señal estimable

    aa = float(np.dot(a, a))                  # aᵀa = Σ a_i²
    if aa < EPS:
        return promedio_homogeneo(X)

    w = a / aa                                # paso 3: w = a / (aᵀa)
    return w @ X                              # paso 4: Σ w_i · x_i


def snr_pm(X: np.ndarray, pesos: np.ndarray) -> float:
    """
    SNR del promedio por el método ± (plus-minus reference).

    - Promedio ponderado normal:        contiene señal + ruido residual.
    - Promedio con signos alternados:   la señal coherente se cancela y queda
                                        solo el ruido residual.
        SNR = var(promedio) / var(promedio_alternado)

    No requiere baseline pre-estímulo y es justa entre métodos: mide el ruido
    que realmente queda en cada promedio.

    Args:
        X:      matriz (n_trials, N_SAMPLES)
        pesos:  vector (n_trials,) de pesos del método (1's para homogéneo,
                amplitudes a_i para inhomogéneo). Se renormalizan a suma 1.
    Retorna:
        SNR (float) o np.nan
    """
    s = pesos.sum()
    if abs(s) < EPS:
        return np.nan
    w = pesos / s
    prom   = w @ X
    signos = (-1.0) ** np.arange(X.shape[0])
    ruido  = (w * signos) @ X
    var_n  = np.var(ruido)
    return float(np.var(prom) / var_n) if var_n > EPS else np.nan


# =============================================================================
# CÁLCULO DE PE INDIVIDUAL (+ SNR por método)
# =============================================================================

def calcular_PE_individual(df: pd.DataFrame) -> tuple:
    """
    Para cada combinación (sujeto, grupo, canal, condición):
        1. Arma la matriz de trials X.
        2. Calcula PE homogéneo e inhomogéneo.
        3. Calcula SNR± de cada método.

    Retorna:
        pe_hom_df: DataFrame con PE homogéneo (formato largo, una fila/muestra)
        pe_inh_df: DataFrame con PE inhomogéneo
        snr_df:    DataFrame con SNR de ambos métodos (una fila por método)
    """
    grupos = df.groupby(["sujeto", "grupo", "canal", "condicion"])
    n_total = len(grupos)
    print(f"  Combinaciones sujeto×canal×condición: {n_total:,}")

    filas_hom = []
    filas_inh = []
    filas_snr = []

    for idx, ((sujeto, grupo, canal, cond), sub) in enumerate(grupos):
        if idx % 2000 == 0:
            print(f"  Progreso: {idx:,}/{n_total:,}...")

        X = trials_a_matriz(sub)
        if X is None:
            continue

        n_trials = X.shape[0]
        pe_hom   = promedio_homogeneo(X)
        pe_inh   = promedio_inhomogeneo(X)
        muestras = np.arange(N_SAMPLES)

        base = {
            "sujeto":    sujeto,
            "grupo":     grupo,
            "canal":     canal,
            "condicion": cond,
            "n_trials":  n_trials,
        }

        for m, v_h, v_i in zip(muestras, pe_hom, pe_inh):
            filas_hom.append({**base, "muestra": int(m), "PE_uV": float(v_h)})
            filas_inh.append({**base, "muestra": int(m), "PE_uV": float(v_i)})

        # SNR de ambos métodos sobre los MISMOS trials
        a = estimar_amplitudes(X)
        w_inh = a if a is not None else np.ones(n_trials)
        filas_snr.append({**base, "metodo": "homogeneo",
                          "snr": snr_pm(X, np.ones(n_trials))})
        filas_snr.append({**base, "metodo": "inhomogeneo",
                          "snr": snr_pm(X, w_inh)})

    return (pd.DataFrame(filas_hom),
            pd.DataFrame(filas_inh),
            pd.DataFrame(filas_snr))


# =============================================================================
# GRAND AVERAGE
# =============================================================================

def calcular_grand_average(pe_ind: pd.DataFrame) -> pd.DataFrame:
    """
    Promedia los PE individuales de todos los sujetos del mismo grupo y calcula
    el error estándar (SEM) entre sujetos para graficar bandas de variabilidad.

    Retorna columnas:
        grupo, canal, condicion, muestra, grand_avg_uV, std_uV, n_sujetos, sem_uV
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

def comparar_snr(snr_df: pd.DataFrame):
    """Imprime resumen comparativo de SNR± entre métodos."""
    print("\n" + "=" * 64)
    print("COMPARACIÓN DE SNR (método ±)  —  SNR = var(promedio)/var(±)")
    print("=" * 64)

    for metodo in ["homogeneo", "inhomogeneo"]:
        vals = snr_df[snr_df["metodo"] == metodo]["snr"].dropna()
        print(f"\n{metodo.capitalize()}:")
        print(f"  Mediana SNR: {vals.median():.3f}")
        print(f"  Media SNR:   {vals.mean():.3f}")
        print(f"  N:           {len(vals)}")

    print(f"\n{'Canal':<5} {'Condición':<12} "
          f"{'SNR Hom':>10} {'SNR Inh':>10} {'Ganador':>10}")
    print("-" * 52)

    g = (snr_df.groupby(["canal", "condicion", "metodo"])["snr"]
         .median().unstack("metodo"))
    for canal in CANALES_INTERES:
        for cond in CONDICIONES:
            try:
                sh = g.loc[(canal, cond), "homogeneo"]
                si = g.loc[(canal, cond), "inhomogeneo"]
            except KeyError:
                continue
            ganador = "Inh" if si > sh else "Hom"
            print(f"{canal:<5} {cond:<12} {sh:>10.3f} {si:>10.3f} {ganador:>10}")
    print("=" * 64)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def graficar_grand_average_comparativo(ga_hom: pd.DataFrame,
                                        ga_inh: pd.DataFrame):
    """
    Grand Average de ambos métodos superpuestos por canal y condición.
    Homogéneo = línea sólida, Inhomogéneo = línea punteada.
    """
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharex=True, sharey=True
    )
    fig.suptitle(
        "Grand Average — Homogéneo (sólida) vs Inhomogéneo (punteada)\n"
        "Control (azul) vs Alcohólico (rojo)",
        fontsize=13
    )

    tiempo_ms = np.arange(N_SAMPLES) / FS * 1000

    for fila, cond in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]

            for grupo in ["control", "alcoholic"]:
                color = colores[grupo]

                d_hom = ga_hom[
                    (ga_hom["grupo"] == grupo) &
                    (ga_hom["canal"] == canal) &
                    (ga_hom["condicion"] == cond)
                ].sort_values("muestra")
                if not d_hom.empty:
                    ax.plot(tiempo_ms, d_hom["grand_avg_uV"].values,
                            color=color, linewidth=1.8, linestyle="-",
                            label=(f"{grupo.capitalize()} Hom"
                                   if col == 0 and fila == 0 else ""))
                    ax.fill_between(
                        tiempo_ms,
                        d_hom["grand_avg_uV"] - d_hom["sem_uV"],
                        d_hom["grand_avg_uV"] + d_hom["sem_uV"],
                        color=color, alpha=0.1
                    )

                d_inh = ga_inh[
                    (ga_inh["grupo"] == grupo) &
                    (ga_inh["canal"] == canal) &
                    (ga_inh["condicion"] == cond)
                ].sort_values("muestra")
                if not d_inh.empty:
                    ax.plot(tiempo_ms, d_inh["grand_avg_uV"].values,
                            color=color, linewidth=1.8, linestyle="--",
                            label=(f"{grupo.capitalize()} Inh"
                                   if col == 0 and fila == 0 else ""))

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


def graficar_snr_comparacion(snr_df: pd.DataFrame):
    """Barras del SN± mediano por método, agrupadas por canal y condición."""
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5))
    if len(CONDICIONES) == 1:
        axes = [axes]

    fig.suptitle(
        "Comparación de SNR (método ±): Homogéneo vs Inhomogéneo\n"
        "SNR = var(promedio) / var(promedio con signos alternados)",
        fontsize=13
    )

    colores = {"homogeneo": "#94a3b8", "inhomogeneo": "#16a34a"}
    x = np.arange(len(CANALES_INTERES))
    ancho = 0.35

    for ax, cond in zip(axes, CONDICIONES):
        for i, metodo in enumerate(["homogeneo", "inhomogeneo"]):
            medianas = []
            for canal in CANALES_INTERES:
                vals = snr_df[
                    (snr_df["canal"] == canal) &
                    (snr_df["condicion"] == cond) &
                    (snr_df["metodo"] == metodo)
                ]["snr"].dropna().values
                medianas.append(np.median(vals) if len(vals) else np.nan)
            ax.bar(x + i * ancho, medianas, ancho,
                   label=metodo.capitalize(),
                   color=colores[metodo], alpha=0.85)

        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condición: {cond}", fontsize=11)
        ax.set_ylabel("SNR± mediano")
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

    print("=" * 64)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 04: Promediado Homogéneo vs Inhomogéneo")
    print("=" * 64)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'.\n"
            "El Script 04 consume los TRIALS crudos preprocesados "
            "(columnas trial_num, muestra, valor_uV) del Script 03."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas")
    print(f"  Sujetos: {df['sujeto'].nunique()} "
          f"({df[df['grupo']=='alcoholic']['sujeto'].nunique()} alcohólicos, "
          f"{df[df['grupo']=='control']['sujeto'].nunique()} controles)")

    print("\nCalculando PE individuales (homogéneo e inhomogéneo) + SNR...")
    pe_hom, pe_inh, snr_df = calcular_PE_individual(df)
    print(f"\n  PE homogéneo:    {len(pe_hom):,} filas")
    print(f"  PE inhomogéneo:  {len(pe_inh):,} filas")

    print("\nCalculando Grand Average...")
    ga_hom = calcular_grand_average(pe_hom)
    ga_inh = calcular_grand_average(pe_inh)

    print("\nGuardando archivos...")
    pe_hom.to_parquet(SALIDA_HOM, index=False)
    pe_inh.to_parquet(SALIDA_INH, index=False)
    ga_hom.to_parquet(SALIDA_GA_HOM, index=False)
    ga_inh.to_parquet(SALIDA_GA_INH, index=False)
    snr_df.to_csv(SALIDA_SNR, index=False)
    print(f"  PE homogéneo   → '{SALIDA_HOM}'")
    print(f"  PE inhomogéneo → '{SALIDA_INH}'")
    print(f"  Tabla SNR      → '{SALIDA_SNR}'")

    comparar_snr(snr_df)

    print("\nGenerando gráficos...")
    graficar_grand_average_comparativo(ga_hom, ga_inh)
    graficar_snr_comparacion(snr_df)

    print("\n[OK] Script 04 finalizado.")
    print("\nPróximo paso:")
    print("  Revisá la figura del Grand Average y la tabla de SNR.")
    print("  El método con mayor SNR se usa como entrada del Script 05.")
