"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 04: Promediado de Trials — Homogéneo vs Inhomogéneo  
==============================================================================


  1. IGUALACIÓN DE GRUPOS
     El dataset tiene 77 alcohólicos y 45 controles. Para que el Grand
     Average sea comparable entre grupos (mismo N), se seleccionan
     aleatoriamente 45 alcohólicos con semilla fija (SEED = 42).
     Esto asegura reproducibilidad y elimina el sesgo que introduce tener
     más realizaciones en un grupo que en el otro al calcular la media.

  2. SNR CORREGIDA 

     Definición:
         SNR = Potencia(señal estimada) / Potencia(ruido estimado)

     Implementación:
         - Dividir el ensamble en subensambles PAR e IMPAR.
         - s_par   = promedio de trials pares
         - s_impar = promedio de trials impares
         - Señal estimada: s_avg = (s_par + s_impar) / 2   (promedio total)
         - Ruido estimado: e = (s_par - s_impar) / 2
           (la señal se cancela, queda solo el ruido residual del promedio)
         - SNR = var(s_avg) / var(e)

     Esta es exactamente la SNR del promedio del ensamble:
     numerador = potencia de la señal estimada
     denominador = varianza del error de estimación

  3. COMPARACIÓN HEMISFÉRICA
     Se incluyen los 8 canales (4 derechos + 4 izquierdos) para replicar
     la asimetría hemisférica del paper de Zhang et al. (1997).

Propósito general:
    Obtener el PE individual por sujeto, canal y condición mediante dos
    estrategias de promediado, comparar su SNR y calcular el Grand Average.

    (A) HOMOGÉNEO: promedio aritmético clásico.
            PE(t) = (1/M) * Σ x_i(t)

    (B) INHOMOGÉNEO (caso amplitud variable, varianza de ruido constante):
            x_i = a_i · s + v_i
            w   = a / (aᵀa)
            ŝ_w = Σ w_i · x_i

Entrada:  outputs/eeg_data_preprocesado.parquet   (Script 03)
Salida:   outputs/eeg_PE_homogeneo.parquet
          outputs/eeg_PE_inhomogeneo.parquet
          outputs/eeg_GA_homogeneo.parquet
          outputs/eeg_GA_inhomogeneo.parquet
          outputs/tabla_snr_comparacion.csv
          outputs/figura_GA_comparacion.png
          outputs/figura_snr_comparacion.png
          outputs/sujetos_seleccionados.csv   (registro de qué 45 alcohólicos)

Uso:
    Correr desde la carpeta scripts/
    python 04_promediado.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# El script corre desde cualquier carpeta: anclamos el CWD a la raiz del proyecto
# (donde esta outputs/) para que todas las rutas relativas resuelvan igual.
os.chdir(Path(__file__).resolve().parent.parent)
Path("outputs").mkdir(exist_ok=True)

# Salida UTF-8 robusta: evita UnicodeEncodeError al redirigir/pipear en Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS        = 256   # Hz
N_SAMPLES = 256   # muestras por trial
SEED      = 42    # semilla para reproducibilidad al submuestrear alcohólicos
N_SUJETOS = 45    # cantidad de sujetos por grupo (igual al grupo control)

# Canales — igual que en el Script 03
CANALES_DERECHO   = ["P8",  "PO8",  "T8",  "TP8"]
CANALES_IZQUIERDO = ["P7",  "PO7",  "T7",  "TP7"]
CANALES_INTERES   = CANALES_DERECHO + CANALES_IZQUIERDO

CONDICIONES = ["S1 obj", "S2 nomatch"]
EPS = 1e-12

ENTRADA          = Path("outputs/eeg_data_preprocesado.parquet")
SALIDA_HOM       = Path("outputs/eeg_PE_homogeneo.parquet")
SALIDA_INH       = Path("outputs/eeg_PE_inhomogeneo.parquet")
SALIDA_GA_HOM    = Path("outputs/eeg_GA_homogeneo.parquet")
SALIDA_GA_INH    = Path("outputs/eeg_GA_inhomogeneo.parquet")
SALIDA_SNR       = Path("outputs/tabla_snr_comparacion.csv")
SALIDA_SUJETOS   = Path("outputs/sujetos_seleccionados.csv")

# =============================================================================
# IGUALACIÓN DE GRUPOS
# =============================================================================

def igualar_grupos(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Selecciona aleatoriamente 'n' sujetos del grupo alcohólico para
    equiparar el tamaño con el grupo control.

    Justificación: el Grand Average (GA) es el promedio de los PE
    individuales de todos los sujetos del grupo. Si un grupo tiene más
    sujetos que el otro, su GA tendrá menor varianza (más promediación),
    haciendo la comparación entre grupos injusta. Al igualar N, ambos
    GAs tienen la misma precisión estadística.

    Args:
        df:   DataFrame completo (todos los sujetos y canales)
        n:    cantidad de alcohólicos a retener (= tamaño del grupo control)
        seed: semilla aleatoria para reproducibilidad

    Retorna:
        DataFrame filtrado con n alcohólicos + todos los controles
    """
    sujetos_alc = df[df["grupo"] == "alcoholic"]["sujeto"].unique()
    sujetos_ctrl = df[df["grupo"] == "control"]["sujeto"].unique()

    n_alc  = len(sujetos_alc)
    n_ctrl = len(sujetos_ctrl)

    print(f"\n  Sujetos originales: {n_alc} alcohólicos, {n_ctrl} controles")

    if n_alc <= n:
        print(f"  El grupo alcohólico ya tiene ≤ {n} sujetos. No se submuestrea.")
        return df

    rng = np.random.default_rng(seed)
    seleccionados = rng.choice(sujetos_alc, size=n, replace=False)
    seleccionados_sorted = sorted(seleccionados)

    print(f"  Seleccionados {n} alcohólicos (semilla={seed}).")
    print(f"  Alcohólicos incluidos: {seleccionados_sorted[:5]} ...")

    # Guardar registro de sujetos seleccionados
    pd.DataFrame({
        "sujeto": list(seleccionados_sorted) + list(sujetos_ctrl),
        "grupo": ["alcoholic"] * n + ["control"] * n_ctrl
    }).to_csv(SALIDA_SUJETOS, index=False)
    print(f"  Registro guardado en '{SALIDA_SUJETOS}'")

    mask = (
        (df["grupo"] == "control") |
        ((df["grupo"] == "alcoholic") & df["sujeto"].isin(seleccionados))
    )
    return df[mask].copy()


# =============================================================================
# FUNCIONES DE PROMEDIADO
# =============================================================================

def trials_a_matriz(df_grupo: pd.DataFrame) -> np.ndarray:
    """
    Convierte el DataFrame de un grupo (sujeto×canal×condición) en una
    matriz X de shape (n_trials, N_SAMPLES).
    Solo incluye trials con exactamente N_SAMPLES muestras.
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
    """Promedio aritmético clásico: PE = (1/M) * Σ x_i"""
    return X.mean(axis=0)


def estimar_amplitudes(X: np.ndarray) -> np.ndarray:
    """
    Estima las amplitudes relativas a_i proyectando cada trial sobre el
    promedio ordinario normalizado (diapositiva 11 de clase):

        x̄   = (1/M) Σ x_i
        a_i = <x_i, x̄> / <x̄, x̄>

    Un trial similar al PE promedio → a_i alto.
    Un trial dominado por ruido → a_i bajo.
    """
    x_bar = promedio_homogeneo(X)
    den   = float(np.dot(x_bar, x_bar))
    if den < EPS:
        return None
    return (X @ x_bar) / den


def promedio_inhomogeneo(X: np.ndarray) -> np.ndarray:
    """
    Promedio inhomogéneo — amplitud variable, varianza de ruido constante.

    Pasos (diapositivas de clase):
        1. x̄ = promedio ordinario
        2. a_i = <x_i, x̄> / <x̄, x̄>   (amplitudes relativas)
        3. w   = a / (aᵀa)
        4. ŝ_w = Σ w_i · x_i
    """
    a = estimar_amplitudes(X)
    if a is None:
        return promedio_homogeneo(X)

    aa = float(np.dot(a, a))
    if aa < EPS:
        return promedio_homogeneo(X)

    w = a / aa
    return w @ X


# =============================================================================
# SNR 
# =============================================================================

def snr_ensamble(X: np.ndarray, pesos: np.ndarray = None) -> float:
    """
    SNR del promedio del ensamble, definida como:

        SNR = Potencia(señal estimada) / Potencia(ruido estimado)

    Estimación mediante subensambles par/impar:
        - s_par   = promedio de los trials con índice par
        - s_impar = promedio de los trials con índice impar
        - Señal:  s_avg = (s_par + s_impar) / 2  ≈ PE verdadero
        - Error:  e     = (s_par - s_impar) / 2
          (la parte coherente [señal] se cancela; queda solo ruido)
        - SNR = var(s_avg) / var(e)

    Esta definición viene directamente de las diapositivas de clase:
    SNR = P(señal) / P(ruido), donde estimamos ambas a partir de los
    subensambles.

    Si se proveen pesos (caso inhomogéneo), se aplican proporcionalmente
    a cada subensamble para mantener la consistencia con el método.

    Args:
        X:      matriz (n_trials, N_SAMPLES)
        pesos:  array (n_trials,) opcional. None → homogéneo (pesos iguales)

    Retorna:
        SNR (float) o np.nan si hay muy pocos trials
    """
    M = X.shape[0]
    if M < 4:
        return np.nan

    idx_par   = np.arange(0, M, 2)
    idx_impar = np.arange(1, M, 2)

    if pesos is None:
        # Homogéneo: media aritmética en cada subensamble
        s_par   = X[idx_par].mean(axis=0)
        s_impar = X[idx_impar].mean(axis=0)
    else:
        # Inhomogéneo: promedio ponderado en cada subensamble
        w_par   = pesos[idx_par]
        w_impar = pesos[idx_impar]
        sum_w_par   = w_par.sum()
        sum_w_impar = w_impar.sum()
        if sum_w_par < EPS or sum_w_impar < EPS:
            return np.nan
        s_par   = (w_par   @ X[idx_par])   / sum_w_par
        s_impar = (w_impar @ X[idx_impar]) / sum_w_impar

    # Señal estimada y error de estimación
    s_avg = (s_par + s_impar) / 2.0   # ≈ PE verdadero
    error = (s_par - s_impar) / 2.0   # ruido residual del promedio

    var_señal = float(np.var(s_avg, ddof=1))
    var_error = float(np.var(error, ddof=1))

    if var_error < EPS:
        return np.nan

    return var_señal / var_error


# =============================================================================
# CÁLCULO DE PE INDIVIDUAL Y SNR
# =============================================================================

def calcular_PE_individual(df: pd.DataFrame):
    """
    Para cada combinación sujeto × canal × condición:
        - Calcula el PE homogéneo e inhomogéneo
        - Calcula la SNR de cada método (definición correcta)

    Retorna:
        pe_hom:  DataFrame con los PE homogéneos
        pe_inh:  DataFrame con los PE inhomogéneos
        snr_df:  DataFrame con la SNR por sujeto, canal, condición y método
    """
    grupos = df.groupby(["sujeto", "grupo", "canal", "condicion"])
    n_total = len(grupos)

    pe_hom_lista = []
    pe_inh_lista = []
    snr_lista    = []

    print(f"  Combinaciones sujeto×canal×condición: {n_total}")

    for idx, ((sujeto, grupo, canal, cond), sub) in enumerate(grupos):
        if idx % 200 == 0:
            print(f"  Progreso: {idx}/{n_total}...")

        X = trials_a_matriz(sub)
        if X is None:
            continue

        M = X.shape[0]

        # --- PE homogéneo ---
        pe_h = promedio_homogeneo(X)
        snr_h = snr_ensamble(X, pesos=None)

        # --- PE inhomogéneo ---
        a = estimar_amplitudes(X)
        if a is not None:
            aa = float(np.dot(a, a))
            w  = a / aa if aa >= EPS else np.ones(M) / M
            pe_i  = w @ X
            snr_i = snr_ensamble(X, pesos=w)
        else:
            pe_i  = pe_h.copy()
            snr_i = snr_h

        # Guardar PE homogéneo
        pe_hom_lista.append(pd.DataFrame({
            "sujeto":    sujeto,
            "grupo":     grupo,
            "canal":     canal,
            "condicion": cond,
            "metodo":    "homogeneo",
            "n_trials":  M,
            "muestra":   np.arange(N_SAMPLES),
            "valor_uV":  pe_h,
        }))

        # Guardar PE inhomogéneo
        pe_inh_lista.append(pd.DataFrame({
            "sujeto":    sujeto,
            "grupo":     grupo,
            "canal":     canal,
            "condicion": cond,
            "metodo":    "inhomogeneo",
            "n_trials":  M,
            "muestra":   np.arange(N_SAMPLES),
            "valor_uV":  pe_i,
        }))

        # Guardar SNR
        snr_lista.append({
            "sujeto": sujeto, "grupo": grupo,
            "canal": canal, "condicion": cond,
            "n_trials": M,
            "snr_homogeneo":   snr_h,
            "snr_inhomogeneo": snr_i,
        })

    pe_hom = pd.concat(pe_hom_lista, ignore_index=True)
    pe_inh = pd.concat(pe_inh_lista, ignore_index=True)
    snr_df = pd.DataFrame(snr_lista)
    return pe_hom, pe_inh, snr_df


# =============================================================================
# GRAND AVERAGE
# =============================================================================

def calcular_grand_average(pe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Grand Average: promedio de los PE individuales de todos los sujetos
    de un grupo, por canal y condición.
    También calcula el SEM (error estándar de la media entre sujetos).
    """
    ga = (
        pe_df.groupby(["grupo", "canal", "condicion", "muestra"])["valor_uV"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "grand_avg_uV",
                         "std":  "std_uV",
                         "count": "n_sujetos"})
    )
    ga["sem_uV"] = ga["std_uV"] / np.sqrt(ga["n_sujetos"])
    return ga


# =============================================================================
# COMPARACIÓN DE SNR 
# =============================================================================

def comparar_snr(snr_df: pd.DataFrame):
    """
    Imprime la tabla comparativa de SNR en consola.
    SNR = var(señal estimada) / var(error de estimación par/impar)
    """
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE SNR  —  definición: var(s_avg) / var(error par/impar)")
    print("Columnas: SNR mediana por canal y condición")
    print("=" * 70)

    for metodo in ["snr_homogeneo", "snr_inhomogeneo"]:
        vals = snr_df[metodo].dropna()
        label = "Homogéneo" if "hom" in metodo else "Inhomogéneo"
        print(f"\n{label}:")
        print(f"  Mediana SNR global: {vals.median():.3f}")
        print(f"  Media SNR global:   {vals.mean():.3f}")
        print(f"  N combinaciones:    {len(vals)}")

    print(f"\n{'Canal':<6} {'Hemisferio':<12} {'Condición':<13} "
          f"{'SNR Hom':>10} {'SNR Inh':>10}")
    print("-" * 65)

    for canal in CANALES_INTERES:
        hemisferio = "derecho" if canal in CANALES_DERECHO else "izquierdo"
        for cond in CONDICIONES:
            sub = snr_df[(snr_df["canal"] == canal) &
                         (snr_df["condicion"] == cond)]
            if sub.empty:
                continue
            sh = sub["snr_homogeneo"].median()
            si = sub["snr_inhomogeneo"].median()
            print(f"{canal:<6} {hemisferio:<12} {cond:<13} "
                  f"{sh:>10.3f} {si:>10.3f}")
    print("=" * 70)


# =============================================================================
# VISUALIZACIÓN — figuras divididas por hemisferio
# =============================================================================

def graficar_grand_average(ga_hom: pd.DataFrame, ga_inh: pd.DataFrame,
                           canales=None, hemi_label="",
                           out_file="figura_GA.png"):
    """
    Grand Average por canal y condición para un hemisferio.
    Homogéneo = línea sólida, Inhomogéneo = punteada.
    Control = azul, Alcohólico = rojo.
    """
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    tiempo_ms = np.arange(N_SAMPLES) / FS * 1000
    canales_ref = canales if canales is not None else CANALES_INTERES
    canales_plot = [c for c in canales_ref if c in ga_hom["canal"].unique()]

    fig, axes = plt.subplots(
        len(CONDICIONES), len(canales_plot),
        figsize=(5.5 * len(canales_plot), 4.5 * len(CONDICIONES)),
        sharex=True
    )
    if len(CONDICIONES) == 1:
        axes = [axes]

    hdr = f" — Hemisferio {hemi_label}" if hemi_label else ""
    fig.suptitle(
        f"Grand Average — {'Hemisferio derecho' if 'P8' in canales_plot else 'Hemisferio izquierdo'} — "
        "Hom. (sólida) vs Inh. (punteada)\n"
        "Control (azul) vs Alcohólico (rojo) | "
        "Naranja: ventana c240 (220–260 ms) | Púrpura: ventana c320 (290–340 ms)",
        fontsize=13
    )

    for fila, cond in enumerate(CONDICIONES):
        for col, canal in enumerate(canales_plot):
            ax = axes[fila][col]

            for grupo in ["control", "alcoholic"]:
                color = colores[grupo]
                label_base = grupo.capitalize()

                for ga, estilo, label_sfx in [
                    (ga_hom, "-",  " Hom"),
                    (ga_inh, "--", " Inh")
                ]:
                    d = ga[
                        (ga["grupo"] == grupo) &
                        (ga["canal"] == canal) &
                        (ga["condicion"] == cond)
                    ].sort_values("muestra")
                    if d.empty:
                        continue
                    lbl = (label_base + label_sfx
                           if col == 0 and fila == 0 else "")
                    ax.plot(tiempo_ms, d["grand_avg_uV"].values,
                            color=color, linewidth=1.8, linestyle=estilo,
                            label=lbl)

                # Banda SEM solo para homogeneo (mas legible)
                d_hom = ga_hom[
                    (ga_hom["grupo"] == grupo) &
                    (ga_hom["canal"] == canal) &
                    (ga_hom["condicion"] == cond)
                ].sort_values("muestra")
                if not d_hom.empty:
                    ax.fill_between(
                        tiempo_ms,
                        d_hom["grand_avg_uV"] - d_hom["sem_uV"],
                        d_hom["grand_avg_uV"] + d_hom["sem_uV"],
                        color=color, alpha=0.1
                    )

            # Marcar ventanas c240 y c320
            ax.axvspan(220, 260, alpha=0.12, color="orange",
                       label="c240 (220-260 ms)" 
                       if col == 0 and fila == 0 else "")
            ax.axvspan(290, 340, alpha=0.10, color="purple",
                       label="c320 (290-340 ms)"
                       if col == 0 and fila == 0 else "")
            ax.axvline(0,  color="gray",  linestyle=":", linewidth=0.8)
            ax.axhline(0,  color="black", linewidth=0.5)
            ax.set_title(f"Canal: {canal}\n{cond}", fontsize=10)
            ax.set_xlabel("Tiempo (ms)")
            ax.set_ylabel("Amplitud (uV)")
            ax.grid(True, alpha=0.25)
            if fila == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"outputs/{out_file}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{out_file}'")


def graficar_snr(snr_df: pd.DataFrame, canales=None, hemi_label="",
                 out_file="figura_snr.png"):
    """
    Barras de SNR mediana por canal y condicion para un hemisferio.
    """
    canales_ref = canales if canales is not None else CANALES_INTERES
    canales_plot = [c for c in canales_ref if c in snr_df["canal"].unique()]

    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(8 * len(CONDICIONES), 5))
    if len(CONDICIONES) == 1:
        axes = [axes]

    hdr = f" — Hemisferio {hemi_label}" if hemi_label else ""
    fig.suptitle(
        f"SNR del promedio del ensamble{hdr}\n"
        "SNR = var(senal estimada) / var(error par/impar)  |  "
        "Barras: mediana por canal",
        fontsize=12
    )

    colores_metodo = {"snr_homogeneo": "#94a3b8", "snr_inhomogeneo": "#16a34a"}
    labels_metodo  = {"snr_homogeneo": "Homogeneo", "snr_inhomogeneo": "Inhomogeneo"}

    x = np.arange(len(canales_plot))
    ancho = 0.35

    for ax, cond in zip(axes, CONDICIONES):
        for i, metodo in enumerate(["snr_homogeneo", "snr_inhomogeneo"]):
            medianas = []
            for canal in canales_plot:
                vals = snr_df[
                    (snr_df["canal"] == canal) &
                    (snr_df["condicion"] == cond)
                ][metodo].dropna().values
                medianas.append(np.median(vals) if len(vals) else np.nan)

            ax.bar(x + i * ancho, medianas, ancho,
                   label=labels_metodo[metodo],
                   color=colores_metodo[metodo], alpha=0.85)

        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(canales_plot, rotation=30)
        ax.set_title(f"Condicion: {cond}", fontsize=11)
        ax.set_ylabel("SNR mediana")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"outputs/{out_file}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{out_file}'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TPS -- Potenciales Evocados Visuales en Alcoholismo")
    print("Script 04: Promediado  [v2 -- PE completo + GA igualado]")
    print("=" * 70)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA}'.\n"
            "Corre primero el Script 03."
        )

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_parquet(ENTRADA)
    print(f"  {len(df):,} filas cargadas")
    print(f"  Sujetos originales: "
          f"{df[df['grupo']=='alcoholic']['sujeto'].nunique()} alcoholicos, "
          f"{df[df['grupo']=='control']['sujeto'].nunique()} controles")

    # Verificar canales disponibles
    canales_disponibles = df["canal"].unique().tolist()
    canales_en_datos = [c for c in CANALES_INTERES if c in canales_disponibles]
    canales_faltantes = [c for c in CANALES_INTERES if c not in canales_disponibles]
    if canales_faltantes:
        print(f"\n  ADVERTENCIA: canales no encontrados: {canales_faltantes}")
        print(f"  Continuando con: {canales_en_datos}")

    # -------------------------------------------------------------------------
    # Calculo de PE individual + SNR  (TODOS los sujetos: 77 alc + 45 ctrl)
    # -------------------------------------------------------------------------
   
    print("\nCalculando PE individuales (homogeneo e inhomogeneo) + SNR...")
    print("  (TODOS los sujetos -- 77 alc + 45 ctrl -- para inferencia completa)")
    pe_hom, pe_inh, snr_df = calcular_PE_individual(df)
    print(f"\n  PE homogeneo:    {len(pe_hom):,} filas")
    print(f"  PE inhomogeneo:  {len(pe_inh):,} filas")
    print(f"  Sujetos en PE: "
          f"{pe_hom[pe_hom['grupo']=='alcoholic']['sujeto'].nunique()} alc + "
          f"{pe_hom[pe_hom['grupo']=='control']['sujeto'].nunique()} ctrl")

    # Guardar PE individual (muestra COMPLETA para Scripts 05/06)
    print("\nGuardando PE individuales (muestra completa)...")
    pe_hom.to_parquet(SALIDA_HOM, index=False)
    pe_inh.to_parquet(SALIDA_INH, index=False)
    print(f"  PE homogeneo   -> '{SALIDA_HOM}'")
    print(f"  PE inhomogeneo -> '{SALIDA_INH}'")

    # -------------------------------------------------------------------------
    # Igualacion de grupos (SOLO para Grand Average)
    # -------------------------------------------------------------------------
    print(f"\nIgualando grupos a {N_SUJETOS} sujetos por grupo (SOLO para GA)...")
    df_sub = igualar_grupos(df, N_SUJETOS, SEED)
    sujetos_ga = set(df_sub["sujeto"].unique())
    print(f"  Sujetos para GA: "
          f"{df_sub[df_sub['grupo']=='alcoholic']['sujeto'].nunique()} alc + "
          f"{df_sub[df_sub['grupo']=='control']['sujeto'].nunique()} ctrl")

    pe_hom_ga = pe_hom[pe_hom["sujeto"].isin(sujetos_ga)]
    pe_inh_ga = pe_inh[pe_inh["sujeto"].isin(sujetos_ga)]
    snr_df_ga = snr_df[snr_df["sujeto"].isin(sujetos_ga)]

    # -------------------------------------------------------------------------
    # Grand Average (sobre muestra igualada 45+45)
    # -------------------------------------------------------------------------
    print("\nCalculando Grand Average (sobre 45+45)...")
    ga_hom = calcular_grand_average(pe_hom_ga)
    ga_inh = calcular_grand_average(pe_inh_ga)

    ga_hom.to_parquet(SALIDA_GA_HOM, index=False)
    ga_inh.to_parquet(SALIDA_GA_INH, index=False)
    snr_df_ga.to_csv(SALIDA_SNR, index=False)
    print(f"  GA homogeneo   -> '{SALIDA_GA_HOM}'")
    print(f"  GA inhomogeneo -> '{SALIDA_GA_INH}'")
    print(f"  Tabla SNR      -> '{SALIDA_SNR}'")

    # -------------------------------------------------------------------------
    # Comparacion de SNR en consola
    # -------------------------------------------------------------------------
    comparar_snr(snr_df_ga)

    # -------------------------------------------------------------------------
    # Graficos -- divididos por hemisferio
    # -------------------------------------------------------------------------
    print("\nGenerando graficos (por hemisferio)...")
    for canales, label, sufijo in [
        (CANALES_DERECHO,   "derecho",   "derecho"),
        (CANALES_IZQUIERDO, "izquierdo", "izquierdo"),
    ]:
        graficar_grand_average(
            ga_hom, ga_inh, canales=canales, hemi_label=label,
            out_file=f"figura_GA_{sufijo}.png")
        graficar_snr(
            snr_df_ga, canales=canales, hemi_label=label,
            out_file=f"figura_snr_{sufijo}.png")

    print("\n[OK] Script 04 finalizado.")
    print("\nProximo paso:")
    print("  Revisa las figuras del Grand Average y la tabla de SNR.")
    print("  El PE individual (muestra completa) esta listo para el Script 05.")
