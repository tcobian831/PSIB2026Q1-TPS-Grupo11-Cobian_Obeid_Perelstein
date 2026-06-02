"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 06: Análisis Estadístico — Control vs Alcohólico
==============================================================================

Propósito:
    Comparar la amplitud del componente c240/VMP entre el grupo control y el
    grupo alcohólico, usando la amplitud medida en la ventana 200–350 ms
    (columna amp_ventana) sobre el promedio HOMOGÉNEO (elegido en el Script 05).

    Se aplican DOS enfoques y se comparan:

    (1) DESCRIPTIVO / INTUITIVO
        - Media ± Desvío Estándar (SD) de la amplitud por grupo, para cada
          canal y condición.
        - Diferencia (control − alcohólico) y razón (control / alcohólico).
        La lectura es directa: si el control tiene media claramente mayor y
        las distribuciones se solapan poco, hay separación entre grupos.

    (2) T-TEST DE STUDENT  (parte inferencial, pedida en el anteproyecto)
        - Compara las medias de los dos grupos y devuelve un p-value por canal
          y condición. p < 0.05 ⇒ la diferencia es estadísticamente significativa.
        - Hipótesis DIRECCIONAL: A_control > A_alcoholic (test de una cola).
        - Se reporta también el t-test de WELCH (no asume varianzas iguales),
          porque los grupos tienen dispersión y tamaño distintos (n=45 vs 77).
          Si Student y Welch coinciden, la conclusión es robusta a ese supuesto.

    Comparación de los dos enfoques:
        ¿El t-test confirma lo que muestra la lectura descriptiva? Se resume
        en cuántas de las 8 celdas (4 canales × 2 condiciones) el control es
        mayor Y el resultado es significativo.

Hipótesis (anteproyecto):
    A_alcoholic(c240, S1)        <  A_control(c240, S1)
    A_alcoholic(c240, S2 nomatch) <  A_control(c240, S2 nomatch)
    con mayor efecto esperado en canales temporo-occipitales derechos.

Entrada:  outputs/eeg_c240_extraido.csv          (del Script 05)
Salida:   outputs/tabla_estadistica.csv          (medias, SD, p-values)
          outputs/figura_estadistica_barras.png  (barras con significancia)

Requisitos: scipy  (pip install scipy)

Uso:
    Correr desde la carpeta scripts/
    python 06_estadistica.py
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES     = ["S1 obj", "S2 nomatch"]

METODO = "inhomogeneo"      # método de promediado elegido en el Script 05
COL_AMP = "amp_ventana"   # amplitud en la ventana 200–350 ms
ALPHA  = 0.05             # umbral de significancia

ENTRADA    = Path("../outputs/eeg_c240_extraido.csv")
SALIDA_CSV = Path("../outputs/tabla_estadistica.csv")

# =============================================================================
# ANÁLISIS POR CANAL Y CONDICIÓN
# =============================================================================

def analizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada canal × condición:
      - Descriptivo: media ± SD por grupo, diferencia y razón.
      - Inferencial: t-test de Student y de Welch (una cola, control > alcohólico).

    Retorna un DataFrame con una fila por canal × condición.
    """
    filas = []
    for canal in CANALES_INTERES:
        for cond in CONDICIONES:
            sub = df[(df["canal"] == canal) & (df["condicion"] == cond)]
            ctrl = sub[sub["grupo"] == "control"][COL_AMP].dropna().values
            alc  = sub[sub["grupo"] == "alcoholic"][COL_AMP].dropna().values
            if len(ctrl) < 2 or len(alc) < 2:
                continue

            mc, sc = ctrl.mean(), ctrl.std(ddof=1)
            ma, sa = alc.mean(),  alc.std(ddof=1)

            # t-test de una cola: H1 = media(control) > media(alcohólico)
            t_st, p_st = stats.ttest_ind(ctrl, alc, equal_var=True,
                                         alternative="greater")
            _,    p_we = stats.ttest_ind(ctrl, alc, equal_var=False,
                                         alternative="greater")

            filas.append({
                "canal": canal, "condicion": cond,
                "n_control": len(ctrl), "n_alcoholic": len(alc),
                "control_media": mc, "control_sd": sc,
                "alcoholic_media": ma, "alcoholic_sd": sa,
                "diferencia": mc - ma,
                "razon": (mc / ma) if abs(ma) > 1e-9 else np.nan,
                "t_student": t_st,
                "p_student": p_st,     # una cola
                "p_welch":   p_we,     # una cola
                "significativo": p_st < ALPHA,
            })
    return pd.DataFrame(filas)


# =============================================================================
# IMPRESIÓN DE TABLAS
# =============================================================================

def imprimir_descriptivo(tab: pd.DataFrame):
    print("\n" + "=" * 80)
    print("(1) DESCRIPTIVO — Media ± SD por grupo  (amplitud en 200–350 ms, µV)")
    print("=" * 80)
    print(f"  {'Canal':<6}{'Condición':<12}{'Control':>16}{'Alcohólico':>16}"
          f"{'Dif.':>9}{'Razón':>8}")
    print("  " + "-" * 67)
    for _, r in tab.iterrows():
        print(f"  {r['canal']:<6}{r['condicion']:<12}"
              f"{r['control_media']:>7.2f} ± {r['control_sd']:>4.2f}"
              f"{r['alcoholic_media']:>8.2f} ± {r['alcoholic_sd']:>4.2f}"
              f"{r['diferencia']:>+9.2f}{r['razon']:>7.2f}x")


def imprimir_inferencial(tab: pd.DataFrame):
    print("\n" + "=" * 80)
    print("(2) T-TEST — H1: media(control) > media(alcohólico)  [una cola]")
    print("    p < 0.05 ⇒ diferencia significativa.  '*' marca significancia.")
    print("=" * 80)
    print(f"  {'Canal':<6}{'Condición':<12}{'t (Student)':>13}"
          f"{'p Student':>12}{'p Welch':>12}{'Signif.':>9}")
    print("  " + "-" * 64)
    for _, r in tab.iterrows():
        marca = "  *" if r["significativo"] else ""
        print(f"  {r['canal']:<6}{r['condicion']:<12}{r['t_student']:>13.2f}"
              f"{r['p_student']:>12.2e}{r['p_welch']:>12.2e}{marca:>9}")


def comparar_enfoques(tab: pd.DataFrame):
    """Resumen: ¿coinciden la lectura descriptiva y el t-test?"""
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE LOS DOS ENFOQUES")
    print("=" * 80)
    n = len(tab)
    ctrl_mayor = (tab["diferencia"] > 0).sum()
    signif     = tab["significativo"].sum()
    ambos      = ((tab["diferencia"] > 0) & tab["significativo"]).sum()
    welch_ok   = (tab["p_welch"] < ALPHA).sum()

    print(f"  Celdas analizadas (4 canales × 2 condiciones): {n}")
    print(f"  Control con media mayor (descriptivo):         {ctrl_mayor}/{n}")
    print(f"  Significativas con t-test de Student:           {signif}/{n}")
    print(f"  Significativas con t-test de Welch:             {welch_ok}/{n}")
    print(f"  Control mayor Y significativo (coinciden):      {ambos}/{n}")

    # Gradiente espacial: ¿T8 es el más débil?
    print("\n  Diferencia control − alcohólico por canal (promedio de condiciones):")
    for canal in CANALES_INTERES:
        d = tab[tab["canal"] == canal]["diferencia"].mean()
        print(f"    {canal:<5}: {d:>+6.2f} µV")
    print("=" * 80)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def graficar_barras(tab: pd.DataFrame, df: pd.DataFrame):
    """
    Barras de la amplitud media por grupo con barras de error (± error estándar
    de la media) y asterisco donde el t-test da p < 0.05.

    Se usa el error estándar de la media (SEM = SD/√n) en las barras: refleja
    la precisión de la media, así que su (no) solapamiento acompaña visualmente
    al resultado del t-test. El SD (dispersión entre sujetos) está en la tabla.
    """
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5))
    if len(CONDICIONES) == 1:
        axes = [axes]
    fig.suptitle(
        f"Amplitud del c240 por grupo — método {METODO}, ventana 200–350 ms\n"
        "Barras de error: ± SEM   |   '*' : diferencia significativa (t-test, p<0.05)",
        fontsize=13
    )
    x = np.arange(len(CANALES_INTERES)); ancho = 0.38

    for ax, cond in zip(axes, CONDICIONES):
        medias_c, sem_c, medias_a, sem_a = [], [], [], []
        for canal in CANALES_INTERES:
            c = df[(df["canal"] == canal) & (df["condicion"] == cond) &
                   (df["grupo"] == "control")][COL_AMP].dropna().values
            a = df[(df["canal"] == canal) & (df["condicion"] == cond) &
                   (df["grupo"] == "alcoholic")][COL_AMP].dropna().values
            medias_c.append(c.mean()); sem_c.append(c.std(ddof=1)/np.sqrt(len(c)))
            medias_a.append(a.mean()); sem_a.append(a.std(ddof=1)/np.sqrt(len(a)))

        ax.bar(x - ancho/2, medias_c, ancho, yerr=sem_c, capsize=4,
               label="Control", color=colores["control"], alpha=0.8)
        ax.bar(x + ancho/2, medias_a, ancho, yerr=sem_a, capsize=4,
               label="Alcohólico", color=colores["alcoholic"], alpha=0.8)

        # asteriscos de significancia
        for xi, canal in enumerate(CANALES_INTERES):
            r = tab[(tab["canal"] == canal) & (tab["condicion"] == cond)]
            if not r.empty and bool(r["significativo"].values[0]):
                ytop = max(medias_c[xi] + sem_c[xi], medias_a[xi] + sem_a[xi])
                ax.text(xi, ytop + 0.25, "*", ha="center",
                        fontsize=16, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condición: {cond}", fontsize=11)
        ax.set_ylabel("Amplitud media del pico (µV)")
        ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_estadistica_barras.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_estadistica_barras.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 06: Análisis Estadístico (Control vs Alcohólico)")
    print("=" * 80)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontró '{ENTRADA}'. Corré primero el Script 05.")

    print(f"\nCargando '{ENTRADA}'...")
    df = pd.read_csv(ENTRADA)
    df = df[df["metodo"] == METODO].copy()
    print(f"  Método: {METODO}   |   Amplitud: {COL_AMP} (ventana 200–350 ms)")
    print(f"  Sujetos control:   {df[df['grupo']=='control']['sujeto'].nunique()}")
    print(f"  Sujetos alcohólicos: {df[df['grupo']=='alcoholic']['sujeto'].nunique()}")

    tab = analizar(df)
    tab.to_csv(SALIDA_CSV, index=False)

    imprimir_descriptivo(tab)
    imprimir_inferencial(tab)
    comparar_enfoques(tab)

    print(f"\nTabla estadística guardada en '{SALIDA_CSV}'")

    print("\nGenerando gráfico...")
    graficar_barras(tab, df)

    print("\n[OK] Script 06 finalizado.")
