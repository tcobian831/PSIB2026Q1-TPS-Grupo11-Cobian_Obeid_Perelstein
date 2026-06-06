"""
==============================================================================
TPS - Procesamiento de Senales Biomedicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobian, Obeid, Perelstein

Script 06: Analisis Estadistico  [v3 — corregido]
==============================================================================

Proposito:
    Analizar el componente c240/VMP sobre los 8 canales (4 derechos del paper
    + 4 homologos izquierdos), con las siguientes correcciones metodologicas:

    METODO: se lidera con el HOMOGENEO como analisis principal. El inhomogeneo
    se reporta como analisis secundario (su mayor SNR puede ser circular).

    VENTANAS:
      - PRIMARIA: 220-260 ms (c240 de Zhang et al. 1997). Es la medicion
        oficial, anclada al marco teorico (no elegida mirando los datos).
      - SECUNDARIA: 290-340 ms (positividad tardia / posible c320 de Zhang).
        En nuestros datos el pico del GA aparece ~300-340 ms. Esto
        probablemente corresponde al c320 de Zhang, NO al c240 desplazado.

    METRICA PRINCIPAL: media de la senal en la ventana (no maximo positivo).
    El maximo positivo tiene sesgo al alza y se rompe cuando la senal es toda
    negativa (produce medias negativas absurdas). La media es estable.

    TAMANO DE EFECTO: diferencia de medias + Cohen's d (no razon, que explota
    con denominadores ~0 o negativos).

    MUESTRA: 77 alcoholicos vs 45 controles (muestra COMPLETA). El
    submuestreo a 45+45 del Script 04 se aplica solo al Grand Average.
    Welch maneja N desigual sin problema.

    LATERALIZACION: en los datos el efecto es BILATERAL — P7/PO7 muestran
    separaciones control vs alcoholico iguales o mayores que P8/PO8. NO esta
    lateralizado a derecha como en Zhang. El analisis de lateralizacion se
    mantiene pero el resultado se reporta tal cual es.

    Trabajo futuro (no implementado): ANOVA mixto grupo x hemisferio.

    CORRECCION POR COMPARACIONES MULTIPLES: FDR de Benjamini-Hochberg
    aplicado por familia de tests.

Bloques:
    1. Amplitud media control vs alcoholico (c240 + c320)
    2. AUC control vs alcoholico (c240 + c320)
    3. Latencia (direccion INVERSA: H1 alcoholico mas lento)
    4. Lateralizacion: pareado R vs L + LI entre grupos
    5. Analisis secundario: inhomogeneo (robustez)

Hipotesis (anteproyecto):
    A_alcoholic(c240) < A_control(c240)  en ambas condiciones.

Entrada:  outputs/eeg_c240_extraido.csv  (del Script 05, 77+45 sujetos)
Salida:   outputs/tabla_estadistica.csv
          outputs/tabla_lateralizacion.csv
          outputs/figura_barras_derecho_c240.png
          outputs/figura_barras_izquierdo_c240.png
          outputs/figura_lateralizacion.png

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
# CONFIGURACION
# =============================================================================

CANALES_DERECHO    = ["P8",  "PO8",  "T8",  "TP8"]
CANALES_IZQUIERDO  = ["P7",  "PO7",  "T7",  "TP7"]
CANALES_INTERES    = CANALES_DERECHO + CANALES_IZQUIERDO
PARES_HEMISFERICOS = [("P8", "P7"), ("PO8", "PO7"), ("T8", "T7"), ("TP8", "TP7")]
CONDICIONES        = ["S1 obj", "S2 nomatch"]

# Metodo principal: HOMOGENEO (inhomogeneo como secundario)
METODO_PRINCIPAL   = "homogeneo"
METODO_SECUNDARIO  = "inhomogeneo"

# Metricas
COL_MEDIA_C240 = "media_c240"    # media en 220-260 ms (principal)
COL_MEDIA_C320 = "media_c320"    # media en 290-340 ms (secundaria)
COL_AUC_C240   = "auc_c240"
COL_AUC_C320   = "auc_c320"
COL_LAT_C240   = "lat_max_c240"
COL_LAT_C320   = "lat_max_c320"

ALPHA = 0.05

ENTRADA        = Path("../outputs/eeg_c240_extraido.csv")
SALIDA_CSV     = Path("../outputs/tabla_estadistica.csv")
SALIDA_LATERAL = Path("../outputs/tabla_lateralizacion.csv")


# =============================================================================
# COHEN'S D
# =============================================================================

def cohens_d(x, y):
    """Cohen's d (pooled SD). d > 0 => x tiene media mayor."""
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (x.mean() - y.mean()) / sp if sp > 1e-12 else np.nan


# =============================================================================
# FDR (Benjamini-Hochberg)
# =============================================================================

def bh_fdr(pvals) -> np.ndarray:
    """p-values ajustados por FDR de Benjamini-Hochberg. NaN preservados."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    pm = p[mask]
    n = len(pm)
    if n == 0:
        return out
    order = np.argsort(pm)
    ranked = np.empty(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        val = pm[idx] * n / (i + 1)
        cummin = min(cummin, val)
        ranked[idx] = min(cummin, 1.0)
    out[mask] = ranked
    return out


# =============================================================================
# BLOQUE 1-3: CONTRASTE ENTRE GRUPOS
# =============================================================================

def analizar_grupos(df, col, direccion, ventana_label):
    """
    T-test de Welch por canal x condicion.

    direccion = "control>alc"  => H1: control > alcoholico
    direccion = "alc>control"  => H1: alcoholico > control (latencia)
    """
    filas = []
    for canal in CANALES_INTERES:
        for cond in CONDICIONES:
            sub = df[(df["canal"] == canal) & (df["condicion"] == cond)]
            ctrl = sub[sub["grupo"] == "control"][col].dropna().values
            alc  = sub[sub["grupo"] == "alcoholic"][col].dropna().values
            if len(ctrl) < 2 or len(alc) < 2:
                continue

            mc, sc = ctrl.mean(), ctrl.std(ddof=1)
            ma, sa = alc.mean(),  alc.std(ddof=1)

            if direccion == "control>alc":
                t_w, p_w = stats.ttest_ind(ctrl, alc, equal_var=False,
                                           alternative="greater")
                d = cohens_d(ctrl, alc)
            else:
                t_w, p_w = stats.ttest_ind(alc, ctrl, equal_var=False,
                                           alternative="greater")
                d = cohens_d(alc, ctrl)

            filas.append({
                "ventana": ventana_label, "metrica": col,
                "canal": canal, "condicion": cond,
                "hemisferio": "derecho" if canal in CANALES_DERECHO else "izquierdo",
                "n_control": len(ctrl), "n_alcoholic": len(alc),
                "control_media": mc, "control_sd": sc,
                "alcoholic_media": ma, "alcoholic_sd": sa,
                "diferencia": mc - ma,
                "cohens_d": d,
                "t_welch": t_w, "p_welch": p_w,
            })

    tab = pd.DataFrame(filas)
    if len(tab):
        tab["p_fdr"] = bh_fdr(tab["p_welch"].values)
        tab["sig"]     = tab["p_welch"] < ALPHA
        tab["sig_fdr"] = tab["p_fdr"] < ALPHA
    return tab


# =============================================================================
# BLOQUE 4: LATERALIZACION
# =============================================================================

def tabla_lateralizacion(df, col=COL_MEDIA_C240):
    """
    (a) Pareado R vs L (ttest_rel, H1: R > L).
    (b) LI = R - L entre grupos (ttest_ind Welch, H1: LI_control > LI_alc).
    """
    filas = []
    for (R, L) in PARES_HEMISFERICOS:
        for cond in CONDICIONES:
            sub = df[df["condicion"] == cond]
            piv = (sub.pivot_table(index=["sujeto", "grupo"],
                                   columns="canal", values=col)
                      .reset_index())
            if R not in piv.columns or L not in piv.columns:
                continue
            piv = piv.dropna(subset=[R, L])
            if len(piv) < 2:
                continue

            li = piv[R] - piv[L]

            # (a) pareado R vs L
            t_p, p_p = stats.ttest_rel(piv[R].values, piv[L].values,
                                       alternative="greater")
            fila = {
                "par": f"{R}-{L}", "condicion": cond, "n": len(piv),
                "media_R": piv[R].mean(), "media_L": piv[L].mean(),
                "media_LI": li.mean(),
                "t_RvsL": t_p, "p_RvsL": p_p,
            }

            # (b) LI entre grupos
            li_c = li[piv["grupo"] == "control"].values
            li_a = li[piv["grupo"] == "alcoholic"].values
            if len(li_c) >= 2 and len(li_a) >= 2:
                t_g, p_g = stats.ttest_ind(li_c, li_a, equal_var=False,
                                           alternative="greater")
                fila.update({
                    "n_ctrl": len(li_c), "n_alc": len(li_a),
                    "LI_ctrl": li_c.mean(), "LI_alc": li_a.mean(),
                    "dif_LI": li_c.mean() - li_a.mean(),
                    "d_LI": cohens_d(li_c, li_a),
                    "t_LI": t_g, "p_LI": p_g,
                })
            filas.append(fila)

    tab = pd.DataFrame(filas)
    if len(tab):
        tab["p_RvsL_fdr"] = bh_fdr(tab["p_RvsL"].values)
        if "p_LI" in tab.columns:
            tab["p_LI_fdr"] = bh_fdr(tab["p_LI"].values)
    return tab


# =============================================================================
# IMPRESION
# =============================================================================

def imprimir_grupos(tab, titulo, h1):
    print(f"\n{'='*88}")
    print(f"{titulo}")
    print(f"H1: {h1}  [una cola, Welch]")
    print(f"'*' = p<0.05 crudo | '+' = p<0.05 tras FDR")
    print(f"{'='*88}")
    print(f"  {'Canal':<6}{'Cond.':<12}{'Control':>14}{'Alcoholico':>14}"
          f"{'Dif.':>9}{'d':>7}{'t':>8}{'p':>11}{'p(FDR)':>11}")
    print("  " + "-" * 85)
    for _, r in tab.iterrows():
        marca = "+" if r.get("sig_fdr", False) else ("*" if r.get("sig", False) else " ")
        print(f"  {r['canal']:<6}{r['condicion']:<12}"
              f"{r['control_media']:>+7.2f} +- {r['control_sd']:>4.2f}"
              f"{r['alcoholic_media']:>+7.2f} +- {r['alcoholic_sd']:>4.2f}"
              f"{r['diferencia']:>+9.2f}{r['cohens_d']:>7.2f}"
              f"{r['t_welch']:>8.2f}{r['p_welch']:>11.2e}"
              f"{r['p_fdr']:>11.2e} {marca}")


def imprimir_lateralizacion(tab):
    print(f"\n{'='*88}")
    print("LATERALIZACION (hemisferio derecho vs izquierdo)")
    print("NOTA: en estos datos el efecto control>alcoholico es BILATERAL.")
    print("Los canales izquierdos (P7, PO7) muestran separaciones iguales o")
    print("mayores que sus homologos derechos. Esto difiere de Zhang et al.,")
    print("quienes reportaron dominancia derecha.")
    print(f"{'='*88}")

    print("\n  (a) PAREADO R vs L — H1: R > L")
    print(f"      {'Par':<10}{'Cond.':<12}{'R':>8}{'L':>8}"
          f"{'LI':>8}{'t':>7}{'p':>11}{'p(FDR)':>11}")
    print("      " + "-" * 67)
    for _, r in tab.iterrows():
        marca = "+" if r.get("p_RvsL_fdr", 1) < ALPHA else (
            "*" if r["p_RvsL"] < ALPHA else " ")
        print(f"      {r['par']:<10}{r['condicion']:<12}"
              f"{r['media_R']:>+8.2f}{r['media_L']:>+8.2f}"
              f"{r['media_LI']:>+8.2f}{r['t_RvsL']:>7.2f}"
              f"{r['p_RvsL']:>11.2e}{r.get('p_RvsL_fdr',np.nan):>11.2e} {marca}")

    if "p_LI" not in tab.columns:
        return
    print("\n  (b) LI = R - L ENTRE GRUPOS — H1: LI_control > LI_alcoholico")
    print(f"      {'Par':<10}{'Cond.':<12}{'LI ctrl':>9}{'LI alc':>9}"
          f"{'dif':>8}{'d':>7}{'t':>7}{'p':>11}{'p(FDR)':>11}")
    print("      " + "-" * 75)
    for _, r in tab.iterrows():
        if pd.isna(r.get("p_LI", np.nan)):
            continue
        marca = "+" if r.get("p_LI_fdr", 1) < ALPHA else (
            "*" if r["p_LI"] < ALPHA else " ")
        print(f"      {r['par']:<10}{r['condicion']:<12}"
              f"{r['LI_ctrl']:>+9.2f}{r['LI_alc']:>+9.2f}"
              f"{r['dif_LI']:>+8.2f}{r['d_LI']:>7.2f}"
              f"{r['t_LI']:>7.2f}{r['p_LI']:>11.2e}"
              f"{r.get('p_LI_fdr',np.nan):>11.2e} {marca}")


def resumen_bloque(tab, label):
    """Resumen rapido de un bloque."""
    n = len(tab)
    if n == 0:
        return
    print(f"\n  RESUMEN {label}:")
    print(f"    Celdas: {n} | sig crudo: {tab['sig'].sum()}/{n}"
          f" | sig FDR: {tab['sig_fdr'].sum()}/{n}")
    for hemi in ["derecho", "izquierdo"]:
        sub = tab[tab["hemisferio"] == hemi]
        if len(sub):
            d_mean = sub["cohens_d"].mean()
            print(f"    Hemisferio {hemi}: d promedio = {d_mean:.2f}")


# =============================================================================
# FIGURAS — divididas por hemisferio
# =============================================================================

def graficar_barras(tab, df, col, ventana_label,
                    canales, hemi_label, out_file):
    """
    Barras de amplitud media por grupo con barras de error (+-SEM)
    y '+' donde el test es significativo tras FDR.
    """
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5.5))
    if len(CONDICIONES) == 1:
        axes = [axes]
    fig.suptitle(
        f"{ventana_label} — metodo {METODO_PRINCIPAL} — "
        f"hemisferio {hemi_label}\n"
        f"Barras de error: +- SEM  |  '+' : significativo tras FDR (p<0.05)",
        fontsize=13
    )
    x = np.arange(len(canales)); ancho = 0.38

    for ax, cond in zip(axes, CONDICIONES):
        medias_c, sem_c, medias_a, sem_a = [], [], [], []
        for canal in canales:
            c = df[(df["canal"] == canal) & (df["condicion"] == cond) &
                   (df["grupo"] == "control")][col].dropna().values
            a = df[(df["canal"] == canal) & (df["condicion"] == cond) &
                   (df["grupo"] == "alcoholic")][col].dropna().values
            medias_c.append(c.mean()); sem_c.append(c.std(ddof=1)/np.sqrt(len(c)))
            medias_a.append(a.mean()); sem_a.append(a.std(ddof=1)/np.sqrt(len(a)))

        ax.bar(x - ancho/2, medias_c, ancho, yerr=sem_c, capsize=4,
               label="Control", color=colores["control"], alpha=0.8)
        ax.bar(x + ancho/2, medias_a, ancho, yerr=sem_a, capsize=4,
               label="Alcoholico", color=colores["alcoholic"], alpha=0.8)

        for xi, canal in enumerate(canales):
            r = tab[(tab["canal"] == canal) & (tab["condicion"] == cond)]
            if not r.empty and bool(r["sig_fdr"].values[0]):
                ytop = max(medias_c[xi] + sem_c[xi], medias_a[xi] + sem_a[xi])
                ax.text(xi, ytop + 0.3, "+", ha="center",
                        fontsize=16, fontweight="bold")
            # Anotar Cohen's d debajo
            if not r.empty:
                d_val = r["cohens_d"].values[0]
                ax.text(xi, ax.get_ylim()[0] + 0.1, f"d={d_val:.2f}",
                        ha="center", fontsize=7, color="gray")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(canales)
        ax.set_title(f"Condicion: {cond}", fontsize=11)
        ax.set_ylabel(f"{col} (uV)")
        ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"../outputs/{out_file}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{out_file}'")


def graficar_lateralizacion(tab):
    """LI = R - L por par, control vs alcoholico."""
    if "LI_ctrl" not in tab.columns:
        print("  (sin datos para figura de lateralizacion)")
        return

    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    pares = [f"{p[0]}-{p[1]}" for p in PARES_HEMISFERICOS]
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(6.5 * len(CONDICIONES), 5))
    if len(CONDICIONES) == 1:
        axes = [axes]
    fig.suptitle(
        f"Indice de lateralizacion LI = R - L — metodo {METODO_PRINCIPAL}\n"
        "LI > 0 => hemisferio derecho domina  |  "
        "NOTA: el efecto control>alc es bilateral en estos datos",
        fontsize=13
    )
    x = np.arange(len(pares)); ancho = 0.38

    for ax, cond in zip(axes, CONDICIONES):
        sub = tab[tab["condicion"] == cond].set_index("par").reindex(pares)
        li_c = sub["LI_ctrl"].values
        li_a = sub["LI_alc"].values
        ax.bar(x - ancho/2, li_c, ancho, label="Control",
               color=colores["control"], alpha=0.8)
        ax.bar(x + ancho/2, li_a, ancho, label="Alcoholico",
               color=colores["alcoholic"], alpha=0.8)

        for xi, par in enumerate(pares):
            r = tab[(tab["par"] == par) & (tab["condicion"] == cond)]
            if (not r.empty and "p_LI_fdr" in r.columns
                    and pd.notna(r["p_LI_fdr"].values[0])
                    and r["p_LI_fdr"].values[0] < ALPHA):
                ytop = max(li_c[xi], li_a[xi], 0)
                ax.text(xi, ytop + 0.1, "+", ha="center",
                        fontsize=16, fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(pares, rotation=15)
        ax.set_title(f"Condicion: {cond}", fontsize=11)
        ax.set_ylabel("LI = R - L (uV)")
        ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_lateralizacion.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_lateralizacion.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 88)
    print("TPS -- Potenciales Evocados Visuales en Alcoholismo")
    print("Script 06: Analisis Estadistico  [v3 — corregido]")
    print("=" * 88)

    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA}'. Corre primero el Script 05.")

    print(f"\nCargando '{ENTRADA}'...")
    df_full = pd.read_csv(ENTRADA)
    print(f"  Metodos disponibles: {df_full['metodo'].unique().tolist()}")
    print(f"  Sujetos totales: {df_full['sujeto'].nunique()}")

    # Metodo principal: HOMOGENEO
    df = df_full[df_full["metodo"] == METODO_PRINCIPAL].copy()
    print(f"\n  Metodo PRINCIPAL: {METODO_PRINCIPAL}")
    print(f"  Sujetos control:     {df[df['grupo']=='control']['sujeto'].nunique()}")
    print(f"  Sujetos alcoholicos: {df[df['grupo']=='alcoholic']['sujeto'].nunique()}")

    # =====================================================================
    # BLOQUE 1: AMPLITUD MEDIA — c240 (220-260 ms, ventana primaria)
    # =====================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 1a: AMPLITUD MEDIA — c240 (220-260 ms) — VENTANA PRIMARIA")
    print("#" * 88)
    tab_amp_c240 = analizar_grupos(df, COL_MEDIA_C240, "control>alc",
                                   "c240 (220-260 ms)")
    imprimir_grupos(tab_amp_c240,
                    "AMPLITUD MEDIA c240 (220-260 ms) — control vs alcoholico",
                    "media_control > media_alcoholico")
    resumen_bloque(tab_amp_c240, "amplitud c240")

    print("\n\n" + "#" * 88)
    print("  BLOQUE 1b: AMPLITUD MEDIA — c320 (290-340 ms) — VENTANA SECUNDARIA")
    print("#" * 88)
    tab_amp_c320 = analizar_grupos(df, COL_MEDIA_C320, "control>alc",
                                   "c320 (290-340 ms)")
    imprimir_grupos(tab_amp_c320,
                    "AMPLITUD MEDIA c320 (290-340 ms) — positividad tardia",
                    "media_control > media_alcoholico")
    resumen_bloque(tab_amp_c320, "amplitud c320")

    # =====================================================================
    # BLOQUE 2: AUC
    # =====================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 2: AUC (area bajo la curva)")
    print("#" * 88)
    tab_auc_c240 = analizar_grupos(df, COL_AUC_C240, "control>alc",
                                   "AUC c240 (220-260 ms)")
    imprimir_grupos(tab_auc_c240,
                    "AUC c240 (220-260 ms)",
                    "AUC_control > AUC_alcoholico")
    resumen_bloque(tab_auc_c240, "AUC c240")

    # =====================================================================
    # BLOQUE 3: LATENCIA (direccion INVERSA)
    # =====================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 3: LATENCIA (direccion INVERSA)")
    print("#" * 88)
    tab_lat = analizar_grupos(df, COL_LAT_C240, "alc>control",
                              "latencia c240")
    imprimir_grupos(tab_lat,
                    "LATENCIA c240 (220-260 ms)",
                    "latencia_alcoholico > latencia_control (mas lento)")
    resumen_bloque(tab_lat, "latencia c240")

    # =====================================================================
    # BLOQUE 4: LATERALIZACION
    # =====================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 4: LATERALIZACION")
    print("#" * 88)
    tab_lateral = tabla_lateralizacion(df, COL_MEDIA_C240)
    imprimir_lateralizacion(tab_lateral)

    # =====================================================================
    # BLOQUE 5: ANALISIS SECUNDARIO — inhomogeneo (robustez)
    # =====================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 5: ANALISIS SECUNDARIO — inhomogeneo")
    print("#" * 88)
    df_inh = df_full[df_full["metodo"] == METODO_SECUNDARIO].copy()
    tab_inh = analizar_grupos(df_inh, COL_MEDIA_C240, "control>alc",
                              "c240 inh.")
    imprimir_grupos(tab_inh,
                    f"AMPLITUD MEDIA c240 — metodo {METODO_SECUNDARIO.upper()} "
                    "(analisis secundario)",
                    "media_control > media_alcoholico")
    resumen_bloque(tab_inh, "amplitud c240 inhomogeneo")

    # Comparar: la conclusion se sostiene con ambos metodos?
    sig_hom = tab_amp_c240["sig_fdr"].sum()
    sig_inh = tab_inh["sig_fdr"].sum() if len(tab_inh) else 0
    print(f"\n  Robustez: homogeneo {sig_hom}/16 sig FDR, "
          f"inhomogeneo {sig_inh}/16 sig FDR")

    # =====================================================================
    # GUARDAR TABLAS
    # =====================================================================
    all_tabs = pd.concat([tab_amp_c240, tab_amp_c320, tab_auc_c240, tab_lat,
                          tab_inh], ignore_index=True)
    all_tabs.to_csv(SALIDA_CSV, index=False)
    tab_lateral.to_csv(SALIDA_LATERAL, index=False)
    print(f"\nTabla de contrastes -> '{SALIDA_CSV}'")
    print(f"Tabla de lateralizacion -> '{SALIDA_LATERAL}'")

    # =====================================================================
    # FIGURAS — divididas por hemisferio
    # =====================================================================
    print("\nGenerando figuras...")
    for canales, label, sufijo in [
        (CANALES_DERECHO,   "derecho",   "derecho"),
        (CANALES_IZQUIERDO, "izquierdo", "izquierdo"),
    ]:
        graficar_barras(tab_amp_c240, df, COL_MEDIA_C240,
                        "Amplitud media c240 (220-260 ms)",
                        canales, label,
                        f"figura_barras_{sufijo}_c240.png")
        graficar_barras(tab_amp_c320, df, COL_MEDIA_C320,
                        "Amplitud media c320 (290-340 ms)",
                        canales, label,
                        f"figura_barras_{sufijo}_c320.png")

    graficar_lateralizacion(tab_lateral)

    print("\n[OK] Script 06 finalizado.")
