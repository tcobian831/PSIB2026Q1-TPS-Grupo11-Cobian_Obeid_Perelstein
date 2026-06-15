"""
==============================================================================
TPS - Procesamiento de Senales Biomedicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobian, Obeid, Perelstein
 
Script 06: Analisis Estadistico
==============================================================================
 
Proposito:
    Analizar el componente c240/VMP sobre los 8 canales (4 derechos del paper
    + 4 homologos izquierdos), respondiendo dos hipotesis centrales:
 
    H1 (principal): Los controles tienen mayor amplitud del VMP que los
        alcoholicos en la region temporooccipital.
 
    H2 (secundaria): El efecto es mas pronunciado en hemisferio derecho.
 
    METODO PRINCIPAL: promedio homogeneo.
    VENTANA PRIMARIA: 220-260 ms (c240, Zhang et al. 1997).
    VENTANA SECUNDARIA: 290-340 ms (c320, donde cae el pico real en
        nuestros datos por ausencia de baseline pre-estimulo).
    METRICA: media de la senal en la ventana.
    MUESTRA: 77 alcoholicos vs 45 controles.
    ANALISIS: comparacion descriptiva (medias, SD, diferencia ctrl-alc).
 
Bloques:
    1a. Amplitud media c240 (ventana primaria)
    1b. Amplitud media c320 (ventana secundaria)
    2.  AUC c240
    3.  Latencia c240 y c320 (nota: poco informativa, ver comentarios)
    5.  Analisis secundario: metodo inhomogeneo (robustez)
    6.  Metricas descriptivas simples (H1 y H2)
 
Entrada:  outputs/eeg_c240_extraido.csv  (del Script 05)
Salida:   outputs/tabla_estadistica.csv
          outputs/figura_barras_derecho_c240.png
          outputs/figura_barras_izquierdo_c240.png
          outputs/figura_barras_derecho_c320.png
          outputs/figura_barras_izquierdo_c320.png
          outputs/figura_lateralizacion_c240.png
          outputs/figura_lateralizacion_c320.png
 
Uso:
    Correr desde la carpeta scripts/
    python 06_estadistica.py
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
# CONFIGURACION
# =============================================================================
 
CANALES_DERECHO    = ["P8",  "PO8",  "T8",  "TP8"]
CANALES_IZQUIERDO  = ["P7",  "PO7",  "T7",  "TP7"]
CANALES_INTERES    = CANALES_DERECHO + CANALES_IZQUIERDO
PARES_HEMISFERICOS = [("P8", "P7"), ("PO8", "PO7"), ("T8", "T7"), ("TP8", "TP7")]
CONDICIONES        = ["S1 obj", "S2 nomatch"]
 
METODO_PRINCIPAL  = "homogeneo"
METODO_SECUNDARIO = "inhomogeneo"
 
COL_MEDIA_C240 = "media_c240"
COL_MEDIA_C320 = "media_c320"
COL_AUC_C240   = "auc_c240"
COL_AUC_C320   = "auc_c320"
COL_LAT_C240   = "lat_max_c240"
COL_LAT_C320   = "lat_max_c320"

ENTRADA        = Path("outputs/eeg_c240_extraido.csv")
SALIDA_CSV     = Path("outputs/tabla_estadistica.csv")
 
 
# =============================================================================
# CONTRASTE ENTRE GRUPOS (DESCRIPTIVO)
# =============================================================================

def analizar_grupos(df, col, ventana_label):
    """
    Comparacion descriptiva control vs alcoholico por canal x condicion.
    Reporta media +- SD de cada grupo y la diferencia (control - alcoholico).
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

            filas.append({
                "ventana": ventana_label, "metrica": col,
                "canal": canal, "condicion": cond,
                "hemisferio": "derecho" if canal in CANALES_DERECHO else "izquierdo",
                "n_control": len(ctrl), "n_alcoholic": len(alc),
                "control_media": mc, "control_sd": sc,
                "alcoholic_media": ma, "alcoholic_sd": sa,
                "diferencia": mc - ma,
            })

    return pd.DataFrame(filas)


# =============================================================================
# IMPRESION EN CONSOLA
# =============================================================================

def imprimir_grupos(tab, titulo, h1):
    print(f"\n{'='*88}")
    print(f"{titulo}")
    print(f"H1: {h1}")
    print(f"{'='*88}")
    print(f"  {'Canal':<6}{'Cond.':<12}{'Control':>16}{'Alcoholico':>16}"
          f"{'Dif.':>10}")
    print("  " + "-" * 60)
    for _, r in tab.iterrows():
        print(f"  {r['canal']:<6}{r['condicion']:<12}"
              f"{r['control_media']:>+8.2f} +- {r['control_sd']:>4.2f}"
              f"{r['alcoholic_media']:>+8.2f} +- {r['alcoholic_sd']:>4.2f}"
              f"{r['diferencia']:>+10.2f}")


def resumen_bloque(tab, label):
    n = len(tab)
    if n == 0:
        return
    print(f"\n  RESUMEN {label}:")
    print(f"    Celdas: {n}")
    for hemi in ["derecho", "izquierdo"]:
        sub = tab[tab["hemisferio"] == hemi]
        if len(sub):
            print(f"    Hemisferio {hemi}: diferencia promedio = "
                  f"{sub['diferencia'].mean():+.2f}")
 
 
# =============================================================================
# METRICAS DESCRIPTIVAS SIMPLES
# =============================================================================
 
def metricas_simples(df, col=COL_MEDIA_C240):
    """
    Metricas descriptivas simples para afirmar o desmentir las dos
    hipotesis del paper de Zhang et al. (1997):
 
    H1: Control tiene mayor amplitud VMP que alcoholico.
        -> media +- SD por grupo + % sujetos con amplitud positiva.
 
    H2: El efecto es mas pronunciado en hemisferio derecho que izquierdo.
        -> diferencia control-alcoholico por canal, asimetria D-I.
    """
    ventana = "c240 (220-260 ms)" if col == COL_MEDIA_C240 \
              else "c320 (290-340 ms)"
 
    print(f"\n{'='*70}")
    print(f"METRICAS DESCRIPTIVAS SIMPLES — ventana {ventana}")
    print(f"{'='*70}")
 
    # ------------------------------------------------------------------
    # H1: Amplitud media por grupo + % con amplitud positiva
    # ------------------------------------------------------------------
    print("\n--- H1: Control > Alcoholico en amplitud VMP? ---\n")
    print(f"  {'Canal':<6} {'Cond.':<12} {'Ctrl media':>11} "
          f"{'Alc media':>11} {'Dif.':>8} "
          f"{'% pos ctrl':>11} {'% pos alc':>10}")
    print("  " + "-" * 73)
 
    for canal in CANALES_INTERES:
        for cond in CONDICIONES:
            sub  = df[(df["canal"] == canal) & (df["condicion"] == cond)]
            ctrl = sub[sub["grupo"] == "control"][col].dropna().values
            alc  = sub[sub["grupo"] == "alcoholic"][col].dropna().values
            if len(ctrl) < 2 or len(alc) < 2:
                continue
 
            mc, sc       = ctrl.mean(), ctrl.std(ddof=1)
            ma, sa       = alc.mean(),  alc.std(ddof=1)
            dif          = mc - ma
            pct_pos_ctrl = (ctrl > 0).mean() * 100
            pct_pos_alc  = (alc  > 0).mean() * 100
 
            print(f"  {canal:<6} {cond:<12} "
                  f"{mc:>+6.2f}+-{sc:<4.2f} "
                  f"{ma:>+6.2f}+-{sa:<4.2f} "
                  f"{dif:>+8.2f} "
                  f"{pct_pos_ctrl:>10.0f}% "
                  f"{pct_pos_alc:>9.0f}%")
 
    # ------------------------------------------------------------------
    # H2: Asimetria hemisferica
    # ------------------------------------------------------------------
    print("\n--- H2: El efecto es mayor en hemisferio derecho? ---")
    print("    Asimetria = (ctrl-alc) derecho - (ctrl-alc) izquierdo")
    print("    Asimetria > 0.5 uV -> replica Zhang")
    print("    Asimetria entre -0.5 y +0.5 uV -> bilateral\n")
    print(f"  {'Par':<10} {'Cond.':<12} {'Dif. D':>8} "
          f"{'Dif. I':>8} {'Asimetria':>10} {'Conclusion':>15}")
    print("  " + "-" * 65)
 
    resultados_h2 = []
    for (R, L) in PARES_HEMISFERICOS:
        for cond in CONDICIONES:
            sub    = df[df["condicion"] == cond]
            ctrl_R = sub[(sub["canal"] == R) &
                         (sub["grupo"] == "control")][col].dropna().values
            alc_R  = sub[(sub["canal"] == R) &
                         (sub["grupo"] == "alcoholic")][col].dropna().values
            ctrl_L = sub[(sub["canal"] == L) &
                         (sub["grupo"] == "control")][col].dropna().values
            alc_L  = sub[(sub["canal"] == L) &
                         (sub["grupo"] == "alcoholic")][col].dropna().values
 
            if any(len(x) < 2 for x in [ctrl_R, alc_R, ctrl_L, alc_L]):
                continue
 
            dif_D = ctrl_R.mean() - alc_R.mean()
            dif_I = ctrl_L.mean() - alc_L.mean()
            asim  = dif_D - dif_I
 
            if asim > 0.5:
                conclusion = "-> replica Zhang"
                resultados_h2.append("zhang")
            elif asim < -0.5:
                conclusion = "-> domina izq."
                resultados_h2.append("izq")
            else:
                conclusion = "-> bilateral"
                resultados_h2.append("bilateral")
 
            print(f"  {R+'-'+L:<10} {cond:<12} "
                  f"{dif_D:>+8.2f} {dif_I:>+8.2f} "
                  f"{asim:>+10.2f} {conclusion:>16}")
 
    # Resumen H2
    n_zhang    = resultados_h2.count("zhang")
    n_bilateral = resultados_h2.count("bilateral")
    n_izq      = resultados_h2.count("izq")
    n_total    = len(resultados_h2)
    print(f"\n  Resumen H2 ({n_total} pares analizados):")
    print(f"    Replica Zhang (dominancia derecha): {n_zhang}/{n_total}")
    print(f"    Bilateral:                          {n_bilateral}/{n_total}")
    print(f"    Dominancia izquierda:               {n_izq}/{n_total}")
 
    if n_zhang > n_total / 2:
        print("    -> CONCLUSION: H2 CONFIRMADA "
              "(mayoría de pares replica Zhang)")
    elif n_bilateral >= n_zhang:
        print("    -> CONCLUSION: H2 NO REPLICADA "
              "(efecto bilateral en estos datos)")
    print(f"{'='*70}")
 
 
# =============================================================================
# FIGURAS
# =============================================================================
 
def graficar_barras(tab, df, col, ventana_label,
                    canales, hemi_label, out_file):
    """Barras de amplitud media por grupo con SEM."""
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5.5))
    if len(CONDICIONES) == 1:
        axes = [axes]
    fig.suptitle(
        f"Amplitud media por grupo y canal — {hemi_label} — ventana {ventana_label}\n"
        "Azul = Control  |  Rojo = Alcoholico  (barras: media ± SEM)",
        fontsize=12
    )
    
    x = np.arange(len(canales))
    ancho = 0.38
 
    for ax, cond in zip(axes, CONDICIONES):
        medias_c, sem_c, medias_a, sem_a = [], [], [], []
        for canal in canales:
            c = df[(df["canal"] == canal) & (df["condicion"] == cond) &
                   (df["grupo"] == "control")][col].dropna().values
            a = df[(df["canal"] == canal) & (df["condicion"] == cond) &
                   (df["grupo"] == "alcoholic")][col].dropna().values
            medias_c.append(c.mean())
            sem_c.append(c.std(ddof=1) / np.sqrt(len(c)))
            medias_a.append(a.mean())
            sem_a.append(a.std(ddof=1) / np.sqrt(len(a)))
 
        ax.bar(x - ancho/2, medias_c, ancho, yerr=sem_c, capsize=4,
               label="Control", color=colores["control"], alpha=0.8)
        ax.bar(x + ancho/2, medias_a, ancho, yerr=sem_a, capsize=4,
               label="Alcoholico", color=colores["alcoholic"], alpha=0.8)
 
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(canales)
        ax.set_title(f"Condicion: {cond}", fontsize=11)
        ax.set_ylabel(f"{col} (uV)")
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(f"outputs/{out_file}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{out_file}'")


def graficar_lateralizacion(df, col, ventana_label, out_file):
    """
    Barras de la diferencia (control - alcoholico) por par hemisferico (H2).
    Azul oscuro = hemisferio derecho (dif. en canal derecho).
    Azul claro  = hemisferio izquierdo (dif. en canal homologo izquierdo).
    Sobre cada par se anota la asimetria = dif_D - dif_I.
    Barras de error: SEM de la diferencia = sqrt(SEM_ctrl^2 + SEM_alc^2).
    """
    color_der = "#2563eb"   # azul oscuro (hemisferio derecho)
    color_izq = "#93c5fd"   # azul claro  (hemisferio izquierdo)

    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5.5))
    if len(CONDICIONES) == 1:
        axes = [axes]
    fig.suptitle(
        f"Diferencia Control - Alcoholico por hemisferio — ventana {ventana_label}\n"
        "Azul oscuro = hemisferio derecho  |  Azul claro = hemisferio izquierdo\n"
        "Asimetria > 0 -> replica Zhang (dominancia derecha)  |  ~0 -> bilateral",
        fontsize=12
    )

    x = np.arange(len(PARES_HEMISFERICOS))
    ancho = 0.38

    for ax, cond in zip(axes, CONDICIONES):
        dif_D_list, sem_D_list = [], []
        dif_I_list, sem_I_list = [], []
        asim_list, etiquetas_x = [], []

        sub = df[df["condicion"] == cond]
        for (R, L) in PARES_HEMISFERICOS:
            ctrl_R = sub[(sub["canal"] == R) &
                         (sub["grupo"] == "control")][col].dropna().values
            alc_R  = sub[(sub["canal"] == R) &
                         (sub["grupo"] == "alcoholic")][col].dropna().values
            ctrl_L = sub[(sub["canal"] == L) &
                         (sub["grupo"] == "control")][col].dropna().values
            alc_L  = sub[(sub["canal"] == L) &
                         (sub["grupo"] == "alcoholic")][col].dropna().values

            dif_D = ctrl_R.mean() - alc_R.mean()
            dif_I = ctrl_L.mean() - alc_L.mean()
            sem_D = np.sqrt(ctrl_R.std(ddof=1)**2 / len(ctrl_R) +
                            alc_R.std(ddof=1)**2 / len(alc_R))
            sem_I = np.sqrt(ctrl_L.std(ddof=1)**2 / len(ctrl_L) +
                            alc_L.std(ddof=1)**2 / len(alc_L))

            dif_D_list.append(dif_D); sem_D_list.append(sem_D)
            dif_I_list.append(dif_I); sem_I_list.append(sem_I)
            asim_list.append(dif_D - dif_I)
            etiquetas_x.append(f"{R}\nvs\n{L}")

        ax.bar(x - ancho/2, dif_D_list, ancho, yerr=sem_D_list, capsize=4,
               label="Hemisferio derecho", color=color_der, alpha=0.9)
        ax.bar(x + ancho/2, dif_I_list, ancho, yerr=sem_I_list, capsize=4,
               label="Hemisferio izquierdo", color=color_izq, alpha=0.9)

        ax.axhline(0, color="black", linewidth=0.5)

        # Anotacion de asimetria sobre cada par (arriba del tope del par).
        rng = ax.get_ylim()[1] - ax.get_ylim()[0]
        for xi in range(len(PARES_HEMISFERICOS)):
            tope = max(dif_D_list[xi] + sem_D_list[xi],
                       dif_I_list[xi] + sem_I_list[xi])
            ax.text(xi, tope + 0.03 * rng, f"asim: {asim_list[xi]:+.1f} uV",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(top=ax.get_ylim()[1] + 0.10 * rng)

        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas_x)
        ax.set_title(f"Condicion: {cond}", fontsize=11)
        ax.set_ylabel("Diferencia ctrl - alc en amplitud VMP (uV)")
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
 
    print("=" * 88)
    print("TPS -- Potenciales Evocados Visuales en Alcoholismo")
    print("Script 06: Analisis Estadistico  [v4 — final]")
    print("=" * 88)
 
    if not ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encontro '{ENTRADA}'. Corre primero el Script 05.")
 
    print(f"\nCargando '{ENTRADA}'...")
    df_full = pd.read_csv(ENTRADA)
    print(f"  Metodos disponibles: {df_full['metodo'].unique().tolist()}")
    print(f"  Sujetos totales: {df_full['sujeto'].nunique()}")
 
    df = df_full[df_full["metodo"] == METODO_PRINCIPAL].copy()
    print(f"\n  Metodo PRINCIPAL: {METODO_PRINCIPAL}")
    print(f"  Sujetos control:     "
          f"{df[df['grupo']=='control']['sujeto'].nunique()}")
    print(f"  Sujetos alcoholicos: "
          f"{df[df['grupo']=='alcoholic']['sujeto'].nunique()}")
 
    # =========================================================================
    # BLOQUE 1a: AMPLITUD MEDIA c240 (ventana primaria)
    # =========================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 1a: AMPLITUD MEDIA — c240 (220-260 ms) — VENTANA PRIMARIA")
    print("#" * 88)
    tab_amp_c240 = analizar_grupos(df, COL_MEDIA_C240, "c240 (220-260 ms)")
    imprimir_grupos(tab_amp_c240,
                    "AMPLITUD MEDIA c240 (220-260 ms) — control vs alcoholico",
                    "media_control > media_alcoholico")
    resumen_bloque(tab_amp_c240, "amplitud c240")
 
    # =========================================================================
    # BLOQUE 1b: AMPLITUD MEDIA c320 (ventana secundaria)
    # =========================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 1b: AMPLITUD MEDIA — c320 (290-340 ms) — VENTANA SECUNDARIA")
    print("#" * 88)
    tab_amp_c320 = analizar_grupos(df, COL_MEDIA_C320, "c320 (290-340 ms)")
    imprimir_grupos(tab_amp_c320,
                    "AMPLITUD MEDIA c320 (290-340 ms) — positividad tardia",
                    "media_control > media_alcoholico")
    resumen_bloque(tab_amp_c320, "amplitud c320")
 
    # =========================================================================
    # BLOQUE 2: AUC
    # =========================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 2: AUC (area bajo la curva) — c240")
    print("#" * 88)
    tab_auc_c240 = analizar_grupos(df, COL_AUC_C240, "AUC c240 (220-260 ms)")
    imprimir_grupos(tab_auc_c240,
                    "AUC c240 (220-260 ms)",
                    "AUC_control > AUC_alcoholico")
    resumen_bloque(tab_auc_c240, "AUC c240")
 
    # =========================================================================
    # BLOQUE 3: LATENCIA
    # Nota: poco informativa porque el grupo alcoholico no genera un pico
    # positivo claro. El argmax encuentra el punto menos negativo de la
    # señal, no un pico real. Se reporta como limitacion del analisis.
    # =========================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 3: LATENCIA")
    print("  NOTA: metrica poco informativa en alcoholicos (sin pico positivo)")
    print("#" * 88)
    tab_lat_c240 = analizar_grupos(df, COL_LAT_C240, "latencia c240")
    imprimir_grupos(tab_lat_c240,
                    "LATENCIA c240 (220-260 ms)",
                    "latencia_alcoholico > latencia_control")
    resumen_bloque(tab_lat_c240, "latencia c240")
 
    print("\n" + "-" * 88)
    tab_lat_c320 = analizar_grupos(df, COL_LAT_C320, "latencia c320")
    imprimir_grupos(tab_lat_c320,
                    "LATENCIA c320 (290-340 ms) — ventana donde cae el pico real",
                    "latencia_alcoholico > latencia_control")
    resumen_bloque(tab_lat_c320, "latencia c320")
 
    # =========================================================================
    # BLOQUE 5: ANALISIS SECUNDARIO — inhomogeneo (robustez)
    # =========================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 5: ANALISIS SECUNDARIO — inhomogeneo (robustez)")
    print("#" * 88)
    df_inh  = df_full[df_full["metodo"] == METODO_SECUNDARIO].copy()
    tab_inh = analizar_grupos(df_inh, COL_MEDIA_C240, "c240 inh.")
    imprimir_grupos(tab_inh,
                    f"AMPLITUD MEDIA c240 — metodo {METODO_SECUNDARIO.upper()}",
                    "media_control > media_alcoholico")
    resumen_bloque(tab_inh, "amplitud c240 inhomogeneo")
 
    dif_hom = tab_amp_c240["diferencia"].mean()
    dif_inh = tab_inh["diferencia"].mean() if len(tab_inh) else float("nan")
    print(f"\n  Robustez: diferencia media (ctrl-alc) homogeneo {dif_hom:+.2f} uV, "
          f"inhomogeneo {dif_inh:+.2f} uV")
 
    # =========================================================================
    # BLOQUE 6: METRICAS DESCRIPTIVAS SIMPLES
    # =========================================================================
    print("\n\n" + "#" * 88)
    print("  BLOQUE 6: METRICAS DESCRIPTIVAS SIMPLES (H1 y H2)")
    print("#" * 88)
    metricas_simples(df, col=COL_MEDIA_C240)
    metricas_simples(df, col=COL_MEDIA_C320)
 
    # =========================================================================
    # GUARDAR TABLAS
    # =========================================================================
    all_tabs = pd.concat(
        [tab_amp_c240, tab_amp_c320, tab_auc_c240,
         tab_lat_c240, tab_lat_c320, tab_inh],
        ignore_index=True
    )
    all_tabs.to_csv(SALIDA_CSV, index=False)
    print(f"\nTabla de contrastes  -> '{SALIDA_CSV}'")
 
    # =========================================================================
    # FIGURAS
    # =========================================================================
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

    # Figuras de lateralizacion (asimetria hemisferica) — H2
    graficar_lateralizacion(df, COL_MEDIA_C240, "c240 (220-260 ms)",
                            "figura_lateralizacion_c240.png")
    graficar_lateralizacion(df, COL_MEDIA_C320, "c320 (290-340 ms)",
                            "figura_lateralizacion_c320.png")

    print("\n[OK] Script 06 finalizado.")