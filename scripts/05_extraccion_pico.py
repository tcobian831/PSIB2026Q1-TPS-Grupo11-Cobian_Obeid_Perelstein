"""
==============================================================================
TPS - Procesamiento de Senales Biomedicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobian, Obeid, Perelstein

Script 05: Extraccion de Metricas del c240 (y positividad tardia)
==============================================================================

Proposito:
    Para cada sujeto, canal y condicion, extraer metricas del PE individual
    en DOS ventanas temporales justificadas por la teoria:

    VENTANA PRIMARIA: 220-260 ms  (componente c240/VMP de Zhang et al. 1997)
        Esta es la medicion oficial del c240, anclada al marco teorico.
        NO se eligio mirando el Grand Average (eso seria seleccion post-hoc).

    VENTANA SECUNDARIA: 290-340 ms  (positividad tardia / posible c320)
        En nuestros datos, la positividad del Grand Average pica ~300-340 ms,
        algo mas tarde que en Zhang. Probablemente corresponde al c320 de Zhang
        (que el reporta en ventana 290-340 ms), NO al c240 desplazado.
        Se reporta como ANALISIS SECUNDARIO con ese nombre explicito.

    Metricas por sujeto, canal y condicion (para cada ventana):
      - media_c240 / media_c320 : MEDIA de la senal en la ventana (metrica
        principal). Estable y honesta aunque la ventana caiga sobre el flanco.
      - max_c240 / max_c320 : maximo positivo en la ventana (metrica secundaria).
      - lat_max_c240 / lat_max_c320 : latencia del maximo positivo (ms).
      - auc_c240 / auc_c320 : area con signo bajo la curva (uV*ms, trapecio).
        Complementaria al pico: integra toda la deflexion.
    

    Se usa como entrada tanto el PE HOMOGENEO como el INHOMOGENEO del Script 04.
    El Script 06 lidera con el homogeneo como analisis principal y usa el
    inhomogeneo como analisis secundario.


Entrada:  outputs/eeg_PE_homogeneo.parquet
          outputs/eeg_PE_inhomogeneo.parquet    (ambos del Script 04, muestra COMPLETA)
Salida:   outputs/eeg_c240_extraido.csv         (metricas por sujeto, 77+45 sujetos)
          outputs/figura_boxplot_derecho_*.png   (boxplots por hemisferio)
          outputs/figura_boxplot_izquierdo_*.png
          outputs/figura_latencia_derecho.png
          outputs/figura_latencia_izquierdo.png

Uso:
    Correr desde la carpeta scripts/
    python 05_extraccion_pico.py
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

FS        = 256   # Hz
N_SAMPLES = 256   # muestras por trial

# Canales
CANALES_DERECHO    = ["P8",  "PO8",  "T8",  "TP8"]
CANALES_IZQUIERDO  = ["P7",  "PO7",  "T7",  "TP7"]
CANALES_INTERES    = CANALES_DERECHO + CANALES_IZQUIERDO
PARES_HEMISFERICOS = [("P8", "P7"), ("PO8", "PO7"), ("T8", "T7"), ("TP8", "TP7")]
CONDICIONES        = ["S1 obj", "S2 nomatch"]

# --- Ventana PRIMARIA: c240 (Zhang et al. 1997, tabla 1) ---
V_C240 = (220, 260)   # ms
M_C240_LO = int(round(V_C240[0] / 1000 * FS))
M_C240_HI = int(round(V_C240[1] / 1000 * FS))

# --- Ventana SECUNDARIA: positividad tardia / posible c320 ---
V_C320 = (290, 340)   # ms
M_C320_LO = int(round(V_C320[0] / 1000 * FS))
M_C320_HI = int(round(V_C320[1] / 1000 * FS))

METODOS = {
    "homogeneo":   Path("outputs/eeg_PE_homogeneo.parquet"),
    "inhomogeneo": Path("outputs/eeg_PE_inhomogeneo.parquet"),
}

SALIDA_CSV = Path("outputs/eeg_c240_extraido.csv")

# =============================================================================
# FUNCIONES DE EXTRACCION
# =============================================================================

def extraer_metricas_ventana(senal, m_ini, m_fin):
    """
    Para senal[m_ini:m_fin] calcula:
      - media: promedio de la senal en la ventana (metrica principal)
      - maximo: maximo positivo
      - latencia: latencia del maximo positivo (ms)
      - auc: area con signo (trapecio, uV*ms)
    """
    seg = senal[m_ini:m_fin]
    media = float(np.mean(seg))
    idx = int(np.argmax(seg))
    maximo = float(seg[idx])
    latencia = float((idx + m_ini) / FS * 1000)
    auc = float(np.trapz(seg, dx=1000.0 / FS))
    return media, maximo, latencia, auc


def extraer_todos(pe_ind: pd.DataFrame, metodo: str) -> pd.DataFrame:
    """
    Para cada sujeto x canal x condicion, extrae metricas en ambas ventanas.
    """
    resultados = []
    grupos = pe_ind.groupby(["sujeto", "grupo", "canal", "condicion"])
    print(f"  [{metodo}] {len(grupos):,} combinaciones...")

    for (sujeto, grupo, canal, cond), sub in grupos:
        s = sub.sort_values("muestra")["valor_uV"].values
        if len(s) != N_SAMPLES:
            continue

        n_trials = sub["n_trials"].iloc[0] if "n_trials" in sub.columns else np.nan

        # Ventana primaria: c240 (220-260 ms)
        media_c240, max_c240, lat_c240, auc_c240 = extraer_metricas_ventana(
            s, M_C240_LO, M_C240_HI + 1)

        # Ventana secundaria: c320 (290-340 ms)
        media_c320, max_c320, lat_c320, auc_c320 = extraer_metricas_ventana(
            s, M_C320_LO, M_C320_HI + 1)

        resultados.append({
            "sujeto":      sujeto,
            "grupo":       grupo,
            "canal":       canal,
            "condicion":   cond,
            "metodo":      metodo,
            "n_trials":    n_trials,
            # Ventana primaria c240
            "media_c240":  media_c240,
            "max_c240":    max_c240,
            "lat_max_c240": lat_c240,
            "auc_c240":    auc_c240,
            # Ventana secundaria c320
            "media_c320":  media_c320,
            "max_c320":    max_c320,
            "lat_max_c320": lat_c320,
            "auc_c320":    auc_c320,
        })

    return pd.DataFrame(resultados)


# =============================================================================
# RESUMEN DESCRIPTIVO
# =============================================================================

def imprimir_resumen(df_all, metodo, col_media, ventana_label):
    """
    Imprime media +- SD por grupo y diferencia.
    """
    df = df_all[df_all["metodo"] == metodo]
    print(f"\n{'='*76}")
    print(f"RESUMEN — {ventana_label} — metodo {metodo.upper()}")
    print(f"Metrica: {col_media} (media en ventana)")
    print(f"{'='*76}")
    print(f"  {'Canal':<6}{'Cond.':<12}{'Control':>14}{'Alcoholico':>16}"
          f"{'Dif.':>9}{'N ctrl':>7}{'N alc':>7}")
    print("  " + "-" * 73)
    for canal in CANALES_INTERES:
        for cond in CONDICIONES:
            sub = df[(df["canal"] == canal) & (df["condicion"] == cond)]
            ctrl = sub[sub["grupo"] == "control"][col_media].dropna().values
            alc  = sub[sub["grupo"] == "alcoholic"][col_media].dropna().values
            if len(ctrl) < 2 or len(alc) < 2:
                continue
            mc, sc = ctrl.mean(), ctrl.std(ddof=1)
            ma, sa = alc.mean(), alc.std(ddof=1)
            print(f"  {canal:<6}{cond:<12}"
                  f"{mc:>+7.2f} +- {sc:>4.2f}{ma:>+8.2f} +- {sa:>4.2f}"
                  f"{mc-ma:>+9.2f}{len(ctrl):>7}{len(alc):>7}")


# =============================================================================
# VISUALIZACION — por hemisferio
# =============================================================================

def graficar_boxplots(df_pico, metodo, col_amp, ventana_label,
                      canales, hemi_label, out_file):
    """Boxplots de la metrica por grupo, por canal y condicion."""
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    np.random.seed(42)
    df = df_pico[df_pico["metodo"] == metodo]

    fig, axes = plt.subplots(
        len(CONDICIONES), len(canales),
        figsize=(5 * len(canales), 4.5 * len(CONDICIONES)),
        sharey=False
    )
    fig.suptitle(
        f"{ventana_label} por sujeto — metodo {metodo} — "
        f"hemisferio {hemi_label}\n"
        f"Metrica: {col_amp}",
        fontsize=13
    )

    for fila, cond in enumerate(CONDICIONES):
        for col_idx, canal in enumerate(canales):
            ax = axes[fila][col_idx]
            datos, etiquetas, cols_bp = [], [], []
            for grupo in ["control", "alcoholic"]:
                vals = df[(df["canal"] == canal) &
                          (df["condicion"] == cond) &
                          (df["grupo"] == grupo)][col_amp].dropna().values
                datos.append(vals)
                m = np.mean(vals) if len(vals) else np.nan
                sd = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
                etiquetas.append(
                    f"{grupo.capitalize()}\nn={len(vals)}\n{m:+.2f}+-{sd:.2f}")
                cols_bp.append(colores[grupo])

            bp = ax.boxplot(datos, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 2})
            for patch, color in zip(bp["boxes"], cols_bp):
                patch.set_facecolor(color); patch.set_alpha(0.5)
            for i, (vals, color) in enumerate(zip(datos, cols_bp)):
                jit = np.random.normal(i + 1, 0.07, size=len(vals))
                ax.scatter(jit, vals, alpha=0.35, color=color, s=12, zorder=3)

            ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.set_xticks([1, 2]); ax.set_xticklabels(etiquetas, fontsize=8)
            ax.set_title(f"Canal: {canal}\n{cond}", fontsize=10)
            ax.set_ylabel(f"{col_amp} (uV)")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"outputs/{out_file}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{out_file}'")


def graficar_latencia(df_all, metodo, col_lat, ventana_label,
                      canales, hemi_label, out_file):
    """Histogramas de latencia del maximo por grupo."""
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    df = df_all[df_all["metodo"] == metodo]

    fig, axes = plt.subplots(
        len(CONDICIONES), len(canales),
        figsize=(5 * len(canales), 4.5 * len(CONDICIONES)),
        sharex=True
    )
    fig.suptitle(
        f"Latencia del maximo — {ventana_label} — metodo {metodo} — "
        f"hemisferio {hemi_label}",
        fontsize=13
    )

    for fila, cond in enumerate(CONDICIONES):
        for col_idx, canal in enumerate(canales):
            ax = axes[fila][col_idx]
            for grupo in ["control", "alcoholic"]:
                lats = df[(df["canal"] == canal) &
                          (df["condicion"] == cond) &
                          (df["grupo"] == grupo)][col_lat].dropna().values
                ax.hist(lats, bins=15, alpha=0.5, color=colores[grupo],
                        label=grupo.capitalize(), density=True)
            ax.set_title(f"Canal: {canal}\n{cond}", fontsize=10)
            ax.set_xlabel("Latencia (ms)"); ax.set_ylabel("Densidad")
            ax.grid(True, alpha=0.3)
            if fila == 0 and col_idx == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"outputs/{out_file}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{out_file}'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 76)
    print("TPS -- Potenciales Evocados Visuales en Alcoholismo")
    print("Script 05: Extraccion de Metricas del c240 (y positividad tardia)")
    print("=" * 76)
    print(f"\nVentana PRIMARIA (c240):      {V_C240[0]}-{V_C240[1]} ms  "
          "(Zhang et al. 1997)")
    print(f"Ventana SECUNDARIA (c320):    {V_C320[0]}-{V_C320[1]} ms  "
          "(positividad tardia)")
    print(f"Metrica principal: MEDIA en ventana")

    # Extraer metricas de ambos metodos
    tablas = []
    for metodo, ruta in METODOS.items():
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontro '{ruta}'. Corre primero el Script 04.")
        print(f"\nCargando '{ruta}'...")
        pe = pd.read_parquet(ruta)
        print(f"  Sujetos: {pe['sujeto'].nunique()} "
              f"({pe[pe['grupo']=='alcoholic']['sujeto'].nunique()} alc + "
              f"{pe[pe['grupo']=='control']['sujeto'].nunique()} ctrl)")
        tablas.append(extraer_todos(pe, metodo))

    df_all = pd.concat(tablas, ignore_index=True)
    df_all.to_csv(SALIDA_CSV, index=False)
    print(f"\nMetricas guardadas en '{SALIDA_CSV}'  ({len(df_all)} filas)")

    # Resumen descriptivo — homogeneo (principal)
    imprimir_resumen(df_all, "homogeneo", "media_c240",
                     f"c240 ({V_C240[0]}-{V_C240[1]} ms)")
    imprimir_resumen(df_all, "homogeneo", "media_c320",
                     f"positividad tardia ({V_C320[0]}-{V_C320[1]} ms)")

    # Graficos — divididos por hemisferio
    print("\nGenerando graficos...")
    for canales, label, sufijo in [
        (CANALES_DERECHO,   "derecho",   "derecho"),
        (CANALES_IZQUIERDO, "izquierdo", "izquierdo"),
    ]:
        # Boxplots c240 — homogeneo
        graficar_boxplots(
            df_all, "homogeneo", "media_c240",
            f"c240 ({V_C240[0]}-{V_C240[1]} ms)",
            canales, label,
            f"figura_boxplot_{sufijo}_c240_hom.png")

        # Boxplots c320 — homogeneo
        graficar_boxplots(
            df_all, "homogeneo", "media_c320",
            f"c320 ({V_C320[0]}-{V_C320[1]} ms)",
            canales, label,
            f"figura_boxplot_{sufijo}_c320_hom.png")

        # Latencia c240 — homogeneo
        graficar_latencia(
            df_all, "homogeneo", "lat_max_c240",
            f"c240 ({V_C240[0]}-{V_C240[1]} ms)",
            canales, label,
            f"figura_latencia_{sufijo}_c240.png")
        
        # Latencia c320 — homogeneo
        graficar_latencia(
            df_all, "homogeneo", "lat_max_c320",
            f"c320 ({V_C320[0]}-{V_C320[1]} ms)",
            canales, label,
            f"figura_latencia_{sufijo}_c320.png")

    print("\n[OK] Script 05 finalizado.")
    print("\nProximo paso: Script 06 (estadistica).")