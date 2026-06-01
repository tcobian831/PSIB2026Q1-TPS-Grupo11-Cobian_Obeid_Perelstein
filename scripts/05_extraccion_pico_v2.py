"""
==============================================================================
TPS - Procesamiento de Señales Biomédicas
Potenciales Evocados Visuales en Sujetos con Alcoholismo
Grupo 11: Cobián, Obeid, Perelstein

Script 05: Extracción del Pico, Jerarquía de Promedios
           y comparación Homogéneo vs Inhomogéneo
==============================================================================

Propósito:
    Para cada sujeto, canal y condición, extraer la amplitud máxima del PE y
    el instante en que ocurre (latencia), a partir de los dos métodos de
    promediado del Script 04 (homogéneo e inhomogéneo) y con DOS medidas:

    (a) GLOBAL  — máximo positivo en toda la señal (sin ventana).
        Es la mirada exploratoria: deja ver DÓNDE pica realmente el PE. Su
        histograma de latencias muestra que el máximo global suele caer en el
        P1 (~100 ms) o en la onda lenta tardía (>400 ms), NO en el c240. Por
        eso el máximo global, por sí solo, no aísla el componente de interés.

    (b) VENTANA — máximo positivo dentro de 200–350 ms.
        Ventana justificada por el Grand Average (la positividad de los
        controles pica ~250–350 ms, algo más tarde que la ventana clásica
        220–260 ms de Zhang et al.). Excluye el P1 temprano y la onda lenta,
        de modo que la amplitud medida corresponde al c240/VMP.

    Se usa el MÁXIMO POSITIVO (no absoluto): el c240 es positivo, y el máximo
    absoluto capturaría la deflexión negativa de ~180 ms (grande en alcohólicos).

    Reportar la amplitud con las dos medidas permite mostrar que la conclusión
    (control > alcohólico) se sostiene midamos con ventana o sin ella.

Jerarquía de promedios (para cada método, sobre la medida en ventana):
    A(c,p,k)  amplitud por canal c, sujeto p, condición k   (un valor/sujeto)
        ↓ promediar sobre sujetos p
    A(c,k)    amplitud representativa por canal y condición
        ↓ promediar sobre canales c
    A(k)      amplitud representativa por grupo y condición

Comparación entre grupos (sin estadística avanzada):
    Razón de medias = amplitud media control / amplitud media alcohólico.
    Cuanto mayor que 1, más separados están los grupos en la dirección de la
    hipótesis. El solapamiento de los boxplots complementa esta lectura de
    forma visual. El test estadístico formal se hace en el Script 06.

Entrada:  outputs/eeg_PE_homogeneo.parquet
          outputs/eeg_PE_inhomogeneo.parquet     (ambos del Script 04)
Salida:   outputs/eeg_c240_extraido.csv          (pico por sujeto: global y ventana)
          outputs/figura_pico_amplitud_homogeneo.png
          outputs/figura_pico_amplitud_inhomogeneo.png
          outputs/figura_pico_latencia.png
          outputs/figura_comparacion_grupos.png

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

CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES     = ["S1 obj", "S2 nomatch"]

# Medida GLOBAL: máximo positivo en toda la señal (sin ventana).
# Subir T_MIN_MS a ~20-30 ms solo si aparece artefacto de estímulo al inicio.
T_MIN_MS  = 0
M_MIN     = int(round(T_MIN_MS / 1000 * FS))

# Medida VENTANA: máximo positivo dentro de esta ventana (ms).
VENTANA_ANALISIS = (200, 350)
M_VENT_LO = int(round(VENTANA_ANALISIS[0] / 1000 * FS))   # 51
M_VENT_HI = int(round(VENTANA_ANALISIS[1] / 1000 * FS))   # 90

# Ventana clásica del paper, SOLO para marcarla en los gráficos y reportar
# cuántos picos globales caen dentro.
VENTANA_PAPER = (220, 260)  # ms

METODOS = {
    "homogeneo":   Path("../outputs/eeg_PE_homogeneo.parquet"),
    "inhomogeneo": Path("../outputs/eeg_PE_inhomogeneo.parquet"),
}

SALIDA_CSV = Path("../outputs/eeg_c240_extraido.csv")

# =============================================================================
# EXTRACCIÓN DEL PICO
# =============================================================================

def extraer_pico(senal: np.ndarray, m_ini: int, m_fin: int) -> tuple:
    """
    Máximo POSITIVO y su latencia dentro de senal[m_ini:m_fin].

    Args:
        senal: array (N_SAMPLES,)
        m_ini, m_fin: índices de muestra que delimitan la búsqueda [m_ini, m_fin)
    Retorna:
        (amplitud_uV, latencia_ms)
    """
    seg = senal[m_ini:m_fin]
    idx = int(np.argmax(seg))
    amplitud = float(seg[idx])
    latencia = float((idx + m_ini) / FS * 1000)
    return amplitud, latencia


def extraer_todos_picos(pe_ind: pd.DataFrame, metodo: str) -> pd.DataFrame:
    """
    Para cada combinación sujeto×canal×condición extrae el pico con las dos
    medidas: GLOBAL (toda la señal) y VENTANA (200–350 ms).

    Retorna DataFrame con columnas:
        sujeto, grupo, canal, condicion, metodo,
        amp_global, lat_global, amp_ventana, lat_ventana,
        global_en_ventana_paper, n_trials
    """
    resultados = []
    grupos = pe_ind.groupby(["sujeto", "grupo", "canal", "condicion"])
    print(f"  [{metodo}] {len(grupos):,} combinaciones...")

    for (sujeto, grupo, canal, cond), sub in grupos:
        s = sub.sort_values("muestra")["PE_uV"].values
        if len(s) != N_SAMPLES:
            continue

        amp_g, lat_g = extraer_pico(s, M_MIN, N_SAMPLES)
        amp_v, lat_v = extraer_pico(s, M_VENT_LO, M_VENT_HI + 1)
        n_trials = (sub["n_trials"].iloc[0]
                    if "n_trials" in sub.columns else np.nan)

        resultados.append({
            "sujeto":      sujeto,
            "grupo":       grupo,
            "canal":       canal,
            "condicion":   cond,
            "metodo":      metodo,
            "amp_global":  amp_g,
            "lat_global":  lat_g,
            "amp_ventana": amp_v,
            "lat_ventana": lat_v,
            "global_en_ventana_paper": VENTANA_PAPER[0] <= lat_g <= VENTANA_PAPER[1],
            "n_trials":    n_trials,
        })

    return pd.DataFrame(resultados)


# =============================================================================
# JERARQUÍA DE PROMEDIOS
# =============================================================================

def calcular_jerarquia(df_pico: pd.DataFrame,
                       col_amp: str = "amp_ventana",
                       col_lat: str = "lat_ventana") -> tuple:
    """
    Construye la jerarquía A(c,p,k) → A(c,k) → A(k) para un método y una medida.

    Retorna:
        nivel2: A(c,k) y t(c,k)   (promedio sobre sujetos)
        nivel3: A(k)   y t(k)     (promedio sobre canales)
    """
    nivel2 = (
        df_pico
        .groupby(["grupo", "canal", "condicion"])
        .agg(
            A_media=(col_amp, "mean"),
            A_sd=(col_amp, "std"),
            t_media=(col_lat, "mean"),
            t_sd=(col_lat, "std"),
            n_sujetos=("sujeto", "count"),
        )
        .reset_index()
    )
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


def imprimir_jerarquia(nivel2: pd.DataFrame, nivel3: pd.DataFrame, titulo: str):
    print("\n" + "=" * 72)
    print(f"JERARQUÍA DE PROMEDIOS — {titulo}")
    print("=" * 72)

    print("\n--- NIVEL 2: A(c,k) — promedio por canal y condición ---")
    print(f"  {'Grupo':<11}{'Canal':<6}{'Condición':<12}"
          f"{'Amplitud (µV)':>17}{'Latencia (ms)':>17}{'N':>5}")
    print("  " + "-" * 66)
    for _, r in nivel2.sort_values(["condicion", "grupo", "canal"]).iterrows():
        print(f"  {r['grupo']:<11}{r['canal']:<6}{r['condicion']:<12}"
              f"{r['A_media']:>+9.3f} ± {r['A_sd']:>5.3f}"
              f"{r['t_media']:>9.1f} ± {r['t_sd']:>5.1f}"
              f"{int(r['n_sujetos']):>5}")

    print("\n--- NIVEL 3: A(k) — amplitud representativa por grupo ---")
    print(f"  {'Grupo':<11}{'Condición':<12}{'A global (µV)':>17}{'t global (ms)':>17}")
    print("  " + "-" * 56)
    for _, r in nivel3.sort_values(["condicion", "grupo"]).iterrows():
        print(f"  {r['grupo']:<11}{r['condicion']:<12}"
              f"{r['A_global']:>+9.3f} ± {r['A_global_sd']:>5.3f}"
              f"{r['t_global']:>9.1f} ± {r['t_global_sd']:>5.1f}")


# =============================================================================
# COMPARACIÓN ENTRE GRUPOS (razón de medias — sin estadística avanzada)
# =============================================================================

def comparar_grupos(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada método × medida × canal × condición: media de cada grupo y
    razón control/alcohólico. Razón > 1 ⇒ control mayor (hipótesis).
    """
    filas = []
    for metodo in df_all["metodo"].unique():
        for medida, col in [("global", "amp_global"), ("ventana", "amp_ventana")]:
            for canal in CANALES_INTERES:
                for cond in CONDICIONES:
                    sub = df_all[(df_all["metodo"] == metodo) &
                                 (df_all["canal"] == canal) &
                                 (df_all["condicion"] == cond)]
                    ctrl = sub[sub["grupo"] == "control"][col].dropna()
                    alc  = sub[sub["grupo"] == "alcoholic"][col].dropna()
                    if len(ctrl) == 0 or len(alc) == 0:
                        continue
                    mc, ma = ctrl.mean(), alc.mean()
                    filas.append({
                        "metodo": metodo, "medida": medida,
                        "canal": canal, "condicion": cond,
                        "control_media": mc, "alcoholic_media": ma,
                        "razon": (mc / ma) if ma > 1e-9 else np.nan,
                        "n_ctrl": len(ctrl), "n_alc": len(alc),
                    })
    comp = pd.DataFrame(filas)

    print("\n" + "=" * 78)
    print("COMPARACIÓN ENTRE GRUPOS — razón de medias control / alcohólico")
    print("razón > 1 ⇒ control mayor (hipótesis). Más alto = más separación.")
    print("=" * 78)
    for metodo in df_all["metodo"].unique():
        for medida in ["global", "ventana"]:
            sub = comp[(comp["metodo"] == metodo) & (comp["medida"] == medida)]
            if sub.empty:
                continue
            print(f"\n  Método {metodo} — medida {medida}:")
            print(f"    {'Canal':<6}{'Condición':<12}"
                  f"{'Control':>10}{'Alcohólico':>12}{'Razón C/A':>12}")
            print("    " + "-" * 50)
            for _, r in sub.iterrows():
                print(f"    {r['canal']:<6}{r['condicion']:<12}"
                      f"{r['control_media']:>+10.2f}{r['alcoholic_media']:>+12.2f}"
                      f"{r['razon']:>11.2f}x")
    print("=" * 78)
    return comp


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def graficar_amplitud_pico(df_pico: pd.DataFrame, metodo: str,
                            col_amp: str = "amp_ventana"):
    """Boxplots de la amplitud del pico por grupo, por canal y condición."""
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    np.random.seed(42)

    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharey=False
    )
    fig.suptitle(
        f"Amplitud del pico del PE por sujeto — método {metodo}\n"
        f"Máximo positivo en ventana {VENTANA_ANALISIS[0]}–{VENTANA_ANALISIS[1]} ms",
        fontsize=13
    )

    for fila, cond in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]
            datos, etiquetas, cols_bp = [], [], []
            for grupo in ["control", "alcoholic"]:
                vals = df_pico[(df_pico["canal"] == canal) &
                               (df_pico["condicion"] == cond) &
                               (df_pico["grupo"] == grupo)][col_amp].dropna().values
                datos.append(vals)
                m = np.mean(vals) if len(vals) else np.nan
                sd = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
                etiquetas.append(f"{grupo.capitalize()}\nn={len(vals)}\n{m:+.2f}±{sd:.2f}")
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
            ax.set_ylabel("Amplitud pico (µV)")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = f"../outputs/figura_pico_amplitud_{metodo}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figura guardada: '{out}'")


def graficar_latencia_pico(df_all: pd.DataFrame, metodo_ref: str = "homogeneo"):
    """
    Histogramas de latencia del MÁXIMO GLOBAL por grupo. Muestra que el pico
    global se dispersa (P1 ~100 ms, onda lenta tardía) y rara vez cae en la
    ventana del paper, justificando el uso de una ventana de análisis.
    """
    df_pico = df_all[df_all["metodo"] == metodo_ref]
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    fig, axes = plt.subplots(
        len(CONDICIONES), len(CANALES_INTERES),
        figsize=(5 * len(CANALES_INTERES), 4 * len(CONDICIONES)),
        sharex=True
    )
    fig.suptitle(
        f"Distribución de latencias del máximo global — método {metodo_ref}\n"
        f"Líneas punteadas: ventana clásica {VENTANA_PAPER[0]}–{VENTANA_PAPER[1]} ms "
        "(Zhang et al.)",
        fontsize=13
    )

    for fila, cond in enumerate(CONDICIONES):
        for col, canal in enumerate(CANALES_INTERES):
            ax = axes[fila][col]
            for grupo in ["control", "alcoholic"]:
                lats = df_pico[(df_pico["canal"] == canal) &
                               (df_pico["condicion"] == cond) &
                               (df_pico["grupo"] == grupo)]["lat_global"].dropna().values
                ax.hist(lats, bins=20, alpha=0.5, color=colores[grupo],
                        label=grupo.capitalize(), density=True)
            for xline in VENTANA_PAPER:
                ax.axvline(xline, color="#f59e0b", linestyle="--", linewidth=1.4)
            ax.set_title(f"Canal: {canal}\n{cond}", fontsize=10)
            ax.set_xlabel("Latencia (ms)"); ax.set_ylabel("Densidad")
            ax.grid(True, alpha=0.3)
            if fila == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("../outputs/figura_pico_latencia.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_pico_latencia.png'")


def graficar_comparacion_grupos(comp: pd.DataFrame, metodo: str = "homogeneo"):
    """
    Barras de la amplitud media de cada grupo (control vs alcohólico) por canal,
    medida en ventana, con la razón C/A anotada encima. Lectura intuitiva del
    tamaño de la diferencia sin estadística.
    """
    sub = comp[(comp["metodo"] == metodo) & (comp["medida"] == "ventana")]
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    fig, axes = plt.subplots(1, len(CONDICIONES),
                             figsize=(7 * len(CONDICIONES), 5))
    if len(CONDICIONES) == 1:
        axes = [axes]
    fig.suptitle(
        f"Amplitud media del c240 por grupo — método {metodo}, ventana "
        f"{VENTANA_ANALISIS[0]}–{VENTANA_ANALISIS[1]} ms\n"
        "Número sobre las barras: razón control / alcohólico",
        fontsize=13
    )
    x = np.arange(len(CANALES_INTERES)); ancho = 0.38

    for ax, cond in zip(axes, CONDICIONES):
        sc = sub[sub["condicion"] == cond].set_index("canal").reindex(CANALES_INTERES)
        ctrl = sc["control_media"].values
        alc  = sc["alcoholic_media"].values
        razon = sc["razon"].values
        ax.bar(x - ancho/2, ctrl, ancho, label="Control",
               color=colores["control"], alpha=0.8)
        ax.bar(x + ancho/2, alc, ancho, label="Alcohólico",
               color=colores["alcoholic"], alpha=0.8)
        for xi, (c, r) in enumerate(zip(ctrl, razon)):
            if not np.isnan(r):
                ax.text(xi, c + 0.15, f"{r:.1f}x", ha="center",
                        fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(CANALES_INTERES)
        ax.set_title(f"Condición: {cond}", fontsize=11)
        ax.set_ylabel("Amplitud media del pico (µV)")
        ax.legend(fontsize=10); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/figura_comparacion_grupos.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figura guardada: 'figura_comparacion_grupos.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 72)
    print("TPS — Potenciales Evocados Visuales en Alcoholismo")
    print("Script 05: Extracción del Pico (Homogéneo vs Inhomogéneo)")
    print("=" * 72)
    print(f"\nMedida GLOBAL : máximo positivo desde {T_MIN_MS} ms (sin ventana).")
    print(f"Medida VENTANA: máximo positivo en {VENTANA_ANALISIS[0]}–"
          f"{VENTANA_ANALISIS[1]} ms.")

    # Extraer picos de ambos métodos
    tablas = []
    for metodo, ruta in METODOS.items():
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró '{ruta}'. Corré primero el Script 04.")
        print(f"\nCargando '{ruta}'...")
        pe = pd.read_parquet(ruta)
        tablas.append(extraer_todos_picos(pe, metodo))

    df_all = pd.concat(tablas, ignore_index=True)
    df_all.to_csv(SALIDA_CSV, index=False)
    print(f"\nPicos guardados en '{SALIDA_CSV}'  ({len(df_all)} filas)")

    # % de máximos globales que caen en la ventana clásica del paper
    print("\n% de máximos GLOBALES dentro de 220–260 ms (ventana del paper):")
    pct = (df_all.groupby(["metodo", "grupo"])["global_en_ventana_paper"]
           .mean().mul(100).round(1))
    print(pct.to_string())

    # Jerarquía por método (sobre la medida en ventana)
    for metodo in METODOS:
        n2, n3 = calcular_jerarquia(df_all[df_all["metodo"] == metodo],
                                    col_amp="amp_ventana", col_lat="lat_ventana")
        imprimir_jerarquia(n2, n3, f"método {metodo.upper()} (ventana 200–350 ms)")

    # Comparación entre grupos (razón de medias)
    comp = comparar_grupos(df_all)

    # Gráficos
    print("\nGenerando gráficos...")
    for metodo in METODOS:
        graficar_amplitud_pico(df_all[df_all["metodo"] == metodo], metodo,
                               col_amp="amp_ventana")
    graficar_latencia_pico(df_all, metodo_ref="homogeneo")
    graficar_comparacion_grupos(comp, metodo="homogeneo")

    print("\n[OK] Script 05 finalizado.")
    print("\nPróximo paso:")
    print("  - Mirá que la razón C/A se mantenga > 1 tanto en 'global' como en")
    print("    'ventana' (robustez de la conclusión).")
    print("  - El Script 06 corre el análisis final sobre 'amp_ventana'.")
