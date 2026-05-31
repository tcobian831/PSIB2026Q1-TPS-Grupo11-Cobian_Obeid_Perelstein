"""
==============================================================================
Script 05b v3: Extracción de pico GLOBAL del PE (sin ventana c240)
==============================================================================

Propósito
---------
Para cada (sujeto, canal, condición) tomar el PE individual y extraer el
pico de mayor amplitud en TODO el rango temporal post-estímulo, junto con
la latencia en la que ocurre. NO se restringe a la ventana 220–260 ms.

Sirve como verificación independiente del componente c240/VMP:
  - Si el pico cae sistemáticamente cerca de 240 ms en canales
    temporo-occipitales derechos, la ventana 220–260 ms está bien elegida.
  - Si cae en otro lado, hay que reconsiderar la ventana.

Se calculan tres definiciones de pico para que la elección sea defendible:
  A) max:  el valor más positivo  (estricto al anteproyecto).
  B) min:  el valor más negativo  (por si la referencia invierte el componente).
  C) absmax: el de mayor magnitud absoluta, sin importar signo.

Cada una con su latencia en ms.

Excluye los primeros 30 ms para evitar contaminación del baseline y de
transitorios del filtro filtfilt.

Entrada
-------
  outputs/eeg_PE_individual_v3.parquet      (preferido)
  outputs/eeg_PE_individual_v2.parquet      (fallback)

Salidas
-------
  outputs/eeg_pico_global_extraido_v3.csv
  outputs/figura_pico_global_latencias_v3.png
  outputs/figura_pico_global_amplitudes_v3.png

Uso
---
  python .\\scripts\\05b_pico_global_v3.py
==============================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FS = 256
N_SAMPLES = 256

CANALES_INTERES = ["P8", "PO8", "T8", "TP8"]
CONDICIONES_PRINCIPALES = ["S1 obj", "S2 nomatch"]
CONDICIONES_TODAS = ["S1 obj", "S2 nomatch", "S2 match"]

# Ventana c240 (solo se usa para marcar referencia en las figuras)
T_C240_INI_MS = 220
T_C240_FIN_MS = 260

# Excluir los primeros milisegundos para evitar transitorios del filtro
# y la zona de baseline temprano usada en el preprocesamiento.
T_EXCLUIR_INICIO_MS = 30


def get_project_dirs():
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "scripts":
        project_dir = here.parent.parent
    else:
        project_dir = here.parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return project_dir, output_dir


PROJECT_DIR, OUTPUT_DIR = get_project_dirs()


def first_existing(paths, label):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    msg = "\n".join(f"  - {Path(p)}" for p in paths)
    raise FileNotFoundError(f"No se encontró {label}. Rutas:\n{msg}")


ENTRADA = first_existing(
    [
        OUTPUT_DIR / "eeg_PE_individual_v3.parquet",
        OUTPUT_DIR / "eeg_PE_individual_homogeneo_v3.parquet",
        OUTPUT_DIR / "eeg_PE_individual_v2.parquet",
        PROJECT_DIR / "eeg_PE_individual_v3.parquet",
    ],
    "PE individual",
)

SALIDA_CSV = OUTPUT_DIR / "eeg_pico_global_extraido_v3.csv"


# =============================================================================
# FUNCIONES
# =============================================================================

def ms_a_muestra(t_ms):
    return int(t_ms / 1000 * FS)


def extraer_picos_sujeto(df_sce, m_min):
    """
    Extrae los tres picos (max, min, absmax) y sus latencias para un PE.

    df_sce: DataFrame con las columnas muestra y PE_uV de UN sujeto×canal×condición.
    m_min:  muestra mínima a considerar (excluir las primeras m_min muestras).

    Devuelve un dict con 6 valores.
    """
    df = df_sce.sort_values("muestra")
    pe = df["PE_uV"].values
    muestras = df["muestra"].values

    # Recorte: solo considerar muestras a partir de m_min
    mask = muestras >= m_min
    if mask.sum() == 0:
        return {
            "amp_max_uV": np.nan, "lat_max_ms": np.nan,
            "amp_min_uV": np.nan, "lat_min_ms": np.nan,
            "amp_absmax_uV": np.nan, "lat_absmax_ms": np.nan,
            "n_muestras_consideradas": 0,
        }

    pe_v = pe[mask]
    mu_v = muestras[mask]

    # A) Pico positivo
    idx_max = int(np.argmax(pe_v))
    amp_max = float(pe_v[idx_max])
    lat_max = float(mu_v[idx_max]) / FS * 1000

    # B) Pico negativo
    idx_min = int(np.argmin(pe_v))
    amp_min = float(pe_v[idx_min])
    lat_min = float(mu_v[idx_min]) / FS * 1000

    # C) Pico por magnitud absoluta
    idx_abs = int(np.argmax(np.abs(pe_v)))
    amp_abs = float(pe_v[idx_abs])
    lat_abs = float(mu_v[idx_abs]) / FS * 1000

    return {
        "amp_max_uV": amp_max,
        "lat_max_ms": lat_max,
        "amp_min_uV": amp_min,
        "lat_min_ms": lat_min,
        "amp_absmax_uV": amp_abs,
        "lat_absmax_ms": lat_abs,
        "n_muestras_consideradas": int(mask.sum()),
    }


def extraer_todos(pe_ind, m_min):
    filas = []
    grupos = pe_ind.groupby(
        ["sujeto", "grupo", "canal", "condicion"], observed=True
    )
    n_total = len(grupos)
    print(f"  Procesando {n_total:,} combinaciones sujeto×canal×condición...")
    for idx, ((sujeto, grupo, canal, cond), df_sub) in enumerate(grupos):
        if idx % 1000 == 0 and idx > 0:
            print(f"    {idx:,}/{n_total:,}")
        vals = extraer_picos_sujeto(df_sub, m_min)
        filas.append({
            "sujeto": sujeto,
            "grupo": grupo,
            "canal": canal,
            "condicion": cond,
            **vals,
        })
    return pd.DataFrame(filas)


def resumen_picos(df_pico, condiciones):
    print("\n" + "=" * 90)
    print("RESUMEN PICO GLOBAL (sin restricción de ventana)")
    print(f"Excluye primeros {T_EXCLUIR_INICIO_MS} ms")
    print("=" * 90)

    for cond in condiciones:
        print(f"\nCondición: {cond}")
        print(f"  {'Canal':<6} {'Grupo':<11} "
              f"{'lat max ms':>13} {'lat min ms':>13} {'lat |abs| ms':>15} "
              f"{'% pico=+':>10}")
        for canal in CANALES_INTERES:
            for grupo in ["control", "alcoholic"]:
                sub = df_pico[
                    (df_pico["canal"] == canal)
                    & (df_pico["condicion"] == cond)
                    & (df_pico["grupo"] == grupo)
                ]
                if sub.empty:
                    continue
                lat_max = sub["lat_max_ms"].mean()
                lat_min = sub["lat_min_ms"].mean()
                lat_abs = sub["lat_absmax_ms"].mean()
                # ¿Qué fracción de sujetos tiene el pico absmax POSITIVO?
                signo_pos = (sub["amp_absmax_uV"] > 0).mean() * 100
                print(f"  {canal:<6} {grupo:<11} "
                      f"{lat_max:>12.1f}  {lat_min:>12.1f}  {lat_abs:>14.1f}  "
                      f"{signo_pos:>9.1f}%")
    print("=" * 90)


def plot_latencias(df_pico, condiciones):
    """
    Histograma de latencias por canal/condición/grupo, en tres paneles:
    una columna por tipo de pico (max, min, absmax).
    Marca con una línea naranja la ventana c240.
    """
    tipos = [
        ("lat_max_ms", "Pico máximo (positivo)"),
        ("lat_min_ms", "Pico mínimo (negativo)"),
        ("lat_absmax_ms", "Pico por magnitud absoluta"),
    ]
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}

    n_rows = len(CANALES_INTERES) * len(condiciones)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(15, 2.6 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Latencia del pico GLOBAL del PE (excluye primeros {T_EXCLUIR_INICIO_MS} ms)\n"
        f"Banda naranja: ventana c240 = {T_C240_INI_MS}–{T_C240_FIN_MS} ms",
        fontsize=12,
    )

    t_max_ms = N_SAMPLES / FS * 1000  # 1000 ms
    bins = np.linspace(0, t_max_ms, 26)

    fila = 0
    for cond in condiciones:
        for canal in CANALES_INTERES:
            for col, (col_lat, etiqueta) in enumerate(tipos):
                ax = axes[fila, col]
                for grupo in ["control", "alcoholic"]:
                    lats = df_pico[
                        (df_pico["canal"] == canal)
                        & (df_pico["condicion"] == cond)
                        & (df_pico["grupo"] == grupo)
                    ][col_lat].dropna().values
                    if len(lats) == 0:
                        continue
                    ax.hist(lats, bins=bins, alpha=0.45,
                            color=colores[grupo], label=f"{grupo} (n={len(lats)})")
                ax.axvspan(T_C240_INI_MS, T_C240_FIN_MS,
                           alpha=0.2, color="orange")
                ax.set_title(f"{canal} — {cond}\n{etiqueta}", fontsize=8)
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel("n sujetos")
                if fila == n_rows - 1:
                    ax.set_xlabel("Latencia (ms)")
                if fila == 0 and col == 0:
                    ax.legend(fontsize=7)
            fila += 1

    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_pico_global_latencias_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")


def plot_amplitudes(df_pico, condiciones):
    """Boxplots de amplitud por canal/condición/grupo y por tipo de pico."""
    tipos = [
        ("amp_max_uV", "Pico positivo (max)"),
        ("amp_min_uV", "Pico negativo (min)"),
        ("amp_absmax_uV", "Pico por |amp| (con signo)"),
    ]
    colores = {"control": "#2563eb", "alcoholic": "#dc2626"}
    rng = np.random.default_rng(42)

    n_rows = len(condiciones) * 3
    fig, axes = plt.subplots(
        n_rows, len(CANALES_INTERES),
        figsize=(4 * len(CANALES_INTERES), 3 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        "Amplitud del pico GLOBAL por canal, condición y tipo de pico",
        fontsize=12,
    )

    fila = 0
    for cond in condiciones:
        for tipo_idx, (col_amp, etiqueta) in enumerate(tipos):
            for j, canal in enumerate(CANALES_INTERES):
                ax = axes[fila, j]
                data, labels, cols = [], [], []
                for grupo in ["control", "alcoholic"]:
                    vals = df_pico[
                        (df_pico["canal"] == canal)
                        & (df_pico["condicion"] == cond)
                        & (df_pico["grupo"] == grupo)
                    ][col_amp].dropna().values
                    data.append(vals)
                    labels.append(f"{grupo}\n(n={len(vals)})")
                    cols.append(colores[grupo])
                bp = ax.boxplot(data, patch_artist=True,
                                medianprops={"color": "black", "linewidth": 2})
                for patch, c in zip(bp["boxes"], cols):
                    patch.set_facecolor(c); patch.set_alpha(0.55)
                for k, vals in enumerate(data):
                    if len(vals):
                        ax.scatter(rng.normal(k + 1, 0.06, len(vals)),
                                   vals, s=12, alpha=0.4, color=cols[k])
                ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
                ax.set_xticks([1, 2]); ax.set_xticklabels(labels, fontsize=7)
                ax.set_title(f"{canal} — {cond}\n{etiqueta}", fontsize=8)
                ax.set_ylabel("µV")
                ax.grid(True, axis="y", alpha=0.3)
            fila += 1

    plt.tight_layout()
    ruta = OUTPUT_DIR / "figura_pico_global_amplitudes_v3.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta}")


def diagnostico_ventana_c240(df_pico, condiciones):
    """
    Para cada canal/condición/grupo, calcula qué porcentaje de sujetos
    tiene el pico global cayendo dentro de la ventana c240 (220–260 ms).
    Sirve para defender que la ventana fue una elección razonable.
    """
    print("\n" + "=" * 90)
    print(f"¿El pico global cae en la ventana c240 ({T_C240_INI_MS}–{T_C240_FIN_MS} ms)?")
    print("=" * 90)
    print(f"  {'Canal':<6} {'Cond':<12} {'Grupo':<11} "
          f"{'% pico+ en c240':>17} {'% pico- en c240':>17} "
          f"{'% |abs| en c240':>17}")
    for cond in condiciones:
        for canal in CANALES_INTERES:
            for grupo in ["control", "alcoholic"]:
                sub = df_pico[
                    (df_pico["canal"] == canal)
                    & (df_pico["condicion"] == cond)
                    & (df_pico["grupo"] == grupo)
                ]
                if sub.empty:
                    continue
                pct_pos = ((sub["lat_max_ms"] >= T_C240_INI_MS)
                           & (sub["lat_max_ms"] <= T_C240_FIN_MS)).mean() * 100
                pct_neg = ((sub["lat_min_ms"] >= T_C240_INI_MS)
                           & (sub["lat_min_ms"] <= T_C240_FIN_MS)).mean() * 100
                pct_abs = ((sub["lat_absmax_ms"] >= T_C240_INI_MS)
                           & (sub["lat_absmax_ms"] <= T_C240_FIN_MS)).mean() * 100
                print(f"  {canal:<6} {cond:<12} {grupo:<11} "
                      f"{pct_pos:>16.1f}% {pct_neg:>16.1f}% {pct_abs:>16.1f}%")
    print("=" * 90)


# =============================================================================
# MAIN
# =============================================================================

def filter_existing_conditions(df, desired):
    present = list(df["condicion"].dropna().unique())
    return [c for c in desired if c in present]


if __name__ == "__main__":
    print("=" * 70)
    print("Script 05b v3: Pico GLOBAL del PE (sin ventana c240)")
    print("=" * 70)

    print(f"\nCargando: {ENTRADA}")
    pe_ind = pd.read_parquet(ENTRADA)
    pe_ind = pe_ind[pe_ind["canal"].isin(CANALES_INTERES)].copy()
    condiciones = filter_existing_conditions(pe_ind, CONDICIONES_TODAS)
    pe_ind = pe_ind[pe_ind["condicion"].isin(condiciones)].copy()

    print(f"  Sujetos: {pe_ind['sujeto'].nunique()}")
    print(f"  Canales: {sorted(pe_ind['canal'].unique())}")
    print(f"  Condiciones: {condiciones}")

    m_min = ms_a_muestra(T_EXCLUIR_INICIO_MS)
    print(f"\nExcluyendo primeras {m_min} muestras "
          f"({T_EXCLUIR_INICIO_MS} ms iniciales).")

    df_pico = extraer_todos(pe_ind, m_min)

    print("\nGuardando CSV...")
    df_pico.to_csv(SALIDA_CSV, index=False)
    print(f"  {SALIDA_CSV}")
    print(f"  Filas: {len(df_pico):,}")

    resumen_picos(df_pico, condiciones)
    diagnostico_ventana_c240(df_pico, condiciones)

    print("\nGenerando figuras...")
    plot_latencias(df_pico, condiciones)
    plot_amplitudes(df_pico, condiciones)

    print("\n[OK] Script 05b v3 finalizado.")
    print("\nInterpretación esperada:")
    print("  - Mirá la tabla 'diagnostico_ventana_c240'.")
    print("  - Si los % en c240 son altos en P8/PO8/TP8/T8 (sobre todo en S1),")
    print("    la ventana 220-260 ms está bien justificada.")
    print("  - Si son bajos, hay que revisar la ventana o el preprocesamiento.")
