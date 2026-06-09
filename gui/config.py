"""
config.py — Constantes compartidas por toda la GUI.

Replica los bloques CONFIGURACIÓN de los Scripts 03/04/05/06 en un único lugar
(canales, condiciones, ventanas temporales, SEED, paleta) para que la app use
exactamente los mismos valores que el pipeline de análisis.
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# RUTAS
# =============================================================================
# config.py vive en  <root>/gui/  ->  el proyecto es el padre de gui/.
# Resolvemos rutas absolutas para que la app corra desde cualquier directorio.
ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

# =============================================================================
# PARÁMETROS DE ADQUISICIÓN  (Scripts 03/04/05)
# =============================================================================
FS = 256            # Hz
N_SAMPLES = 256     # muestras por trial
EPS = 1e-12

# Vector de tiempo en ms (estímulo en t=0, muestra 0) — idéntico a los scripts
TIEMPO_MS = [i / FS * 1000.0 for i in range(N_SAMPLES)]

# =============================================================================
# CANALES Y CONDICIONES  (Scripts 03/04/05/06)
# =============================================================================
CANALES_DERECHO = ["P8", "PO8", "T8", "TP8"]        # hemisferio derecho (paper)
CANALES_IZQUIERDO = ["P7", "PO7", "T7", "TP7"]      # homólogos izquierdos
CANALES_INTERES = CANALES_DERECHO + CANALES_IZQUIERDO

# Pares homólogos R-L (para etiquetar hemisferio)
PARES_HEMISFERICOS = [("P8", "P7"), ("PO8", "PO7"), ("T8", "T7"), ("TP8", "TP7")]

CONDICIONES = ["S1 obj", "S2 nomatch"]

GRUPOS = ["control", "alcoholic"]


def hemisferio_de(canal: str) -> str:
    """Devuelve 'derecho' / 'izquierdo' para un canal (igual que los scripts)."""
    return "derecho" if canal in CANALES_DERECHO else "izquierdo"


# =============================================================================
# BALANCEO DE GRUPOS  (Script 04 v2: igualar_grupos, SEED=42)
# =============================================================================
SEED_DEFAULT = 42      # semilla de reproducibilidad del Script 04
N_CONTROL = 45         # controles disponibles (fijos; nunca se submuestrean)
N_ALC_MIN = 45         # extremo inferior del slider = grupos igualados 45/45
N_ALC_MAX = 77         # extremo superior = muestra alcohólica completa

# =============================================================================
# VENTANAS TEMPORALES  (Scripts 05/06)
# =============================================================================
# PRIMARIA: c240 / VMP de Zhang et al. (1997). Métrica principal: MEDIA en ventana.
V_C240 = (220, 260)    # ms
# SECUNDARIA: positividad tardía / posible c320 de Zhang.
V_C320 = (290, 340)    # ms

# Métodos de promediado (Script 04). El homogéneo es el principal (Script 06).
METODO_PRINCIPAL = "homogeneo"
METODO_SECUNDARIO = "inhomogeneo"
METODOS = [METODO_PRINCIPAL, METODO_SECUNDARIO]

# Mapeo método -> (ventana, métrica) de la fila oficial en tabla_estadistica.csv.
# El Script 06 guarda el homogéneo como "c240 (220-260 ms)" y el inhomogéneo
# (análisis secundario) bajo la ventana "c240 inh.".
OFICIAL_VENTANA_C240 = {
    "homogeneo": "c240 (220-260 ms)",
    "inhomogeneo": "c240 inh.",
}

# Ventana AMPLIA para detectar el pico positivo real del Grand Average (solo
# visualización: marcar dónde pica el componente, que se mueve entre c240 y la
# positividad tardía según la condición). NO afecta la métrica oficial del c240,
# que sigue usando la ventana fija 220-260 ms de Zhang.
V_PICO = (150, 400)   # ms

# =============================================================================
# PALETA / ESTÉTICA  (modo oscuro, científico)
# =============================================================================
# Colores de grupo — los mismos que los scripts de matplotlib.
COLOR_CONTROL = "#2563eb"      # azul
COLOR_ALCOHOLIC = "#dc2626"    # rojo

# Sombreado de ventanas críticas.
COLOR_VENT_C240 = (245, 158, 11)    # naranja (c240, como en los scripts)
COLOR_VENT_C320 = (168, 85, 247)    # violeta (positividad tardía 290-340 ms)

# Paleta de la interfaz (slate / dark).
BG_APP = "#0f172a"        # fondo general
BG_PANEL = "#1e293b"      # sidebar / tarjetas
BG_PLOT = "#0b1220"       # fondo del gráfico
BORDER = "#334155"
TEXT = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
ACCENT = "#38bdf8"
GRID = "#1f2a3d"
GOOD = "#22c55e"
WARN = "#f59e0b"

FONT_FAMILY = "Segoe UI, Inter, Helvetica, Arial, sans-serif"

# Nombre legible de grupo para etiquetas.
NOMBRE_GRUPO = {"control": "Control", "alcoholic": "Alcohólico"}
COLOR_GRUPO = {"control": COLOR_CONTROL, "alcoholic": COLOR_ALCOHOLIC}
