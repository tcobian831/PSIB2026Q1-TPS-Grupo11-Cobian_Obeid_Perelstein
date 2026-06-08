"""
app.py — Ventana principal de la GUI de análisis ERP.

Ejecutar:
    python gui/app.py
    (o, desde la raíz del proyecto)  python -m gui.app

Layout: sidebar de controles fijo a la izquierda + área central con el panel de
KPIs arriba y el gráfico comparativo grande debajo. Modo oscuro, fuente
sans-serif.

Flujo ante cualquier cambio de control:
    1. Re-seleccionar la cohorte (n alcohólicos del slider + 45 controles, SEED).
    2. Re-calcular el Grand Average desde los PE individuales (Script 04).
    3. Re-calcular los KPIs en vivo (Scripts 05/06) y leer la referencia oficial.
    4. Redibujar gráfico y tarjetas.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# El entorno tiene PyQt6 y PySide6 instalados. Si pyqtgraph se importa antes que
# PySide6, elige PyQt6 por defecto y carga SUS DLLs de Qt6; un import posterior
# de PySide6 choca (dos bindings Qt en el mismo proceso -> "DLL load failed").
# Forzamos el binding a PySide6 antes de cualquier import de pyqtgraph.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

# Permite ejecutar tanto `python gui/app.py` como `python -m gui.app`.
# Al correr como script, Python pone la carpeta gui/ en sys.path[0]. La
# reemplazamos por la raíz del proyecto para (a) habilitar los imports del
# paquete 'gui' y (b) evitar que los módulos de gui/ queden en el espacio
# top-level y shadowen dependencias que pyqtgraph/PySide6 cargan de forma
# diferida — eso rompe la resolución de DLLs de Qt en Windows.
if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parent.parent)
    __package__ = "gui"

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import averaging, config, stats
from .data_loader import AppData, MissingDataError, cargar_datos
from .plotting import ERPPlotWidget
from .widgets import Sidebar, StatsPanel


# =============================================================================
# Hoja de estilos (modo oscuro)
# =============================================================================
def construir_qss() -> str:
    c = config
    return f"""
    QWidget {{
        background-color: {c.BG_APP};
        color: {c.TEXT};
        font-family: {c.FONT_FAMILY};
        font-size: 13px;
    }}
    QFrame#sidebar {{
        background-color: {c.BG_PANEL};
        border-right: 1px solid {c.BORDER};
    }}
    QLabel#sidebarTitle {{ font-size: 20px; font-weight: 700; }}
    QLabel#sidebarSub   {{ color: {c.TEXT_MUTED}; font-size: 11px; }}
    QLabel#sidebarFoot  {{ color: {c.TEXT_MUTED}; font-size: 10px; }}
    QLabel#seccion {{
        color: {c.ACCENT}; font-size: 10px; font-weight: 700;
        letter-spacing: 1px; margin-top: 4px;
    }}
    QLabel#sliderValue {{ font-size: 13px; }}
    QLabel#nota {{
        color: {c.TEXT_MUTED}; font-size: 11px;
        background: rgba(56,189,248,0.07);
        border: 1px solid {c.BORDER}; border-radius: 6px; padding: 7px;
    }}
    QLabel#appTitle    {{ font-size: 18px; font-weight: 700; }}
    QLabel#appSubtitle {{ color: {c.TEXT_MUTED}; font-size: 12px; }}

    QComboBox, QSpinBox {{
        background: {c.BG_APP}; color: {c.TEXT};
        border: 1px solid {c.BORDER};
        border-radius: 6px; padding: 6px 8px;
        selection-background-color: {c.ACCENT}; selection-color: #04121f;
    }}
    QComboBox:hover, QSpinBox:hover, QSpinBox:focus {{ border-color: {c.ACCENT}; }}
    QSpinBox::up-button, QSpinBox::down-button {{
        background: {c.BG_PANEL}; border: none; width: 16px;
    }}
    QComboBox QAbstractItemView {{
        background: {c.BG_PANEL}; border: 1px solid {c.BORDER};
        selection-background-color: {c.ACCENT}; selection-color: #04121f;
        outline: none;
    }}
    QRadioButton, QCheckBox {{ padding: 3px 0; }}
    QRadioButton::indicator, QCheckBox::indicator {{ width: 15px; height: 15px; }}

    QSlider::groove:horizontal {{
        height: 5px; background: {c.BORDER}; border-radius: 3px;
    }}
    QSlider::sub-page:horizontal {{
        background: {c.ACCENT}; border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {c.TEXT}; width: 16px; margin: -7px 0;
        border-radius: 8px;
    }}

    QFrame#kpiCard {{
        background: {c.BG_PANEL}; border: 1px solid {c.BORDER};
        border-radius: 10px;
    }}
    QLabel#kpiTitle {{ color: {c.TEXT_MUTED}; font-size: 11px; }}
    QLabel#kpiValue {{ font-size: 22px; font-weight: 700; }}
    QLabel#kpiSub   {{ color: {c.TEXT_MUTED}; font-size: 10px; }}

    QLabel#refOficial  {{
        color: {c.TEXT}; font-size: 12px;
        background: {c.BG_PANEL}; border: 1px solid {c.BORDER};
        border-radius: 8px; padding: 8px 10px;
    }}
    QLabel#secundarios {{ color: {c.TEXT_MUTED}; font-size: 11px; padding: 0 2px; }}
    QToolTip {{
        background: {c.BG_PANEL}; color: {c.TEXT};
        border: 1px solid {c.BORDER};
    }}
    """


# =============================================================================
# Ventana principal
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data: AppData):
        super().__init__()
        self.data = data
        self.setWindowTitle("Análisis ERP · c240 / VMP — Alcohólicos vs Control")
        self.resize(1380, 880)

        # Clave de vista para decidir cuándo reajustar el zoom.
        self._ultima_vista = None

        # --- widgets ----------------------------------------------------------
        self.sidebar = Sidebar(
            canales=data.canales_disponibles(),
            condiciones=data.condiciones_disponibles(),
            n_alc_total=data.n_alcoholicos_total(),
        )
        self.stats_panel = StatsPanel()
        self.plot = ERPPlotWidget()

        # --- header del área central -----------------------------------------
        header = QtWidgets.QWidget()
        hlay = QtWidgets.QVBoxLayout(header)
        hlay.setContentsMargins(2, 0, 2, 0)
        hlay.setSpacing(1)
        t = QtWidgets.QLabel("Potenciales Evocados Visuales — componente c240 / VMP")
        t.setObjectName("appTitle")
        s = QtWidgets.QLabel(
            "Control (azul) vs Alcohólico (rojo) · Grand Average ± SEM entre "
            "sujetos · ventanas sombreadas: c240 (220–260 ms) y positividad "
            "tardía (290–340 ms)")
        s.setObjectName("appSubtitle")
        s.setWordWrap(True)
        hlay.addWidget(t)
        hlay.addWidget(s)

        # --- área central -----------------------------------------------------
        central = QtWidgets.QWidget()
        clay = QtWidgets.QVBoxLayout(central)
        clay.setContentsMargins(18, 16, 18, 16)
        clay.setSpacing(12)
        clay.addWidget(header)
        clay.addWidget(self.stats_panel)
        clay.addWidget(self.plot, stretch=1)

        # --- ensamblado -------------------------------------------------------
        cont = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(cont)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.sidebar)
        lay.addWidget(central, stretch=1)
        self.setCentralWidget(cont)

        # --- señales ----------------------------------------------------------
        self.sidebar.configChanged.connect(self.refrescar)

        # primer render
        self.refrescar()

    # -------------------------------------------------------------------------
    def refrescar(self):
        st = self.sidebar.estado()
        pe_sel = self.data.pe_metodo(st.metodo)

        # 1. Cohorte: n alcohólicos (slider, SEED) + todos los controles.
        incluidos = averaging.sujetos_incluidos(pe_sel, st.n_alc, st.seed)

        # 2. Grand Average del método seleccionado.
        curvas_sel = averaging.calcular_grand_average(
            pe_sel, incluidos, st.canal, st.condicion)

        # 2b. Otro método superpuesto (misma cohorte).
        curvas_otro = None
        metodo_otro = None
        if st.superponer:
            metodo_otro = ("inhomogeneo" if st.metodo == "homogeneo"
                           else "homogeneo")
            curvas_otro = averaging.calcular_grand_average(
                self.data.pe_metodo(metodo_otro), incluidos,
                st.canal, st.condicion)

        # 3. ¿Reajustar zoom? Solo si cambió canal/condición/método.
        vista = (st.canal, st.condicion, st.metodo)
        reset_view = vista != self._ultima_vista
        self._ultima_vista = vista

        # 4. Gráfico.
        self.plot.actualizar(curvas_sel, st.metodo, curvas_otro, metodo_otro,
                             reset_view=reset_view)

        # 5. KPIs.
        vivo = stats.kpi_en_vivo(
            self.data.c240, st.metodo, st.canal, st.condicion, incluidos)
        oficial = stats.kpi_oficial(
            self.data.estadistica, st.metodo, st.canal, st.condicion)
        snr = stats.kpi_snr(self.data.snr, st.canal, st.condicion)
        self.stats_panel.actualizar(vivo, oficial, snr)


# =============================================================================
# main
# =============================================================================
def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setStyleSheet(construir_qss())

    # Carga + validación de datos. Error claro si falta algún archivo.
    try:
        data = cargar_datos()
    except MissingDataError as e:
        box = QtWidgets.QMessageBox()
        box.setIcon(QtWidgets.QMessageBox.Critical)
        box.setWindowTitle("Faltan datos en outputs/")
        box.setText("No se puede iniciar la aplicación.")
        box.setDetailedText(str(e))
        box.setStandardButtons(QtWidgets.QMessageBox.Close)
        box.exec()
        return 1

    win = MainWindow(data)
    win.show()

    # Modo de auto-prueba: construye, muestra y cierra solo (para verificación).
    if "--selftest" in sys.argv:
        QtCore.QTimer.singleShot(800, app.quit)
        print("[selftest] ventana construida y mostrada correctamente; cerrando.")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
