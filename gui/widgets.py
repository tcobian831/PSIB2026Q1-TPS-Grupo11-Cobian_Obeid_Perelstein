"""
widgets.py — Sidebar de controles y panel de KPIs.

Sidebar (panel lateral fijo):
  • Selector de canal y de condición.
  • Selector de método de promediado (homogéneo / inhomogéneo).
  • Casilla para superponer el método alternativo (hom. sólido vs inh. punteado).
  • Slider de balanceo de grupos (45–77 alcohólicos) + semilla configurable.
  • Nota explícita: el slider re-balancea el Grand Average a partir de los PE
    individuales ya calculados; NO re-promedia trials crudos.

Panel de KPIs (sobre el gráfico):
  • Métrica PRINCIPAL = media en ventana c240 (no el máximo).
  • media ± SD por grupo y diferencia control - alcohólico (en vivo).
  • SNR (homogéneo / inhomogéneo) con nota honesta sobre la circularidad.
  • Referencia oficial del Script 06 (muestra completa 77 vs 45) y secundarios.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal

from . import config
from .stats import KpiOficial, KpiSnr, KpiVivo


# =============================================================================
# Formateo
# =============================================================================
def fmt(v, dec=2, suf=""):
    try:
        if v is None or v != v:   # NaN
            return "—"
        return f"{v:.{dec}f}{suf}"
    except (TypeError, ValueError):
        return "—"


def fmt_signed(v, dec=2, suf=""):
    try:
        if v is None or v != v:
            return "—"
        return f"{v:+.{dec}f}{suf}"
    except (TypeError, ValueError):
        return "—"


def fmt_p(p):
    """p-valor legible: notación científica si es muy chico."""
    try:
        if p is None or p != p:
            return "—"
        if p < 1e-3:
            return f"{p:.2e}"
        return f"{p:.3f}"
    except (TypeError, ValueError):
        return "—"


# =============================================================================
# Sidebar
# =============================================================================
@dataclass
class SidebarState:
    canal: str
    condicion: str
    metodo: str
    superponer: bool
    n_alc: int
    seed: int


class Sidebar(QtWidgets.QFrame):
    """Panel lateral de controles. Emite configChanged ante cualquier cambio."""

    configChanged = Signal()

    def __init__(self, canales: list[str], condiciones: list[str],
                 n_alc_total: int, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(330)

        self._n_alc_max = max(n_alc_total, config.N_ALC_MIN)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(10)

        # --- título -----------------------------------------------------------
        titulo = QtWidgets.QLabel("Controles")
        titulo.setObjectName("sidebarTitle")
        lay.addWidget(titulo)
        sub = QtWidgets.QLabel("Potenciales evocados · c240 / VMP")
        sub.setObjectName("sidebarSub")
        lay.addWidget(sub)
        lay.addSpacing(6)

        # --- canal ------------------------------------------------------------
        lay.addWidget(self._seccion("Canal"))
        self.cmb_canal = QtWidgets.QComboBox()
        for c in canales:
            self.cmb_canal.addItem(f"{c}  ·  hemisferio {config.hemisferio_de(c)}", c)
        lay.addWidget(self.cmb_canal)

        # --- condición --------------------------------------------------------
        lay.addWidget(self._seccion("Condición"))
        self.cmb_cond = QtWidgets.QComboBox()
        for c in condiciones:
            self.cmb_cond.addItem(c, c)
        lay.addWidget(self.cmb_cond)

        # --- método -----------------------------------------------------------
        lay.addWidget(self._seccion("Método de promediado"))
        self.rb_hom = QtWidgets.QRadioButton("Homogéneo (sólido)")
        self.rb_inh = QtWidgets.QRadioButton("Inhomogéneo (punteado)")
        self.rb_hom.setChecked(True)
        self.grp_metodo = QtWidgets.QButtonGroup(self)
        self.grp_metodo.addButton(self.rb_hom)
        self.grp_metodo.addButton(self.rb_inh)
        lay.addWidget(self.rb_hom)
        lay.addWidget(self.rb_inh)
        self.chk_superponer = QtWidgets.QCheckBox("Superponer el método alternativo")
        lay.addWidget(self.chk_superponer)

        # --- balanceo de grupos ----------------------------------------------
        lay.addSpacing(4)
        lay.addWidget(self._seccion("Balanceo de grupos"))
        self.lbl_slider = QtWidgets.QLabel()
        self.lbl_slider.setObjectName("sliderValue")
        lay.addWidget(self.lbl_slider)

        self.sld = QtWidgets.QSlider(Qt.Horizontal)
        self.sld.setRange(config.N_ALC_MIN, self._n_alc_max)
        self.sld.setValue(config.N_ALC_MIN)
        self.sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sld.setTickInterval(8)
        lay.addWidget(self.sld)

        rango = QtWidgets.QLabel(
            f"{config.N_ALC_MIN} (igualado 45/45)  →  "
            f"{self._n_alc_max} (muestra completa)")
        rango.setObjectName("sidebarSub")
        lay.addWidget(rango)

        # semilla
        fila_seed = QtWidgets.QHBoxLayout()
        fila_seed.addWidget(QtWidgets.QLabel("Semilla (SEED)"))
        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(0, 999_999)        # cualquier entero válido
        self.spin_seed.setValue(config.SEED_DEFAULT)
        # Sin keyboardTracking: solo emite al confirmar (Enter / perder foco), así
        # no se dispara en cada tecla ni queda en estado intermedio "inválido"
        # (que el estilo de Qt marca con fondo rojo mientras se edita).
        self.spin_seed.setKeyboardTracking(False)
        self.spin_seed.setFixedWidth(110)
        fila_seed.addStretch(1)
        fila_seed.addWidget(self.spin_seed)
        lay.addLayout(fila_seed)

        # nota explícita sobre qué hace el slider
        self.nota_slider = QtWidgets.QLabel(
            "El slider re-balancea el <b>Grand Average</b> a partir de los PE "
            "individuales ya calculados (Script 04). <b>No</b> re-promedia "
            "trials crudos. Controles fijos en 45.")
        self.nota_slider.setObjectName("nota")
        self.nota_slider.setWordWrap(True)
        lay.addWidget(self.nota_slider)

        lay.addStretch(1)

        # pie
        pie = QtWidgets.QLabel("Grupo 11 · UCI EEG\nCobián · Obeid · Perelstein")
        pie.setObjectName("sidebarFoot")
        lay.addWidget(pie)

        # --- conexiones -------------------------------------------------------
        self.cmb_canal.currentIndexChanged.connect(self._emit)
        self.cmb_cond.currentIndexChanged.connect(self._emit)
        self.rb_hom.toggled.connect(self._emit)
        self.chk_superponer.toggled.connect(self._emit)
        self.sld.valueChanged.connect(self._on_slider)
        self.spin_seed.valueChanged.connect(self._emit)

        self._actualizar_label_slider(self.sld.value())

    # ------------------------------------------------------------------ helpers
    def _seccion(self, texto: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(texto.upper())
        lbl.setObjectName("seccion")
        return lbl

    def _actualizar_label_slider(self, n: int):
        self.lbl_slider.setText(
            f"Alcohólicos incluidos: <b>{n}</b>  +  45 controles")

    def _on_slider(self, n: int):
        self._actualizar_label_slider(n)
        self._emit()

    def _emit(self, *_):
        self.configChanged.emit()

    # -------------------------------------------------------------------- estado
    def estado(self) -> SidebarState:
        return SidebarState(
            canal=self.cmb_canal.currentData(),
            condicion=self.cmb_cond.currentData(),
            metodo=("homogeneo" if self.rb_hom.isChecked() else "inhomogeneo"),
            superponer=self.chk_superponer.isChecked(),
            n_alc=self.sld.value(),
            seed=self.spin_seed.value(),
        )


# =============================================================================
# KPI cards
# =============================================================================
class KPICard(QtWidgets.QFrame):
    """Tarjeta compacta: título, valor grande y subtítulo."""

    def __init__(self, titulo: str, parent=None):
        super().__init__(parent)
        self.setObjectName("kpiCard")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        self.lbl_titulo = QtWidgets.QLabel(titulo)
        self.lbl_titulo.setObjectName("kpiTitle")
        self.lbl_valor = QtWidgets.QLabel("—")
        self.lbl_valor.setObjectName("kpiValue")
        self.lbl_sub = QtWidgets.QLabel("")
        self.lbl_sub.setObjectName("kpiSub")
        self.lbl_sub.setWordWrap(True)

        lay.addWidget(self.lbl_titulo)
        lay.addWidget(self.lbl_valor)
        lay.addWidget(self.lbl_sub)

    def set(self, valor: str, sub: str = "", color: str | None = None):
        self.lbl_valor.setText(valor)
        self.lbl_sub.setText(sub)
        if color:
            self.lbl_valor.setStyleSheet(f"color: {color};")


class StatsPanel(QtWidgets.QWidget):
    """Fila de KPI cards + tira de notas/referencia/secundarios."""

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # --- fila de tarjetas -------------------------------------------------
        fila = QtWidgets.QHBoxLayout()
        fila.setSpacing(10)
        self.card_ctrl = KPICard("Media c240 · Control")
        self.card_alc = KPICard("Media c240 · Alcohólico")
        self.card_dif = KPICard("Diferencia (C − A)")
        self.card_snr = KPICard("SNR · señal/ruido")
        for c in (self.card_ctrl, self.card_alc, self.card_dif,
                  self.card_snr):
            fila.addWidget(c)
        root.addLayout(fila)

        # --- referencia oficial + secundarios --------------------------------
        self.lbl_oficial = QtWidgets.QLabel()
        self.lbl_oficial.setObjectName("refOficial")
        self.lbl_oficial.setWordWrap(True)
        root.addWidget(self.lbl_oficial)

        self.lbl_secundarios = QtWidgets.QLabel()
        self.lbl_secundarios.setObjectName("secundarios")
        self.lbl_secundarios.setWordWrap(True)
        root.addWidget(self.lbl_secundarios)

        self.lbl_nota_snr = QtWidgets.QLabel()
        self.lbl_nota_snr.setObjectName("nota")
        self.lbl_nota_snr.setWordWrap(True)
        root.addWidget(self.lbl_nota_snr)

    # ----------------------------------------------------------------- update
    def actualizar(self, vivo: KpiVivo, oficial: KpiOficial, snr: KpiSnr,
                   picos: dict | None = None):
        # --- tarjeta métrica principal ---------------------------------------
        self.card_ctrl.set(
            f"{fmt_signed(vivo.control_media)} µV",
            f"± {fmt(vivo.control_sd)} · n={vivo.n_control}",
            color=config.COLOR_CONTROL,
        )
        self.card_alc.set(
            f"{fmt_signed(vivo.alcoholic_media)} µV",
            f"± {fmt(vivo.alcoholic_sd)} · n={vivo.n_alcoholic}",
            color=config.COLOR_ALCOHOLIC,
        )
        self.card_dif.set(
            f"{fmt_signed(vivo.diferencia)} µV",
            "media en ventana 220–260 ms",
        )

        # SNR (relación señal/ruido) — contenido visto en clase.
        if snr.disponible:
            self.card_snr.set(
                f"Hom {fmt(snr.snr_homogeneo, 2)}  ·  Inh {fmt(snr.snr_inhomogeneo, 2)}",
                "homogéneo vs inhomogéneo · más alto = mejor",
            )
        else:
            self.card_snr.set("—", "sin datos de SNR")

        # --- referencia: muestra completa (todos los sujetos) ----------------
        if oficial.disponible:
            self.lbl_oficial.setText(
                f"<b>Muestra completa (todos los sujetos: "
                f"{oficial.n_alcoholic} alcohólicos vs {oficial.n_control} "
                f"controles):</b> "
                f"Control {fmt_signed(oficial.control_media)} ± "
                f"{fmt(oficial.control_sd)} µV · "
                f"Alcohólico {fmt_signed(oficial.alcoholic_media)} ± "
                f"{fmt(oficial.alcoholic_sd)} µV · "
                f"diferencia {fmt_signed(oficial.diferencia)} µV"
            )
        else:
            self.lbl_oficial.setText(
                "<b>Muestra completa:</b> sin datos para esta combinación.")

        # --- pico real del promedio (ventana amplia) -------------------------
        pico_txt = ""
        if picos:
            def _pk(g):
                p = picos.get(g)
                return (f"{fmt_signed(p[1])} µV @ {fmt(p[0], 0)} ms"
                        if p else "—")
            pico_txt = (
                "<b>Pico del promedio (ventana 200–400 ms):</b> "
                f"Control {_pk('control')} · Alcohólico {_pk('alcoholic')}<br>"
            )

        # --- secundarios (claramente etiquetados) ----------------------------
        self.lbl_secundarios.setText(
            pico_txt +
            "<b>Secundarios (no principales):</b> "
            f"máx en c240 C {fmt_signed(vivo.control_max)} / A {fmt_signed(vivo.alcoholic_max)} µV · "
            f"latencia en c240 C {fmt(vivo.control_lat, 0)} / A {fmt(vivo.alcoholic_lat, 0)} ms · "
            f"AUC c240 C {fmt_signed(vivo.control_auc, 1)} / A {fmt_signed(vivo.alcoholic_auc, 1)} µV·ms · "
            f"media c320 C {fmt_signed(vivo.control_media_c320)} / "
            f"A {fmt_signed(vivo.alcoholic_media_c320)} µV"
        )

        # --- nota honesta sobre SNR ------------------------------------------
        self.lbl_nota_snr.setText(
            "Nota SNR: la mayor SNR del inhomogéneo puede ser parcialmente "
            "circular (pondera cada trial por su parecido al promedio). Se "
            "presentan ambos métodos; no se concluye que «inhomogéneo = mejor»."
        )
