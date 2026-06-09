"""
plotting.py — Widget de gráfico ERP basado en pyqtgraph.

Se eligió pyqtgraph sobre matplotlib porque necesitamos hover en vivo (µV/ms con
crosshair) y zoom/pan nativo fluido; matplotlib embebido en Qt redibuja toda la
figura en cada movimiento del mouse.

Dibuja, para un canal/condición:
  • Control (azul) vs Alcohólico (rojo), Grand Average.
  • Banda de SEM semitransparente (entre sujetos) del método seleccionado.
  • Convención de estilo: homogéneo = línea sólida, inhomogéneo = punteada.
  • Sombreado de ventanas críticas: c240 (220-260 ms) y positividad tardía
    (290-340 ms).
  • Hover comparativo: crosshair vertical + tooltip fijo (arriba a la derecha)
    con el tiempo (ms) y el valor interpolado de Control (azul) y Alcohólico
    (rojo) en ese instante, más la diferencia (C−A).
"""

from __future__ import annotations

import os

# Forzar el binding Qt de pyqtgraph a PySide6 (el entorno también tiene PyQt6 y,
# si no se fija, pyqtgraph lo elegiría, provocando un conflicto de DLLs de Qt6).
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import config
from .averaging import CurvaGA


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore


def _pen_metodo(color_hex: str, metodo: str, width: float = 2.2, alpha: int = 255):
    """Pen con la convención de estilo por método (homogéneo sólido / inh. punteado)."""
    r, g, b = _hex_rgb(color_hex)
    estilo = QtCore.Qt.SolidLine if metodo == "homogeneo" else QtCore.Qt.DashLine
    return pg.mkPen(color=(r, g, b, alpha), width=width, style=estilo)


def pico_positivo(t, y, lo: float, hi: float):
    """
    Latencia (ms) y amplitud (µV) del pico positivo de una curva dentro de la
    ventana [lo, hi]. Solo para marcar visualmente dónde pica el componente
    (NO interviene en la métrica del c240). Devuelve (lat, amp) o None.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    m = (t >= lo) & (t <= hi)
    if not m.any():
        return None
    seg_t, seg_y = t[m], y[m]
    i = int(np.argmax(seg_y))
    return float(seg_t[i]), float(seg_y[i])


class ERPPlotWidget(pg.PlotWidget):
    """Gráfico comparativo Control vs Alcohólico con hover y sombreado."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setBackground(config.BG_PLOT)
        self.showGrid(x=True, y=True, alpha=0.18)
        self.setMenuEnabled(True)
        self.setMouseEnabled(x=True, y=True)

        pi = self.getPlotItem()
        pi.setLabel("bottom", "Tiempo", units="ms")
        pi.setLabel("left", "Amplitud", units="µV")
        pi.getAxis("bottom").setTextPen(config.TEXT_MUTED)
        pi.getAxis("left").setTextPen(config.TEXT_MUTED)
        pi.getAxis("bottom").setPen(config.BORDER)
        pi.getAxis("left").setPen(config.BORDER)

        # Leyenda en una esquina libre (abajo a la derecha) para no pisarse con
        # las etiquetas de ventana (arriba) ni con el tooltip de hover.
        self.legend = pi.addLegend(offset=(-12, -12), labelTextColor=config.TEXT)

        # --- ejes de referencia: estímulo (t=0) y línea de 0 µV ---------------
        self.addItem(pg.InfiniteLine(pos=0, angle=90,
                     pen=pg.mkPen(config.TEXT_MUTED, width=0.8,
                                  style=QtCore.Qt.DotLine)))
        self.addItem(pg.InfiniteLine(pos=0, angle=0,
                     pen=pg.mkPen(config.BORDER, width=0.8)))

        # --- sombreado de ventanas críticas -----------------------------------
        self._region_c240 = self._crear_region(config.V_C240, config.COLOR_VENT_C240)
        self._region_c320 = self._crear_region(config.V_C320, config.COLOR_VENT_C320)
        self.addItem(self._region_c240)
        self.addItem(self._region_c320)
        self._etq_c240 = self._crear_etiqueta_region("c240 / VMP\n220–260 ms",
                                                     np.mean(config.V_C240),
                                                     config.COLOR_VENT_C240)
        self._etq_c320 = self._crear_etiqueta_region("c320 (Zhang)\n290–340 ms",
                                                     np.mean(config.V_C320),
                                                     config.COLOR_VENT_C320)
        self.addItem(self._etq_c240)
        self.addItem(self._etq_c320)

        # --- crosshair vertical (sigue el cursor) -----------------------------
        self._vline = pg.InfiniteLine(angle=90, movable=False,
                                      pen=pg.mkPen(config.ACCENT, width=1.2,
                                                   style=QtCore.Qt.DashLine))
        self._vline.setZValue(50)
        self._vline.setVisible(False)
        self.addItem(self._vline, ignoreBounds=True)

        # --- tooltip de hover: QLabel fijo arriba a la derecha ----------------
        # Es un widget hijo del viewport (no un item del gráfico): nunca queda
        # tapado por las curvas ni se recorta, y se mantiene en una zona fija.
        self._tooltip = QtWidgets.QLabel(self.viewport())
        self._tooltip.setObjectName("hoverTooltip")
        self._tooltip.setTextFormat(QtCore.Qt.RichText)
        self._tooltip.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self._tooltip.setStyleSheet(
            f"#hoverTooltip {{"
            f" background: rgba(15,23,42,235);"
            f" border: 1px solid {config.BORDER};"
            f" border-radius: 8px; padding: 8px 11px; color: {config.TEXT};"
            f" font-size: 12px; }}"
        )
        self._tooltip.setVisible(False)

        # Estado dinámico.
        self._items_dinamicos: list = []
        self._series: list[dict] = []   # [{grupo, metodo, color, t, y}]
        self._metodo_sel: str | None = None
        self._metodo_otro: str | None = None

        # Señales de mouse y de rango (para reposicionar etiquetas de ventana).
        self._proxy = pg.SignalProxy(self.scene().sigMouseMoved,
                                     rateLimit=60, slot=self._on_mouse_moved)
        self.getViewBox().sigYRangeChanged.connect(self._reposicionar_etiquetas)

    # -------------------------------------------------------------------------
    # Construcción de elementos estáticos
    # -------------------------------------------------------------------------
    def _crear_region(self, ventana, color_rgb):
        r, g, b = color_rgb
        region = pg.LinearRegionItem(
            values=ventana, movable=False,
            brush=pg.mkBrush(r, g, b, 32),
            pen=pg.mkPen(r, g, b, 80),
        )
        region.setZValue(-20)
        return region

    def _crear_etiqueta_region(self, texto, x_centro, color_rgb):
        r, g, b = color_rgb
        etq = pg.TextItem(text=texto, color=(r, g, b), anchor=(0.5, 0))
        etq.setPos(x_centro, 0)
        etq.setZValue(-10)
        return etq

    def _reposicionar_etiquetas(self):
        """Etiquetas de ventana arriba, escalonadas para que no se pisen.

        Las ventanas c240 (220-260) y c320 (290-340) están muy cerca; si las
        etiquetas se ponen a la misma altura, los textos quedan apretados.
        Solución: c240 pegada al borde superior, c320 un escalón por debajo.
        """
        try:
            (_, _), (y0, y1) = self.getViewBox().viewRange()
        except Exception:
            return
        rng = y1 - y0
        y_c240 = y1 - 0.02 * rng    # arriba del todo
        y_c320 = y1 - 0.12 * rng    # un escalón más abajo
        self._etq_c240.setPos(float(np.mean(config.V_C240)), y_c240)
        self._etq_c320.setPos(float(np.mean(config.V_C320)), y_c320)

    # -------------------------------------------------------------------------
    # API pública
    # -------------------------------------------------------------------------
    def actualizar(
        self,
        curvas_sel: dict[str, CurvaGA],
        metodo_sel: str,
        curvas_otro: dict[str, CurvaGA] | None,
        metodo_otro: str | None,
        reset_view: bool,
        picos: dict[str, tuple[float, float]] | None = None,
    ):
        """
        Redibuja el gráfico.

        curvas_sel  : Grand Average del método seleccionado (con banda SEM).
        curvas_otro : Grand Average del otro método, superpuesto sin banda
                      (None si no se quiere superponer).
        reset_view  : True para reajustar el zoom (cambió canal/condición/método).
        picos       : {grupo: (lat_ms, amp_uV)} para marcar el pico real del
                      promedio del método seleccionado.
        """
        # Limpiar dinámicos previos.
        for it in self._items_dinamicos:
            self.removeItem(it)
        self._items_dinamicos.clear()
        self.legend.clear()
        self._series.clear()
        self._metodo_sel = metodo_sel
        self._metodo_otro = metodo_otro if curvas_otro else None

        # --- método seleccionado: banda SEM + línea ---------------------------
        for grupo in config.GRUPOS:
            curva = curvas_sel.get(grupo)
            if curva is None:
                continue
            color = config.COLOR_GRUPO[grupo]
            self._dibujar_banda_sem(curva, color)
            nombre = (f"{config.NOMBRE_GRUPO[grupo]} · {metodo_sel} "
                      f"(n={curva.n_sujetos})")
            self._dibujar_linea(curva, grupo, color, metodo_sel, nombre,
                                width=2.4, alpha=255)

        # --- otro método superpuesto (sin banda, atenuado) --------------------
        if curvas_otro and metodo_otro:
            for grupo in config.GRUPOS:
                curva = curvas_otro.get(grupo)
                if curva is None:
                    continue
                color = config.COLOR_GRUPO[grupo]
                nombre = f"{config.NOMBRE_GRUPO[grupo]} · {metodo_otro}"
                self._dibujar_linea(curva, grupo, color, metodo_otro, nombre,
                                    width=1.6, alpha=150)

        # --- marcador del pico real del promedio (método seleccionado) --------
        if picos:
            for grupo, par in picos.items():
                if par is None:
                    continue
                lat, amp = par
                self._marcar_pico(lat, amp, config.COLOR_GRUPO[grupo])

        if reset_view:
            self.enableAutoRange()
            self.autoRange()
        self._reposicionar_etiquetas()

    # -------------------------------------------------------------------------
    # Helpers de dibujo
    # -------------------------------------------------------------------------
    def _dibujar_banda_sem(self, curva: CurvaGA, color_hex: str):
        r, g, b = _hex_rgb(color_hex)
        t = curva.tiempo_ms
        sup = pg.PlotDataItem(t, curva.media + curva.sem, pen=None)
        inf = pg.PlotDataItem(t, curva.media - curva.sem, pen=None)
        fill = pg.FillBetweenItem(sup, inf, brush=pg.mkBrush(r, g, b, 55))
        fill.setZValue(-5)
        for it in (sup, inf, fill):
            self.addItem(it)
            self._items_dinamicos.append(it)

    def _dibujar_linea(self, curva: CurvaGA, grupo: str, color_hex: str,
                       metodo: str, nombre: str, width: float, alpha: int):
        pen = _pen_metodo(color_hex, metodo, width=width, alpha=alpha)
        item = pg.PlotDataItem(curva.tiempo_ms, curva.media, pen=pen, name=nombre)
        item.setZValue(10)
        self.addItem(item)
        self._items_dinamicos.append(item)
        self._series.append({
            "grupo": grupo,
            "metodo": metodo,
            "color": color_hex,
            "t": curva.tiempo_ms,
            "y": curva.media,
        })

    def _marcar_pico(self, lat: float, amp: float, color_hex: str):
        """Punto en el pico positivo del promedio + etiqueta con la latencia."""
        r, g, b = _hex_rgb(color_hex)
        punto = pg.ScatterPlotItem(
            [lat], [amp], symbol="o", size=11,
            brush=pg.mkBrush(r, g, b, 255), pen=pg.mkPen("w", width=1.2),
        )
        punto.setZValue(30)
        self.addItem(punto)
        self._items_dinamicos.append(punto)

        etq = pg.TextItem(f"{lat:.0f} ms", color=(r, g, b), anchor=(0.5, 1.4))
        etq.setPos(lat, amp)
        etq.setZValue(30)
        self.addItem(etq)
        self._items_dinamicos.append(etq)

    # -------------------------------------------------------------------------
    # Hover comparativo
    # -------------------------------------------------------------------------
    def valor_interpolado(self, x: float, grupo: str, metodo: str):
        """Valor (µV) interpolado linealmente de una curva en el tiempo x (ms)."""
        for s in self._series:
            if s["grupo"] == grupo and s["metodo"] == metodo:
                return float(np.interp(x, s["t"], s["y"]))
        return None

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        if not self._series or not self.sceneBoundingRect().contains(pos):
            self._vline.setVisible(False)
            self._tooltip.setVisible(False)
            return

        vb = self.getViewBox()
        x = float(vb.mapSceneToView(pos).x())
        t = self._series[0]["t"]
        xc = float(min(max(x, t[0]), t[-1]))   # acotar al rango de datos

        self._tooltip.setText(self._html_tooltip(xc))
        self._posicionar_tooltip()
        self._tooltip.setVisible(True)
        self._tooltip.raise_()

        self._vline.setPos(xc)
        self._vline.setVisible(True)

    def _html_tooltip(self, xc: float) -> str:
        """Tabla HTML: t + Control (azul) y Alcohólico (rojo) + Δ(C−A) en xc."""
        cc, ca, mut = config.COLOR_CONTROL, config.COLOR_ALCOHOLIC, config.TEXT_MUTED
        ms = self._metodo_sel or config.METODO_PRINCIPAL

        def cell(v):
            return f"{v:+.2f} µV" if v is not None else "—"

        c = self.valor_interpolado(xc, "control", ms)
        a = self.valor_interpolado(xc, "alcoholic", ms)
        filas = [
            f"<tr><td style='color:{cc}'>● Control</td>"
            f"<td align='right'>&nbsp;<b>{cell(c)}</b></td></tr>",
            f"<tr><td style='color:{ca}'>● Alcohólico</td>"
            f"<td align='right'>&nbsp;<b>{cell(a)}</b></td></tr>",
        ]
        if c is not None and a is not None:
            filas.append(
                f"<tr><td style='color:{mut}'>Δ (C−A)</td>"
                f"<td align='right'>&nbsp;{(c - a):+.2f} µV</td></tr>")

        html = (
            f"<b>t = {xc:.1f} ms</b>"
            f"&nbsp;<span style='color:{mut}'>· {ms}</span>"
            f"<table cellspacing='2' style='margin-top:3px'>{''.join(filas)}</table>"
        )

        # Si hay método alternativo superpuesto, sus dos valores aparte.
        if self._metodo_otro:
            mo = self._metodo_otro
            c2 = self.valor_interpolado(xc, "control", mo)
            a2 = self.valor_interpolado(xc, "alcoholic", mo)
            html += (
                f"<div style='color:{mut};margin-top:5px'>método {mo}:</div>"
                f"<table cellspacing='2'>"
                f"<tr><td style='color:{cc}'>● Control</td>"
                f"<td align='right'>&nbsp;{cell(c2)}</td></tr>"
                f"<tr><td style='color:{ca}'>● Alcohólico</td>"
                f"<td align='right'>&nbsp;{cell(a2)}</td></tr></table>"
            )
        return html

    def _posicionar_tooltip(self):
        """Fija el tooltip en la esquina superior derecha del área de gráfico."""
        # pyqtgraph llama a resizeEvent durante super().__init__(), antes de que
        # exista el tooltip: salir si todavía no se creó.
        tt = getattr(self, "_tooltip", None)
        if tt is None:
            return
        tt.adjustSize()
        m = 14
        x = self.viewport().width() - tt.width() - m
        tt.move(max(m, x), m)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._posicionar_tooltip()
