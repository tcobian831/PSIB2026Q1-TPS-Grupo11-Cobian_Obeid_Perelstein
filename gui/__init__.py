"""
Interfaz gráfica de escritorio (PySide6 + pyqtgraph) para el análisis de
Potenciales Evocados Visuales (componente c240/VMP) en sujetos alcohólicos
vs control — dataset UCI EEG.

La app NO recalcula nada desde los trials crudos: consume los productos ya
generados por los Scripts 04 v2 / 05 / 06 (PE individuales, métricas del c240,
SNR y tabla estadística) y replica EXACTAMENTE su lógica de balanceo y Grand
Average a partir de los PE individuales.

Grupo 11: Cobián, Obeid, Perelstein.
"""

__version__ = "1.0.0"
