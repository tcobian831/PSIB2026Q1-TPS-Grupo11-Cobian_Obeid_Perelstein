# GUI — Análisis de Potenciales Evocados Visuales (c240 / VMP)

Interfaz de escritorio (PySide6 + pyqtgraph) para explorar el componente
**c240/VMP (220–260 ms)** en sujetos **alcohólicos vs control** (dataset UCI EEG).

La app **no recalcula nada desde los trials crudos**: consume los productos ya
generados por los Scripts 04 v2 / 05 / 06 y replica su lógica de balanceo y
Grand Average a partir de los **PE individuales** ya guardados.

## Requisitos

```bash
python -m pip install -r requirements.txt
```

Necesita estos archivos en `outputs/` (si falta alguno, la app lo avisa e
indica qué script correr):

| Archivo | Generado por |
|---|---|
| `eeg_PE_homogeneo.parquet` / `eeg_PE_inhomogeneo.parquet` | `scripts/04_promediado_v2.py` |
| `eeg_c240_extraido.csv` | `scripts/05_extraccion_pico.py` |
| `tabla_snr_comparacion.csv` | `scripts/04_promediado_v2.py` |
| `tabla_estadistica.csv` | `scripts/06_estadistica.py` |

## Ejecutar

```bash
python gui/app.py
# o, desde la raíz:
python -m gui.app
```

## Qué hace cada control

- **Canal / Condición:** los 8 canales (4 derechos + 4 homólogos izquierdos) y
  las 2 condiciones (`S1 obj`, `S2 nomatch`).
- **Método de promediado:** homogéneo (línea sólida) o inhomogéneo (punteada).
  Casilla para **superponer** ambos.
- **Slider de balanceo (45–77):** re-muestrea aleatoriamente alcohólicos
  (semilla configurable, default 42) y **re-balancea el Grand Average** desde los
  PE individuales. **No** re-promedia trials crudos. Los controles quedan fijos
  en 45. En 45 reproduce el GA igualado del Script 04; en 77, la muestra completa.

## Panel de KPIs

- **Métrica principal = MEDIA en ventana c240** (no el máximo): media ± SD por
  grupo y diferencia **recalculados en vivo**
  sobre la cohorte del slider.
- **SNR (mediana hom / inh)** de `tabla_snr_comparacion.csv` (cohorte de
  referencia 45+45).
- **Secundarios** (máximo, latencia, AUC, media c320) claramente etiquetados.

## Solución de problemas

- **`ImportError: DLL load failed while importing QtGui/QtCore`**: ocurre cuando
  en el entorno conviven **PyQt6 y PySide6** y pyqtgraph elige PyQt6, chocando con
  PySide6 (dos copias de Qt6 en el mismo proceso). La app ya lo evita fijando
  `PYQTGRAPH_QT_LIB=PySide6` antes de importar pyqtgraph (en `app.py` y
  `plotting.py`). Si lo ves al integrar este código en otro script, exportá esa
  variable o importá PySide6 antes que pyqtgraph.



## Módulos
```
gui/
  config.py       constantes de configuración: canales, ventanas, colores, rutas y SEED
  data_loader.py  carga y validación de archivos desde outputs/
  averaging.py    cálculo de Grand Average desde los PE individuales ya generados
  stats.py        métricas descriptivas: media c240, SD, diferencia entre grupos y SNR
  plotting.py     gráfico interactivo de Grand Average, ventanas c240/c320, cursor y picos
  widgets.py      controles laterales y tarjetas de resultados
  app.py          ventana principal e integración de la interfaz
```
