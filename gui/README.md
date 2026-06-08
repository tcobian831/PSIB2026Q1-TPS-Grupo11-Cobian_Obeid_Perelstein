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
  grupo, diferencia, Cohen's d y p-Welch (una cola, C>A) **recalculados en vivo**
  sobre la cohorte del slider.
- **Referencia oficial (Script 06):** los mismos contrastes sobre la muestra
  completa 77 vs 45, con `p(FDR)`. En el slider a 77 ambos coinciden.
- **SNR (mediana hom / inh)** de `tabla_snr_comparacion.csv` (cohorte de
  referencia 45+45), con la nota honesta sobre la posible **circularidad** del
  inhomogéneo.
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
  config.py       constantes (canales, ventanas, SEED, paleta, rutas)
  data_loader.py  carga + validación de outputs/ (error claro si falta algo)
  averaging.py    igualar_grupos + Grand Average  (réplica Script 04 v2)
  stats.py        Cohen's d + Welch una cola + SNR (réplica Scripts 05/06)
  plotting.py     widget pyqtgraph (hover µV/ms, banda SEM, sombreados)
  widgets.py      sidebar + KPI cards
  app.py          ventana principal, tema oscuro, cableado
```
