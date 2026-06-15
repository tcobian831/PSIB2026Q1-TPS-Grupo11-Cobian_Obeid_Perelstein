# PSIB2026Q1-TPS-Grupo11-Cobian_Obeid_Perelstein

Trabajo práctico de la materia **Procesamiento de Señales e Imágenes Biomédicas** (ITBA). Analiza potenciales evocados visuales en señales EEG del dataset **UCI EEG Database** (https://archive.ics.uci.edu/dataset/121/eeg+database), con foco en el componente **c240/VMP** asociado a memoria visual, comparando sujetos **controles** y sujetos con **alcoholismo**.

El análisis se centra en dos condiciones del paradigma de reconocimiento visual:

```text
S1 obj       estímulo sample / codificación visual inicial
S2 nomatch   estímulo test / comparación con memoria visual
```

La ventana principal es el componente **c240/VMP entre 220 y 260 ms**. La ventana **290–340 ms** (componente tardío c320 reportado por Zhang et al. 1997) se conserva como análisis secundario y exploratorio.

---

## Estructura del repositorio

```text
scripts/      códigos del pipeline principal (01 a 06)
gui/          interfaz gráfica de exploración del c240/VMP (PySide6 + pyqtgraph)
data/         dataset original UCI EEG (NO versionado; se agrega localmente)
outputs/      tablas, métricas y figuras generadas por los scripts
```

La carpeta `data/` no se sube a GitHub. Debe estar disponible localmente para correr el pipeline desde cero. La carpeta `outputs/` contiene los productos derivados; la GUI consume esos archivos ya procesados (no recalcula desde los trials crudos).

---

## Requisitos e instalación

Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

(En Linux/Mac: `source .venv/bin/activate`)

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Dependencias (de `requirements.txt`):

```text
numpy
pandas
scipy
matplotlib
pyarrow
PySide6
pyqtgraph
```

---

## Datos de entrada

El dataset original no se incluye en el repositorio. Para correr el proyecto desde cero, ubicarlo localmente con esta estructura:

```text
data/
  eeg_full/
    co2a0000364.tar.gz    <- sujeto alcohólico (4ta letra 'a')
    co2c0000337.tar.gz    <- sujeto control    (4ta letra 'c')
    ...
```

El grupo de cada sujeto se identifica por la 4ta letra del nombre del archivo (`a` = alcohólico, `c` = control). Cada `.tar.gz` contiene los trials EEG individuales del sujeto. El registro es a **256 Hz, 256 muestras por trial** (1 segundo).

> Importante: los scripts anclan el directorio de trabajo a la raíz del proyecto automáticamente (`os.chdir(...)`), así que pueden ejecutarse desde cualquier carpeta. Las rutas de entrada/salida resuelven siempre contra `outputs/` en la raíz.

---

## Orden de ejecución del pipeline

Correr los scripts en orden. Cada uno depende de la salida del anterior.

### Módulo 01 — Carga del dataset

```powershell
python scripts\01_carga_exploracion.py
```

Carga los `.tar.gz`, parsea los trials individuales y arma un único DataFrame con metadatos (sujeto, grupo, condición, canal, muestra, amplitud en µV). Identifica grupos, condiciones y canales disponibles.

Salidas:

```text
outputs/eeg_data_cargado.parquet
outputs/figura_trial_ejemplo.png
outputs/figura_distribucion_trials.png
```

### Módulo 02 — Exploración en tiempo y frecuencia

```powershell
python scripts\02_exploracion_inicial.py
```

Analiza las señales **crudas** antes del filtrado: visualización temporal de trials individuales, espectro de potencia (PSD) por método de Welch y distribución de amplitudes, para justificar el filtro pasa-banda y el umbral de rechazo de artefactos.

Salidas:

```text
outputs/figura_exploracion_tiempo.png
outputs/figura_exploracion_multicanal_derecho.png
outputs/figura_exploracion_multicanal_izquierdo.png
outputs/figura_exploracion_psd_derecho.png
outputs/figura_exploracion_psd_izquierdo.png
outputs/figura_exploracion_artefactos_derecho.png
outputs/figura_exploracion_artefactos_izquierdo.png
```

### Módulo 03 — Preprocesamiento

```powershell
python scripts\03_preprocesamiento.py
```

Aplica el preprocesamiento sobre las condiciones y canales de interés:

```text
selección de condiciones S1 obj y S2 nomatch
selección de canales temporo-occipitales (derechos + homólogos izquierdos)
filtro Butterworth pasa-banda 0.1–30 Hz (orden 4)
filtrado bidireccional (filtfilt) para fase cero
corrección de offset local con los primeros 50 ms de la época
rechazo de trials con artefactos por umbral ±100 µV
```

Canales analizados:

```text
Hemisferio derecho:    P8, PO8, T8, TP8
Hemisferio izquierdo:  P7, PO7, T7, TP7
```

Nota metodológica: la corrección de los primeros 50 ms es una corrección de offset local (pseudo-baseline), no una línea de base pre-estímulo estricta, dado que las épocas no contienen una ventana pre-estímulo confiable.

Salidas:

```text
outputs/eeg_data_preprocesado.parquet
outputs/figura_preprocesamiento_canales.png
outputs/figura_preprocesamiento_artefacto.png
```

### Módulo 04 — Promediado de trials

```powershell
python scripts\04_promediado.py
```

Calcula el potencial evocado individual por sujeto, canal y condición con dos estrategias:

```text
Promedio homogéneo:   promedio aritmético clásico (todos los trials pesan igual). MÉTODO PRINCIPAL.
Promedio inhomogéneo: promedio ponderado (amplitud variable, ruido constante). Análisis de robustez.
```

Además, para el Grand Average iguala los grupos a una cohorte de referencia (**45 controles + 45 alcohólicos** seleccionados con SEED=42), de modo que ambos GA tengan el mismo N. También calcula una **SNR** por subensambles par/impar. El PE individual se guarda con la muestra completa (77 alc + 45 ctrl) para los scripts 05/06.

Salidas:

```text
outputs/eeg_PE_homogeneo.parquet
outputs/eeg_PE_inhomogeneo.parquet
outputs/eeg_GA_homogeneo.parquet
outputs/eeg_GA_inhomogeneo.parquet
outputs/tabla_snr_comparacion.csv
outputs/sujetos_seleccionados.csv
outputs/figura_GA_derecho.png
outputs/figura_GA_izquierdo.png
outputs/figura_snr_derecho.png
outputs/figura_snr_izquierdo.png
```

### Módulo 05 — Extracción de métricas del c240 y c320

```powershell
python scripts\05_extraccion_pico.py
```

Extrae métricas por sujeto, canal, condición y método de promediado en dos ventanas:

```text
Ventana primaria   c240 / VMP: 220–260 ms
Ventana secundaria c320:        290–340 ms
```

Métricas calculadas en cada ventana:

```text
media en ventana       (métrica principal del análisis)
máximo positivo        (secundaria)
latencia del máximo    (secundaria)
área bajo la curva con signo (secundaria)
```

La métrica principal es la **media firmada en la ventana c240**. El máximo, la latencia, el AUC y la ventana c320 se reportan como secundarios.

Salidas:

```text
outputs/eeg_c240_extraido.csv
outputs/figura_boxplot_derecho_c240_hom.png
outputs/figura_boxplot_izquierdo_c240_hom.png
outputs/figura_boxplot_derecho_c320_hom.png
outputs/figura_boxplot_izquierdo_c320_hom.png
outputs/figura_latencia_derecho_c240.png
outputs/figura_latencia_izquierdo_c240.png
outputs/figura_latencia_derecho_c320.png
outputs/figura_latencia_izquierdo_c320.png
```

### Módulo 06 — Análisis estadístico

```powershell
python scripts\06_estadistica.py
```

Contrasta controles vs alcohólicos para las métricas extraídas. El análisis es **descriptivo**: reporta media ± SD por grupo y la diferencia (control − alcohólico) por canal y condición, e incluye métricas descriptivas simples para las dos hipótesis:

```text
H1: Control > Alcohólico en amplitud del VMP (región temporo-occipital).
H2: El efecto es más pronunciado en el hemisferio derecho (asimetría D − I).
```

Bloques que imprime en consola:

```text
1a. Amplitud media c240 (ventana primaria)
1b. Amplitud media c320 (ventana secundaria)
2.  AUC c240
3.  Latencia c240 y c320 (poco informativa; ver nota en el código)
5.  Análisis secundario: promedio inhomogéneo (robustez)
6.  Métricas descriptivas simples (H1 y H2, con asimetría hemisférica)
```

Salidas:

```text
outputs/tabla_estadistica.csv
outputs/figura_barras_derecho_c240.png
outputs/figura_barras_izquierdo_c240.png
outputs/figura_barras_derecho_c320.png
outputs/figura_barras_izquierdo_c320.png
outputs/figura_lateralizacion_c240.png
outputs/figura_lateralizacion_c320.png
```

---

## Pipeline completo (resumen)

```powershell
python scripts\01_carga_exploracion.py
python scripts\02_exploracion_inicial.py
python scripts\03_preprocesamiento.py
python scripts\04_promediado.py
python scripts\05_extraccion_pico.py
python scripts\06_estadistica.py
python -m gui.app
```

La GUI requiere que los módulos 04, 05 y 06 hayan corrido previamente.

---

## Interfaz gráfica (GUI)

La carpeta `gui/` contiene una aplicación de escritorio (PySide6 + pyqtgraph) para explorar interactivamente los potenciales evocados ya procesados. **No recalcula desde los trials crudos**: consume los archivos de `outputs/` generados por los módulos 04, 05 y 06.

### Cómo ejecutarla

Desde la raíz del proyecto:

```powershell
python -m gui.app
```

o bien:

```powershell
python gui\app.py
```

Si falta algún archivo requerido, la app muestra un cuadro de error indicando cuál falta y qué script lo regenera.

Archivos que la GUI necesita en `outputs/`:

```text
eeg_PE_homogeneo.parquet
eeg_PE_inhomogeneo.parquet
eeg_c240_extraido.csv
tabla_snr_comparacion.csv
tabla_estadistica.csv
```

### Controles (panel lateral izquierdo)

```text
Canal:        los 8 canales (4 derechos + 4 homólogos izquierdos).
Condición:    S1 obj o S2 nomatch.
Método:       homogéneo (línea sólida) o inhomogéneo (punteada).
Superponer:   casilla para dibujar ambos métodos a la vez.
Slider 45–77: re-balancea la cohorte de alcohólicos y recalcula el Grand Average
              desde los PE individuales (NO re-promedia trials crudos). Controles
              fijos en 45. En 45 reproduce el GA igualado del Script 04; en 77 usa
              la muestra completa.
Semilla:      semilla de la selección aleatoria de alcohólicos (default 42).
```

### Qué muestra

```text
Grand Average control (azul) vs alcohólico (rojo).
Banda de SEM entre sujetos.
Ventanas c240 (220–260 ms) y c320 (290–340 ms) sombreadas.
Hover con crosshair: tiempo (ms) y valor µV de cada grupo + diferencia (C−A).
Marcador del pico del promedio (ventana amplia 200–400 ms, solo visual).
```

### Panel de KPIs

```text
Media c240 por grupo (± SD) y diferencia Control − Alcohólico, recalculadas en
vivo sobre la cohorte del slider.
SNR homogénea / inhomogénea (de tabla_snr_comparacion.csv).
Referencia de la muestra completa (todos los sujetos) tomada de tabla_estadistica.csv.
Secundarios: máximo, latencia y AUC en c240, y media c320 (claramente etiquetados).
```

> Nota: el pico mostrado en la GUI se calcula en la ventana amplia 200–400 ms y sirve solo para describir la morfología de la curva (dónde pica el componente, que se desplaza entre c240 y la positividad tardía). La métrica oficial del c240 sigue siendo la **media en la ventana fija 220–260 ms**.

### Módulos de la GUI

```text
gui/
  config.py       constantes (canales, ventanas, SEED, paleta, rutas)
  data_loader.py  carga y validación de outputs/ (error claro si falta algo)
  averaging.py    igualar grupos + Grand Average (réplica del Script 04)
  stats.py        media ± SD por grupo + diferencia + SNR (réplica de los Scripts 05/06)
  plotting.py     widget pyqtgraph (hover µV/ms, banda SEM, sombreados, marcador de pico)
  widgets.py      sidebar de controles + tarjetas de KPI
  app.py          ventana principal, tema oscuro, cableado de señales
```

> Solución de problemas (Windows): si aparece `ImportError: DLL load failed while importing QtGui/QtCore`, ocurre cuando conviven PyQt6 y PySide6 en el entorno. La app ya lo evita fijando `PYQTGRAPH_QT_LIB=PySide6` antes de importar pyqtgraph. Si lo ves al integrar este código en otro lado, exportá esa variable o importá PySide6 antes que pyqtgraph.

---

## Interpretación de las ventanas

```text
c240 / VMP: 220–260 ms   ventana principal. Respuesta asociada a memoria visual
                         en regiones temporo-occipitales.
c320:        290–340 ms   ventana secundaria / exploratoria. Su morfología puede
                         diferir entre estímulos sample (S1) y test (S2 nomatch).
```

Interpretación por condición:

```text
S1 obj:      estímulo sample. Codificación visual inicial.
S2 nomatch:  estímulo test no coincidente. Condición más directamente asociada a
             comparación con memoria visual y al análisis del VMP.
```

---

## Archivos principales de outputs

```text
eeg_data_cargado.parquet       dataset parseado y organizado desde los .tar.gz.
eeg_data_preprocesado.parquet  trials filtrados, corregidos y limpios.
eeg_PE_homogeneo.parquet       PE individuales (promediado homogéneo).
eeg_PE_inhomogeneo.parquet     PE individuales (promediado inhomogéneo).
eeg_GA_homogeneo.parquet       Grand Average homogéneo por grupo/canal/condición.
eeg_GA_inhomogeneo.parquet     Grand Average inhomogéneo por grupo/canal/condición.
eeg_c240_extraido.csv          métricas c240 y c320 por sujeto/canal/condición/método.
tabla_snr_comparacion.csv      SNR homogénea vs inhomogénea (cohorte 45+45).
tabla_estadistica.csv          comparación descriptiva entre grupos (media, SD, diferencia).
sujetos_seleccionados.csv      registro de los 45 alcohólicos elegidos para el GA.
```

---

---

## Grupo 11

Cobián · Obeid · Perelstein — ITBA, PSIB 2026 Q1.