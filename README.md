# PSIB2026Q1-TPS-Grupo11-Cobian_Obeid_Perelstein

Trabajo práctico integrador de Procesamiento de Señales e Imágenes Biomédicas. Este proyecto analiza potenciales evocados visuales en señales EEG del dataset UCI EEG Database, con foco en el componente c240/VMP asociado a memoria visual, comparando sujetos controles y sujetos con alcoholismo.

El análisis se centra en dos condiciones del paradigma de reconocimiento visual:

```text
S1 obj       estímulo sample / codificación visual inicial
S2 nomatch   estímulo test / comparación con memoria visual
```

La ventana principal del trabajo es el componente c240/VMP entre 220 y 260 ms. La ventana 290-340 ms, asociada al componente tardío c320 reportado por Zhang et al., se conserva como análisis secundario y exploratorio.

## Estructura general

El pipeline está organizado en scripts numerados dentro de la carpeta `scripts/`.

La carpeta `data/` no se versiona en GitHub. Debe estar disponible localmente para ejecutar el proyecto desde cero. Los resultados derivados se guardan en `outputs/` y la interfaz gráfica consume esos archivos ya procesados.

Carpetas principales:

```text
scripts/                    códigos del pipeline principal
scripts/legacy/             códigos preliminares o de respaldo, si corresponde
gui/                        interfaz gráfica de exploración del c240/VMP
data/eeg_full/              dataset original UCI EEG, no versionado
outputs/                    tablas, métricas y archivos procesados generados
outputs/figures/            figuras seleccionadas para documentación, si corresponde
```

## Requisitos

Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Dependencias principales:

```text
numpy
pandas
scipy
matplotlib
pyarrow
PySide6
pyqtgraph
```

## Datos de entrada

El dataset original no se sube al repositorio. Para correr el proyecto desde cero, debe ubicarse localmente con la siguiente estructura:

```text
data/
  eeg_full/
    co2a0000364.tar.gz
    co2c0000337.tar.gz
    ...
```

La identificación del grupo se realiza a partir del nombre del archivo:

```text
co2a...   sujeto alcohólico
co2c...   sujeto control
```

Cada archivo comprimido contiene trials EEG individuales. El dataset se registra a 256 Hz, con 256 muestras por trial.

## Orden de ejecución

### Módulo 01: carga y exploración inicial del dataset

```powershell
python scripts\01_carga_exploracion.py
```

Este módulo carga los archivos `.tar.gz` del dataset, parsea los trials individuales y organiza la información en un único DataFrame con metadatos de sujeto, grupo, condición, canal, muestra y amplitud en microvoltios.

Salidas principales:

```text
outputs/eeg_data_cargado.parquet
```

El módulo también permite verificar la cantidad de sujetos, grupos, condiciones, canales disponibles y rango de amplitudes.

### Módulo 02: exploración inicial en tiempo y frecuencia

```powershell
python scripts\02_exploracion_inicial.py
```

Este módulo analiza las señales crudas antes del filtrado. Incluye visualización temporal de trials individuales, análisis espectral mediante Welch y distribución de amplitudes para justificar el rechazo de artefactos.

Análisis realizados:

```text
visualización de señales crudas
PSD mediante Welch
comparación entre grupos y canales
histogramas de amplitud
justificación del filtrado y del umbral de artefactos
```

Salidas principales:

```text
outputs/figura_exploracion_tiempo.png
outputs/figura_exploracion_multicanal.png
outputs/figura_exploracion_psd.png
outputs/figura_exploracion_artefactos.png
```

### Módulo 03: preprocesamiento de señales EEG

```powershell
python scripts\03_preprocesamiento.py
```

Este módulo aplica el preprocesamiento principal sobre las condiciones y canales de interés.

Pasos incluidos:

```text
selección de condiciones S1 obj y S2 nomatch
selección de canales temporo-occipitales derechos e izquierdos
filtro Butterworth pasa-banda 0.1-30 Hz
filtrado bidireccional con filtfilt para fase cero
corrección de offset local usando los primeros 50 ms de la época
rechazo de trials con artefactos por umbral ±100 µV
```

Canales analizados:

```text
Hemisferio derecho:    P8, PO8, T8, TP8
Hemisferio izquierdo:  P7, PO7, T7, TP7
```

Nota metodológica: la corrección de los primeros 50 ms se interpreta como una corrección de offset local o pseudo-baseline, no como una línea de base pre-estímulo estricta, dado que las épocas utilizadas no contienen una ventana pre-estímulo confiable.

Salidas principales:

```text
outputs/eeg_data_preprocesado.parquet
```

### Módulo 04: promediado de trials

```powershell
python scripts\04_promediado.py
```

Este módulo calcula el potencial evocado individual por sujeto, canal y condición mediante dos estrategias de promediado.

Métodos incluidos:

```text
Promedio homogéneo:
promedio aritmético clásico, todos los trials pesan igual.

Promedio inhomogéneo:
promedio ponderado considerando amplitud variable entre trials y varianza de ruido constante.
```

El promedio homogéneo se utiliza como método principal del trabajo. El promedio inhomogéneo se conserva como análisis secundario y de robustez metodológica.

Además, el módulo iguala grupos para una cohorte de referencia:

```text
45 controles
45 alcohólicos seleccionados aleatoriamente con SEED fijo
```

Esto permite comparar Grand Averages con el mismo número de sujetos por grupo. También se calcula una SNR basada en subensambles par/impar.

Salidas principales:

```text
outputs/eeg_PE_homogeneo.parquet
outputs/eeg_PE_inhomogeneo.parquet
outputs/eeg_GA_homogeneo.parquet
outputs/eeg_GA_inhomogeneo.parquet
outputs/tabla_snr_comparacion.csv
outputs/sujetos_seleccionados.csv
```

### Módulo 05: extracción de métricas del c240 y c320

```powershell
python scripts\05_extraccion_pico.py
```

Este módulo extrae métricas por sujeto, canal, condición y método de promediado.

Ventana principal:

```text
c240 / VMP: 220-260 ms
```

Ventana secundaria:

```text
c320: 290-340 ms
```

Métricas calculadas:

```text
media en ventana
máximo positivo en ventana
latencia del máximo
área bajo la curva con signo
```

La métrica principal del análisis es la media firmada en la ventana c240. El máximo, la latencia, el AUC y la ventana c320 se reportan como métricas secundarias.

Salidas principales:

```text
outputs/eeg_c240_extraido.csv
outputs/figura_boxplot_derecho_c240.png
outputs/figura_boxplot_izquierdo_c240.png
outputs/figura_boxplot_derecho_c320.png
outputs/figura_boxplot_izquierdo_c320.png
outputs/figura_latencia_derecho.png
outputs/figura_latencia_izquierdo.png
```

### Módulo 06: análisis estadístico

```powershell
python scripts\06_estadistica.py
```

Este módulo realiza el contraste estadístico entre controles y alcohólicos para las métricas extraídas.

Análisis principal:

```text
método: homogéneo
ventana: c240, 220-260 ms
métrica: media en ventana
hipótesis: Control > Alcohólico
test: t-test de Welch una cola
corrección por múltiples comparaciones: FDR Benjamini-Hochberg
```

El test de Welch se utiliza porque compara dos grupos independientes sin asumir igualdad de varianzas. La hipótesis una cola se justifica porque la dirección esperada Control > Alcohólico fue definida a priori a partir del marco teórico.

Análisis secundarios:

```text
c320 290-340 ms
AUC c240
latencia c240 y c320
lateralización hemisférica
promedio inhomogéneo como robustez
```

Salidas principales:

```text
outputs/tabla_estadistica.csv
outputs/tabla_lateralizacion.csv
outputs/figura_barras_derecho_c240.png
outputs/figura_barras_izquierdo_c240.png
outputs/figura_barras_derecho_c320.png
outputs/figura_barras_izquierdo_c320.png
outputs/figura_lateralizacion_c240.png
outputs/figura_lateralizacion_c320.png
```

## Interfaz gráfica

La carpeta `gui/` contiene una aplicación de escritorio desarrollada con PySide6 y pyqtgraph para explorar los potenciales evocados ya procesados.

Ejecutar desde la raíz del proyecto:

```powershell
python -m gui.app
```

También puede ejecutarse como:

```powershell
python gui\app.py
```

La GUI no recalcula desde los trials crudos. Consume los archivos generados por los módulos 04, 05 y 06.

Archivos requeridos:

```text
outputs/eeg_PE_homogeneo.parquet
outputs/eeg_PE_inhomogeneo.parquet
outputs/eeg_c240_extraido.csv
outputs/tabla_snr_comparacion.csv
outputs/tabla_estadistica.csv
```

Controles disponibles:

```text
selección de canal
selección de condición
método de promediado homogéneo o inhomogéneo
superposición de métodos
slider de balanceo de grupos
semilla aleatoria para selección de alcohólicos
```

La interfaz muestra:

```text
Grand Average control vs alcohólico
banda de error estándar entre sujetos
ventanas c240 y c320 sombreadas
pico visual del promedio en ventana amplia 150-400 ms
media c240 por grupo
diferencia Control - Alcohólico
SNR homogénea e inhomogénea
referencia estadística de la muestra completa
métricas secundarias
```

Nota: el pico visual mostrado en la GUI se calcula en una ventana amplia de 150 a 400 ms y se usa solo para describir la morfología de la curva. La métrica oficial del c240 sigue siendo la media en la ventana fija 220-260 ms.

## Interpretación de las ventanas

El análisis principal se apoya en la ventana c240/VMP:

```text
c240 / VMP: 220-260 ms
```

Esta ventana se utiliza para cuantificar la respuesta asociada a memoria visual en regiones temporo-occipitales.

La ventana c320 se mantiene como secundaria:

```text
c320: 290-340 ms
```

Esta ventana se interpreta de forma exploratoria, ya que su morfología puede diferir entre estímulos sample y test. En S1 puede observarse como una positividad tardía de codificación visual, mientras que en S2 nomatch puede no comportarse como una positividad directamente comparable.

Interpretación por condición:

```text
S1 obj:
representa el estímulo sample. Se interpreta principalmente como codificación visual inicial.

S2 nomatch:
representa el estímulo test no coincidente. Es la condición más directamente asociada a comparación con memoria visual y al análisis del VMP.
```

## Pipeline completo

Para reproducir el análisis completo desde cero, ejecutar:

```powershell
python scripts\01_carga_exploracion.py
python scripts\02_exploracion_inicial.py
python scripts\03_preprocesamiento.py
python scripts\04_promediado.py
python scripts\05_extraccion_pico.py
python scripts\06_estadistica.py
python -m gui.app
```

La GUI requiere que los módulos 04, 05 y 06 hayan sido ejecutados previamente. El módulo 04 puede incluir una selección aleatoria reproducible de sujetos alcohólicos mediante la semilla configurada.

## Archivos principales de outputs

```text
eeg_data_cargado.parquet:
dataset parseado y organizado desde los archivos originales.

eeg_data_preprocesado.parquet:
trials filtrados, corregidos y limpios.

eeg_PE_homogeneo.parquet:
potenciales evocados individuales con promediado homogéneo.

eeg_PE_inhomogeneo.parquet:
potenciales evocados individuales con promediado inhomogéneo.

eeg_GA_homogeneo.parquet:
Grand Average homogéneo por grupo, canal y condición.

eeg_GA_inhomogeneo.parquet:
Grand Average inhomogéneo por grupo, canal y condición.

eeg_c240_extraido.csv:
métricas c240 y c320 por sujeto, canal, condición y método.

tabla_snr_comparacion.csv:
comparación de SNR entre promediado homogéneo e inhomogéneo.

tabla_estadistica.csv:
tests Welch, tamaño de efecto y corrección FDR.

tabla_lateralizacion.csv:
análisis de lateralización entre hemisferio derecho e izquierdo.
```

## Códigos heredados

La carpeta `scripts/legacy/` puede contener versiones preliminares usadas durante el desarrollo. No forman parte del pipeline final, pero se conservan como respaldo.

Los scripts oficiales del pipeline son los numerados del 01 al 06.

## Nota sobre los datos

El dataset original no se sube al repositorio. La carpeta `data/` debe agregarse localmente con la estructura esperada por los scripts. Los resultados generados por el pipeline se guardan en `outputs/`.

## Limitaciones metodológicas

Este trabajo utiliza una corrección de offset local con los primeros 50 ms de cada época, ya que no se dispone de una ventana pre-estímulo confiable. Por este motivo, la interpretación se centra en diferencias relativas entre grupos y en ventanas temporales definidas a priori.

El componente c320 se reporta como secundario y exploratorio. La conclusión principal del trabajo se basa en la media del c240/VMP entre 220 y 260 ms.

El promedio inhomogéneo se utiliza como comparación metodológica, no como resultado principal, debido a que sus pesos dependen del parecido de cada trial con el promedio estimado.
