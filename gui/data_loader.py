"""
data_loader.py — Carga y validación de los productos de outputs/.

La app NO recalcula nada desde los trials crudos. Consume:

  - eeg_PE_homogeneo.parquet     PE individual por sujeto/canal/condición (Script 04)
  - eeg_PE_inhomogeneo.parquet   idem, promediado inhomogéneo            (Script 04)
  - eeg_c240_extraido.csv        métricas por sujeto (media/max/lat/auc)  (Script 05)
  - tabla_snr_comparacion.csv    SNR par/impar por sujeto (cohorte 45+45) (Script 04)
  - tabla_estadistica.csv        contrastes oficiales Welch + FDR         (Script 06)

Si falta CUALQUIERA de los obligatorios, NO se inventa un fallback: se lanza
MissingDataError con la lista de faltantes y el script que los regenera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

# -----------------------------------------------------------------------------
# Manifiesto de archivos requeridos: clave lógica -> (archivo, script, detalle)
# -----------------------------------------------------------------------------
REQUERIDOS = {
    "pe_homogeneo": (
        "eeg_PE_homogeneo.parquet",
        "scripts/04_promediado_v2.py",
        "PE individual (promediado homogéneo) por sujeto/canal/condición.",
    ),
    "pe_inhomogeneo": (
        "eeg_PE_inhomogeneo.parquet",
        "scripts/04_promediado_v2.py",
        "PE individual (promediado inhomogéneo).",
    ),
    "c240": (
        "eeg_c240_extraido.csv",
        "scripts/05_extraccion_pico.py",
        "Métricas del c240 por sujeto (media/max/latencia/AUC).",
    ),
    "snr": (
        "tabla_snr_comparacion.csv",
        "scripts/04_promediado_v2.py",
        "SNR par/impar por sujeto (cohorte de referencia 45+45).",
    ),
    "estadistica": (
        "tabla_estadistica.csv",
        "scripts/06_estadistica.py",
        "Contrastes oficiales Welch (una cola) + Cohen's d + FDR.",
    ),
}


class MissingDataError(Exception):
    """Falta al menos un archivo obligatorio en outputs/."""

    def __init__(self, faltantes: list[tuple[str, str, str]]):
        self.faltantes = faltantes
        super().__init__(self._mensaje())

    def _mensaje(self) -> str:
        lineas = [
            "No se pudieron cargar todos los datos requeridos desde 'outputs/'.",
            f"Carpeta esperada: {config.OUTPUTS_DIR}",
            "",
            "Archivos faltantes (y cómo regenerarlos):",
        ]
        # Agrupamos por script para sugerir el comando una sola vez.
        scripts = {}
        for archivo, script, detalle in self.faltantes:
            lineas.append(f"  • {archivo}")
            lineas.append(f"      {detalle}")
            scripts.setdefault(script, []).append(archivo)
        lineas.append("")
        lineas.append("Ejecutá (desde la carpeta scripts/):")
        for script in scripts:
            nombre = Path(script).name
            lineas.append(f"  python {nombre}")
        return "\n".join(lineas)


@dataclass
class AppData:
    """Contenedor de todos los datos que la GUI necesita en memoria."""

    pe: dict[str, pd.DataFrame]      # {'homogeneo': df, 'inhomogeneo': df}
    c240: pd.DataFrame               # métricas por sujeto (ambos métodos)
    snr: pd.DataFrame                # SNR por sujeto (cohorte 45+45)
    estadistica: pd.DataFrame        # contrastes oficiales Script 06

    # --- accesos de conveniencia ---------------------------------------------
    def pe_metodo(self, metodo: str) -> pd.DataFrame:
        return self.pe[metodo]

    def canales_disponibles(self) -> list[str]:
        """Canales presentes en los datos, en el orden canónico de config."""
        presentes = set(self.pe[config.METODO_PRINCIPAL]["canal"].unique())
        return [c for c in config.CANALES_INTERES if c in presentes]

    def condiciones_disponibles(self) -> list[str]:
        presentes = set(self.pe[config.METODO_PRINCIPAL]["condicion"].unique())
        return [c for c in config.CONDICIONES if c in presentes]

    def n_alcoholicos_total(self) -> int:
        df = self.pe[config.METODO_PRINCIPAL]
        return int(df[df["grupo"] == "alcoholic"]["sujeto"].nunique())

    def n_control_total(self) -> int:
        df = self.pe[config.METODO_PRINCIPAL]
        return int(df[df["grupo"] == "control"]["sujeto"].nunique())


# -----------------------------------------------------------------------------
# Validación + carga
# -----------------------------------------------------------------------------
def validar_outputs() -> list[tuple[str, str, str]]:
    """Devuelve la lista de (archivo, script, detalle) que faltan en outputs/."""
    faltantes = []
    for _clave, (archivo, script, detalle) in REQUERIDOS.items():
        if not (config.OUTPUTS_DIR / archivo).exists():
            faltantes.append((archivo, script, detalle))
    return faltantes


def _leer(archivo: str) -> Path:
    return config.OUTPUTS_DIR / archivo


def cargar_datos() -> AppData:
    """
    Valida y carga todos los productos requeridos.

    Lanza:
        MissingDataError  si falta algún archivo obligatorio.
    """
    faltantes = validar_outputs()
    if faltantes:
        raise MissingDataError(faltantes)

    pe = {
        "homogeneo": pd.read_parquet(_leer(REQUERIDOS["pe_homogeneo"][0])),
        "inhomogeneo": pd.read_parquet(_leer(REQUERIDOS["pe_inhomogeneo"][0])),
    }
    c240 = pd.read_csv(_leer(REQUERIDOS["c240"][0]))
    snr = pd.read_csv(_leer(REQUERIDOS["snr"][0]))
    estadistica = pd.read_csv(_leer(REQUERIDOS["estadistica"][0]))

    return AppData(pe=pe, c240=c240, snr=snr, estadistica=estadistica)
