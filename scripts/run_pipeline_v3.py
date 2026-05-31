"""
==============================================================================
Script auxiliar: correr pipeline v3 completo
==============================================================================

Uso:
    python .\\scripts\\run_pipeline_v3.py
==============================================================================
"""

from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name.lower() == "scripts" else Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"

scripts = [
    "03_preprocesamiento_v3.py",
    "04_promediado_PotencialesEvocados_v3.py",
    "05_extraccion_c240_v3.py",
    "06_estadistica_v3.py",
    "07_latencia_v3.py",
    "08_efecto_nonmatch_match_v3.py",
    "09_robustez_v3.py",
    "10_especificidad_regional_v3.py",
]

print("=" * 70)
print("Corriendo pipeline v3 completo")
print("=" * 70)

for s in scripts:
    path = SCRIPTS_DIR / s
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    print("\n" + "=" * 70)
    print(f"Ejecutando {s}")
    print("=" * 70)
    subprocess.run([sys.executable, str(path)], cwd=str(PROJECT_DIR), check=True)

print("\n[OK] Pipeline v3 completo.")
