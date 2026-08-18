"""Cuánto tarda separar, por proveedor, y contra el torch original.

Se mide la separación COMPLETA de las cuatro pistas, que es lo que cuesta un
trabajo real — no una llamada suelta al grafo. El número que se cita es el de
punta a punta sobre el fixture, en múltiplos de tiempo real.

    python toolkit/bench.py --ep cpu dml --runs 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.pipeline import UmxDriver

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
FIXTURE = REPO / "refs" / "inputs" / "fixture_mix.wav"
EPS = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", nargs="+", default=["cpu", "dml"], choices=list(EPS))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--torch", action="store_true", help="medir tambien el original")
    args = ap.parse_args()

    mezcla, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
    mezcla = np.ascontiguousarray(mezcla.T).astype(np.float64)
    duracion = mezcla.shape[1] / sr
    manifiesto = json.loads((ARTIFACTS / "manifest.json").read_text())
    grafos = {n: ARTIFACTS / d["file"] for n, d in manifiesto["models"].items()}

    import onnxruntime as ort

    for ep in args.ep:
        if EPS[ep] not in ort.get_available_providers():
            print(f"{ep}: no disponible")
            continue
        driver = UmxDriver(grafos, provider=EPS[ep])
        for _ in range(args.warmup):
            driver.separate(mezcla)
        tiempos = []
        for _ in range(args.runs):
            arranque = time.perf_counter()
            driver.separate(mezcla)
            tiempos.append(time.perf_counter() - arranque)
        mejor, mediana = min(tiempos), statistics.median(tiempos)
        print(f"onnx [{ep}]  mejor={mejor:6.2f}s  mediana={mediana:6.2f}s  "
              f"-> {duracion/mejor:5.2f}x tiempo real ({duracion:.1f}s de audio, 4 pistas)")

    if args.torch:
        import torch

        from toolkit.validate_ort import separador_torch

        separador = separador_torch()
        entrada = torch.from_numpy(mezcla[None].astype(np.float32))
        with torch.no_grad():
            separador(entrada)
            tiempos = []
            for _ in range(args.runs):
                arranque = time.perf_counter()
                separador(entrada)
                tiempos.append(time.perf_counter() - arranque)
        mejor = min(tiempos)
        print(f"torch [cpu] mejor={mejor:6.2f}s  -> {duracion/mejor:5.2f}x tiempo real")


if __name__ == "__main__":
    main()
