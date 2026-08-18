"""¿Sirve fp16 acá? Medición corta, porque la respuesta esperada es "no".

La ganancia de fp16 es de memoria, no de aritmética (medido en el port de
RoFormer: 12x en un grafo que no entra en VRAM, 9% en uno que sí). Estos grafos
pesan 34 MiB cada uno y la red es una LSTM secuencial que ni siquiera aprovecha
la GPU — el torch original en CPU ya le gana. Así que esto se mide para poder
cerrarlo con un número en vez de con una intuición.

    python toolkit/fp16_check.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from driver.pipeline import UmxDriver

ARTIFACTS = REPO / "artifacts"
FP16 = ARTIFACTS / "fp16"
FIXTURE = REPO / "refs" / "inputs" / "fixture_mix.wav"


def convertir(src: Path, destino: Path) -> None:
    import onnx
    from onnxruntime.transformers import float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    destino.unlink(missing_ok=True)
    convertido = float16.convert_float_to_float16(
        onnx.load(str(src)), keep_io_types=True, op_block_list=["Pow", "ReduceMean", "Sqrt", "Div"]
    )
    del convertido.graph.value_info[:]
    envoltorio = OnnxModel(convertido)
    envoltorio.topological_sort()
    onnx.save(envoltorio.model, str(destino))


def si_sdr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    ruido = a - b
    return float(10 * np.log10(np.dot(a, a) / max(np.dot(ruido, ruido), 1e-20)))


def main() -> None:
    FP16.mkdir(parents=True, exist_ok=True)
    manifiesto = json.loads((ARTIFACTS / "manifest.json").read_text())
    fp32 = {n: ARTIFACTS / d["file"] for n, d in manifiesto["models"].items()}

    total32 = total16 = 0
    fp16 = {}
    for nombre, ruta in fp32.items():
        destino = FP16 / f"{ruta.stem}_fp16.onnx"
        convertir(ruta, destino)
        fp16[nombre] = destino
        total32 += ruta.stat().st_size
        total16 += destino.stat().st_size
    print(f"tamano  {total32/2**20:.1f} MiB -> {total16/2**20:.1f} MiB "
          f"({100*(1-total16/total32):.1f}% menos)")

    mezcla, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
    mezcla = np.ascontiguousarray(mezcla.T).astype(np.float64)
    duracion = mezcla.shape[1] / sr

    for etiqueta, grafos in (("fp32", fp32), ("fp16", fp16)):
        for ep in ("CPUExecutionProvider", "DmlExecutionProvider"):
            driver = UmxDriver(grafos, provider=ep)
            driver.separate(mezcla)  # calentamiento
            tiempos = []
            for _ in range(3):
                arranque = time.perf_counter()
                salida = driver.separate(mezcla)
                tiempos.append(time.perf_counter() - arranque)
            mejor = min(tiempos)
            print(f"{etiqueta} [{ep[:3].lower()}] mejor={mejor:5.2f}s -> {duracion/mejor:5.2f}x tiempo real")
            if etiqueta == "fp32" and ep == "CPUExecutionProvider":
                referencia = salida
            if etiqueta == "fp16" and ep == "CPUExecutionProvider":
                for nombre in referencia:
                    print(f"    fidelidad {nombre:7s} {si_sdr(referencia[nombre], salida[nombre]):6.1f} dB")


if __name__ == "__main__":
    main()
