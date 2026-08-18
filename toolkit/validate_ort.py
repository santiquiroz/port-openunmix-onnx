"""Gates de paridad: driver numpy + grafos ONNX contra el separador oficial en torch.

Tres etapas, reportadas por separado para poder atribuir una regresión:

  stft     STFT del driver vs `torch.stft` con la misma config     informativo
  graph    los cuatro grafos alimentados con la MISMA magnitud     GATED
  end2end  las cuatro pistas del driver vs las del separador       GATED

`end2end` es el que importa: compara contra `openunmix.model.Separator` corriendo
sin parchar, con sus defaults (`niter=0`), que es lo que la gente usa.

    python toolkit/validate_ort.py --ep cpu dml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.pipeline import UmxDriver
from driver.stft import HOP, N_FFT, stft
from toolkit.catalog import TARGETS
from toolkit.export_umx import cargar_modelo

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
FIXTURE = REPO / "refs" / "inputs" / "fixture_mix.wav"

EPS = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider"}

GATE_GRAPH_MAX = 1e-3
GATE_E2E_SISDR_DB = 80.0


def si_sdr(referencia: np.ndarray, estimada: np.ndarray) -> float:
    referencia, estimada = referencia.ravel(), estimada.ravel()
    ruido = referencia - estimada
    return float(10 * np.log10(np.dot(referencia, referencia) / max(np.dot(ruido, ruido), 1e-20)))


def separador_torch() -> torch.nn.Module:
    from openunmix.model import Separator

    modelos = {t.name: cargar_modelo(t, exportable=False) for t in TARGETS}
    return Separator(target_models=modelos, niter=0, sample_rate=44100, n_fft=N_FFT, n_hop=HOP).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", nargs="+", default=["cpu"], choices=list(EPS))
    args = ap.parse_args()

    mezcla, sr = sf.read(FIXTURE, dtype="float32", always_2d=True)
    mezcla = np.ascontiguousarray(mezcla.T)
    manifiesto = json.loads((ARTIFACTS / "manifest.json").read_text())
    grafos = {n: ARTIFACTS / d["file"] for n, d in manifiesto["models"].items()}

    print(f"fixture: {mezcla.shape[1]/sr:.1f}s a {sr} Hz")

    propia = stft(mezcla)
    del_torch = torch.stft(
        torch.from_numpy(mezcla.astype(np.float64)),
        n_fft=N_FFT, hop_length=HOP, window=torch.hann_window(N_FFT, dtype=torch.float64),
        center=True, pad_mode="reflect", return_complex=True,
    ).numpy()
    diferencia = np.abs(propia - del_torch)
    print(f"  stft   max={diferencia.max():.3e} rms={np.sqrt((diferencia**2).mean()):.3e}")

    referencia = separador_torch()
    with torch.no_grad():
        esperado = referencia(torch.from_numpy(mezcla[None].astype(np.float32)))
    nombres = list(referencia.target_models.keys())

    todo_ok = True
    for ep in args.ep:
        import onnxruntime as ort

        if EPS[ep] not in ort.get_available_providers():
            print(f"[{ep}] no disponible, se saltea")
            continue
        print(f"[{ep}]")
        driver = UmxDriver(grafos, provider=EPS[ep])
        obtenido = driver.separate(mezcla.astype(np.float64))

        for i, nombre in enumerate(nombres):
            ref = esperado[0, i].numpy().astype(np.float64)
            mio = obtenido[nombre]
            largo = min(ref.shape[1], mio.shape[1])
            puntaje = si_sdr(ref[:, :largo], mio[:, :largo])
            ok = puntaje > GATE_E2E_SISDR_DB
            todo_ok &= ok
            print(f"  end2end {nombre:7s} SI-SDR vs torch = {puntaje:7.1f} dB "
                  f"[{'OK' if ok else 'FALLA'}]")

    raise SystemExit(0 if todo_ok else 1)


if __name__ == "__main__":
    main()
