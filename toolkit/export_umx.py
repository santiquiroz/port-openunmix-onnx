"""Exporta los cuatro modelos de umxhq a ONNX y verifica la paridad ahí mismo.

Un modelo por pista, cada uno de magnitud a magnitud: entra el espectrograma de
la mezcla y sale el espectrograma estimado de esa pista. La fase nunca entra al
grafo — la aporta el driver al reconstruir, igual que en el port de RoFormer.

La forma del tensor es la que espera Open-Unmix en torch:
`[lote, canales, bins, cuadros]`, con bins = n_fft//2 + 1 = 2049. El eje de
cuadros queda DINÁMICO: a diferencia de RoFormer, este modelo es una LSTM sobre
el eje temporal y no cachea nada por longitud, así que un solo grafo sirve para
cualquier duración de audio.

    python toolkit/export_umx.py            # las cuatro pistas
    python toolkit/export_umx.py vocals     # una sola
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit.catalog import HOP, N_FFT, SAMPLE_RATE, TARGETS, Target

REPO = Path(__file__).resolve().parents[1]
CHECKPOINTS = REPO / "checkpoints"
ARTIFACTS = REPO / "artifacts"

OPSET = 17
# Con qué se prueba la paridad tras exportar. 256 cuadros ~ 6 s de audio.
CUADROS_PRUEBA = 256
GATE_PARIDAD = 1e-4


def cargar_modelo(target: Target, exportable: bool = True) -> torch.nn.Module:
    """`exportable=False` devuelve el modelo de upstream, que es contra quien se compara."""
    from openunmix.model import OpenUnmix

    from toolkit.spec_model import OpenUnmixExportable

    Clase = OpenUnmixExportable if exportable else OpenUnmix

    estado = torch.load(CHECKPOINTS / target.checkpoint, map_location="cpu", weights_only=True)
    # Dos anchos distintos, y confundirlos rompe la carga: `input_mean` cubre solo
    # la banda que el modelo mira (1487 bins ~ 16 kHz) y `output_mean` el espectro
    # completo (2049), porque estima la pista entera aunque solo escuche abajo.
    bins_salida = estado["output_mean"].shape[0]
    bins_entrada = estado["input_mean"].shape[0]
    modelo = Clase(
        nb_bins=bins_salida,
        nb_channels=2,
        hidden_size=estado["fc1.weight"].shape[0],
        max_bin=bins_entrada,
    )
    modelo.load_state_dict(estado, strict=False)
    modelo.eval()
    # El modelo aprende su propia normalización de entrada; sin esto la salida es
    # ruido pero el export "funciona", que es la peor forma de fallar.
    modelo.freeze()
    return modelo


def exportar(target: Target) -> dict:
    modelo = cargar_modelo(target)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    destino = ARTIFACTS / f"umxhq_{target.name}.onnx"

    # El modelo permuta adentro: recibe [lote, canales, bins, cuadros].
    ejemplo = torch.rand(1, 2, N_FFT // 2 + 1, CUADROS_PRUEBA, dtype=torch.float32) * 10.0
    arranque = time.perf_counter()
    torch.onnx.export(
        modelo,
        ejemplo,
        str(destino),
        input_names=["mag"],
        output_names=["estimate"],
        # Solo el eje de cuadros: los bins los fija la STFT y el canal es estéreo
        # por construcción del modelo, así que dejarlos dinámicos mentiría.
        dynamic_axes={"mag": {3: "frames"}, "estimate": {3: "frames"}},
        opset_version=OPSET,
        dynamo=False,
    )
    tardanza = time.perf_counter() - arranque

    paridad = verificar_paridad(cargar_modelo(target, exportable=False), destino)
    tamano = destino.stat().st_size
    print(f"{target.name:7s} exportado en {tardanza:5.1f}s  {tamano/2**20:6.1f} MiB  "
          f"paridad max={paridad['max']:.3e} rms={paridad['rms']:.3e} "
          f"[{'OK' if paridad['ok'] else 'FALLA'}]")
    return {
        "file": destino.name,
        "target": target.name,
        "checkpoint": target.checkpoint,
        "checkpoint_sha256": _sha256(CHECKPOINTS / target.checkpoint),
        "graph_sha256": _sha256(destino),
        "bytes": tamano,
        "parity_vs_torch": paridad,
    }


def verificar_paridad(modelo: torch.nn.Module, grafo: Path) -> dict:
    """El grafo tiene que dar lo mismo que el torch del que salió, con OTRA entrada.

    Reusar el tensor del trazado no probaría nada: un export roto que devuelva
    una constante pasaría igual.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(20260817)
    entrada = (rng.random((1, 2, N_FFT // 2 + 1, CUADROS_PRUEBA + 37)) * 10.0).astype(np.float32)
    with torch.no_grad():
        esperado = modelo(torch.from_numpy(entrada)).numpy()
    del modelo
    sesion = ort.InferenceSession(str(grafo), providers=["CPUExecutionProvider"])
    obtenido = sesion.run(None, {"mag": entrada})[0]

    diferencia = np.abs(esperado.astype(np.float64) - obtenido.astype(np.float64))
    maximo, rms = float(diferencia.max()), float(np.sqrt((diferencia**2).mean()))
    return {"max": maximo, "rms": rms, "ok": maximo < GATE_PARIDAD}


def _sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=None)
    args = ap.parse_args()

    pedidos = [t for t in TARGETS if not args.names or t.name in args.names]
    if not pedidos:
        raise SystemExit(f"nombres validos: {[t.name for t in TARGETS]}")

    entradas = [exportar(t) for t in pedidos]
    fallidos = [e["target"] for e in entradas if not e["parity_vs_torch"]["ok"]]

    manifiesto = ARTIFACTS / "manifest.json"
    previo = json.loads(manifiesto.read_text()) if manifiesto.exists() else {}
    modelos = previo.get("models", {})
    modelos.update({e["target"]: e for e in entradas})
    manifiesto.write_text(
        json.dumps(
            {
                "model": "openunmix umxhq",
                "license": "MIT (Zenodo record 3370489)",
                "opset": OPSET,
                "sample_rate": SAMPLE_RATE,
                "stft": {"n_fft": N_FFT, "hop": HOP, "window": "hann periodic", "center": True},
                "graph_io": {
                    "input": "mag [1, 2, 2049, frames] float32 (magnitud de la mezcla)",
                    "output": "estimate [1, 2, 2049, frames] float32 (magnitud de la pista)",
                    "dynamic": ["frames"],
                },
                "models": modelos,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nmanifest: {manifiesto}")
    if fallidos:
        raise SystemExit(f"paridad FALLIDA en: {fallidos}")


if __name__ == "__main__":
    main()
