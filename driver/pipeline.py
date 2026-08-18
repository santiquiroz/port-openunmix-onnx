"""De un wav estéreo a cuatro pistas, sin torch y sin librosa.

Con los valores por defecto de Open-Unmix (`niter=0`, sin softmask) el separador
oficial hace exactamente esto: toma la magnitud que estima cada modelo y le pone
la fase de la mezcla. No hay filtro de Wiener ni EM que reproducir — están
detrás de `niter>0`, que no es el default. Esto no es una aproximación del
original: es el original.

    from driver.pipeline import UmxDriver
    pistas = UmxDriver(grafos).separate(audio)   # {"vocals": [2, N], ...}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from driver.stft import HOP, N_FFT, istft, stft

TARGETS = ("vocals", "drums", "bass", "other")


class UmxDriver:
    def __init__(self, graphs: dict[str, Path], provider: str = "CPUExecutionProvider") -> None:
        import onnxruntime as ort

        self.sessions = {
            nombre: ort.InferenceSession(str(ruta), providers=[provider])
            for nombre, ruta in graphs.items()
        }

    def separate(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """audio [canales, muestras] float -> una pista por modelo, misma forma."""
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise ValueError(f"se espera audio estereo [2, muestras], llego {audio.shape}")

        espectro = stft(audio, N_FFT, HOP)
        magnitud = np.abs(espectro)[None].astype(np.float32)
        fase = np.exp(1j * np.angle(espectro))
        muestras = audio.shape[1]

        pistas: dict[str, np.ndarray] = {}
        for nombre, sesion in self.sessions.items():
            estimada = sesion.run(None, {"mag": magnitud})[0][0]
            pistas[nombre] = istft(estimada * fase, muestras, N_FFT, HOP)
        return pistas


def residual(mezcla: np.ndarray, pistas: dict[str, np.ndarray]) -> np.ndarray:
    """Lo que no se llevó ninguna pista. Útil para oír qué quedó afuera."""
    return mezcla - sum(pistas.values())
