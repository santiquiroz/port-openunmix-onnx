"""Qué se porta y de dónde sale, con el hash para probarlo.

Open-Unmix `umxhq` es el único separador de 4 pistas encontrado cuyos PESOS
llevan una licencia permisiva declarada en el registro donde viven: MIT en el
propio record de Zenodo, no un "MIT" que en realidad cubre solo el código. Esa
distinción es el motivo de este port — ver README.
"""
from __future__ import annotations

from dataclasses import dataclass

# Zenodo 3370489, "Open-Unmix-Pytorch UMX-HQ", licencia mit-license en el record.
ZENODO_RECORD = "https://zenodo.org/records/3370489"
BASE_URL = "https://zenodo.org/records/3370489/files"

SAMPLE_RATE = 44100
N_FFT = 4096
HOP = 1024


@dataclass(frozen=True)
class Target:
    """Una pista. umxhq es un modelo POR pista, no un modelo con cuatro salidas."""

    name: str
    checkpoint: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.checkpoint}?download=1"


TARGETS: tuple[Target, ...] = (
    Target("vocals", "vocals-b62c91ce.pth", ""),
    Target("drums", "drums-9619578f.pth", ""),
    Target("bass", "bass-8d85a5bd.pth", ""),
    Target("other", "other-b52fbbf7.pth", ""),
)

TARGET_NAMES = tuple(t.name for t in TARGETS)
