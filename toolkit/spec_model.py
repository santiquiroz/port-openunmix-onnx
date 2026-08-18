"""El único cambio al modelo antes de trazarlo, y por qué no toca ningún peso.

`OpenUnmix.forward` lee la forma del tensor a variables de Python
(`nb_frames, nb_samples, ... = x.data.shape`) y después las usa dentro de dos
`reshape`. Al trazar, esas variables se congelan como constantes: el grafo queda
atado a la cantidad de cuadros del ejemplo y falla con cualquier otra longitud
con `input_shape_size == size was false`.

Este subclase reescribe esos dos `reshape` para que deduzcan el eje del tensor en
vez de recibirlo horneado. La red es idéntica — mismos módulos, mismo orden,
mismos pesos — y la prueba es el gate de paridad de `export_umx.py`, que compara
contra el modelo ORIGINAL con una cantidad de cuadros distinta a la trazada.

Es la misma clase de cambio "neutral al export" que necesitó el port de RoFormer
para su GLU: nada de lo que decide la salida cambia, solo cómo queda escrito en
el grafo.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from openunmix.model import OpenUnmix
from torch import Tensor


class OpenUnmixExportable(OpenUnmix):
    """`OpenUnmix` con las dos formas deducidas en vez de constantes."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.shape

        mix = x.detach().clone()

        x = x[..., : self.nb_bins]
        x = x + self.input_mean
        x = x * self.input_scale

        x = self.fc1(x.reshape(-1, nb_channels * self.nb_bins))
        x = self.bn1(x)
        # Antes: reshape(nb_frames, nb_samples, hidden). El -1 deduce los cuadros.
        x = x.reshape(-1, nb_samples, self.hidden_size)
        x = torch.tanh(x)

        lstm_out = self.lstm(x)
        x = torch.cat([x, lstm_out[0]], -1)

        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.bn3(x)

        # Antes: reshape(nb_frames, nb_samples, nb_channels, nb_output_bins).
        x = x.reshape(-1, nb_samples, nb_channels, self.nb_output_bins)

        x = x * self.output_scale
        x = x + self.output_mean

        x = F.relu(x) * mix
        return x.permute(1, 2, 3, 0)
