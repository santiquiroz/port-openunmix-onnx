# port-openunmix-onnx

**Open-Unmix `umxhq` (4 pistas: voz / batería / bajo / resto) exportado a ONNX, con un driver en numpy — sin torch en tiempo de inferencia.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ONNX opset](https://img.shields.io/badge/ONNX%20opset-17-005CED.svg)](#qué-hay-adentro)
[![Weights: MIT](https://img.shields.io/badge/pesos-MIT_(Zenodo)-22c55e.svg)](https://zenodo.org/records/3370489)

## Por qué existe

Separar una canción en cuatro pistas es fácil de encontrar y difícil de **redistribuir**. Los
modelos buenos —BS-RoFormer, Demucs, la familia de MSST— tienen los pesos sin licencia, con
licencia que cubre solo el código, o marcados "solo para uso científico". El survey completo
está en [port-bs-roformer-onnx](https://github.com/santiquiroz/port-bs-roformer-onnx#readme):
de todos los separadores de 4 pistas de calidad alta, **ninguno** trae una licencia que permita
republicar los pesos.

Open-Unmix `umxhq` sí: **MIT declarado en el propio record de Zenodo**, sobre los `.pth`, no
sobre el código de otro repo. Es de otra generación —una BiLSTM de 2019, ~5.4 dB de SDR promedio
contra los ~9.4 de los roformer— y esa diferencia se oye. Pero es el único que **se puede
publicar**, y un separador que existe le gana a uno que no.

Los grafos ONNX de este repo están en el release, listos para bajar.

## Qué hay adentro

Un modelo por pista, de magnitud a magnitud: entra el espectrograma de la mezcla y sale el de
esa pista. La fase nunca toca el grafo — la pone el driver al reconstruir.

```
mezcla wav 44.1 kHz estéreo
  → STFT (n_fft 4096, hop 1024, Hann periódica, center)   driver/stft.py
  → magnitud [1, 2, 2049, cuadros]
  → 4 grafos ONNX (uno por pista)                          artifacts/*.onnx
  → magnitud estimada × fase de la mezcla                  driver/pipeline.py
  → iSTFT (WOLA) → 4 wavs
```

**Eso ES el separador oficial, no una aproximación.** Con los valores por defecto de Open-Unmix
(`niter=0`, sin softmask) el `Separator` de upstream hace exactamente esta cuenta; el filtro de
Wiener y el EM viven detrás de `niter>0`, que no es el default.

El eje de cuadros es **dinámico**: un solo grafo sirve para cualquier duración. Eso costó un
cambio, el único que se le hace al modelo antes de trazarlo, y no toca ningún peso: el
`forward` de upstream lee la cantidad de cuadros a una variable de Python y la usa dentro de dos
`reshape`, que al trazar quedan congelados. `toolkit/spec_model.py` los reescribe para que
deduzcan el eje. La prueba de que es equivalente es el gate de paridad, que compara contra el
modelo **de upstream sin tocar** y con otra cantidad de cuadros que la trazada.

## Paridad

Contra `openunmix.model.Separator` sin parchar, con sus defaults, sobre un fixture de 12 s:

| Etapa | CPU EP | DirectML EP |
|---|---|---|
| `stft` — driver numpy vs `torch.stft` | max 1.21e-13, rms 2.39e-15 | igual (no interviene el EP) |
| `end2end` voz | **132.8 dB** SI-SDR | **132.8 dB** |
| `end2end` batería | 123.9 dB | 123.0 dB |
| `end2end` bajo | 135.5 dB | 134.0 dB |
| `end2end` resto | 109.8 dB | 109.4 dB |

Paridad del export, grafo contra torch con entrada aleatoria y otra longitud que la trazada:
máximo **5.5e-06 a 1.1e-05** según la pista.

```powershell
.venv\Scripts\python.exe toolkit\validate_ort.py --ep cpu dml
```

## Velocidad — y por qué acá ONNX no gana

Separación completa de las cuatro pistas, fixture de 12 s, Ryzen + RX 7800 XT:

| Camino | Mejor | Tiempo real |
|---|---:|---:|
| torch CPU (el original) | 0.26 s | **46.5x** |
| ONNX CPU EP | 0.50 s | 24.0x |
| ONNX DirectML | 0.64 s | 18.8x |

**El original es casi el doble de rápido que el port, y la GPU es más lenta que el procesador.**
No es un error de medición: esto es una LSTM chica de tres capas, donde el cuello es una
recurrencia secuencial que no paraleliza — la GPU paga el viaje de datos y no lo recupera, y los
kernels LSTM de torch (MKL) están mejor afinados que los de onnxruntime.

Entonces el motivo para usar esto **no es velocidad**: es no arrastrar torch a producción y
correr donde haya un onnxruntime. Si ya tenés torch instalado y corrés en CPU, usá el original.

Para escala: 19x tiempo real son unos 13 segundos por canción de 4 minutos, con las cuatro
pistas. Un RoFormer de una sola pista, en la misma placa, va a ~1.2x.

## Uso

```powershell
# 1. entorno
python -m venv .venv
.venv\Scripts\python.exe -m pip install openunmix onnx onnxruntime-directml soundfile

# 2. checkpoints originales (Zenodo, MIT) -> checkpoints/
#    o bajá los .onnx ya exportados del release y saltá al paso 4

# 3. exportar (verifica paridad contra torch en el mismo paso)
.venv\Scripts\python.exe toolkit\export_umx.py

# 4. separar
.venv\Scripts\python.exe toolkit\validate_ort.py --ep cpu
.venv\Scripts\python.exe toolkit\bench.py --ep cpu dml --torch
```

```python
from pathlib import Path
import soundfile as sf, numpy as np
from driver.pipeline import UmxDriver

mezcla, sr = sf.read("cancion.wav", dtype="float32", always_2d=True)
grafos = {n: Path(f"artifacts/umxhq_{n}.onnx") for n in ("vocals", "drums", "bass", "other")}
pistas = UmxDriver(grafos).separate(np.ascontiguousarray(mezcla.T, dtype=np.float64))
for nombre, audio in pistas.items():
    sf.write(f"{nombre}.wav", audio.T, sr)
```

## Licencias

- **Pesos**: MIT, declarado en el [record de Zenodo 3370489](https://zenodo.org/records/3370489)
  ("Open-Unmix-Pytorch UMX-HQ"). Por eso los grafos exportados se publican acá.
- **Código de Open-Unmix**: MIT ([sigsep/open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch)).
- **Este repo**: MIT.

`artifacts/manifest.json` guarda el SHA-256 del checkpoint del que salió cada grafo y el del
grafo, así que la procedencia se verifica en las dos direcciones.

## Créditos

Open-Unmix es de Fabian-Robert Stöter, Antoine Liutkus y Nobutaka Ito
([paper, JOSS 2019](https://doi.org/10.21105/joss.01667)). Este repo solo lo exporta y le
reimplementa las puntas; el modelo y su calidad son de ellos.
