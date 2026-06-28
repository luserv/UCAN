# UCAN — Unified Convolutional Attention Network
## Documentación para evaluación — Video 5 min

---

## 1. Modelo propuesto e implementación del módulo Transformer

### Arquitectura general

UCAN (**U**nified **C**onvolutional **A**ttention **N**etwork) es una red híbrida CNN-Transformer para super-resolución de imágenes ligera (_lightweight_). Combina atención por ventanas (Transformer), atención linear global y bloques convolucionales de kernel grande.

**Archivo principal:** `basicsr/archs/ucan_arch.py` (1497 líneas)

### Componentes Transformer implementados

| Componente | Clase | Línea | Descripción |
|---|---|---|---|
| **Window-MHSA** | `WindowsAttention` | 759 | Multi-head self-attention con ventana 16×16 y sesgo de posición relativa |
| **Shared Window-MHSA** | `SharedWindowsAttention` | 822 | Reusa mapa de atención de un RG anterior (ahorra cómputo) |
| **Dual Fusion Layer (SDFL)** | `SDFL` | 546 | Atención dual: rama espacial (Hedgehog) + rama canales (softmax) |
| **Dual Fusion Layer (DFRL)** | `DFRL` | 642 | Versión que reusa QK de un RG anterior |
| **High Performance Attention** | `WindowsAttention(flash=True)` | 1003 | Flash Attention con ventana 32×32 |
| **Rotary Position Embedding** | `RotaryEmbedding` | 42 | Codificación posicional rotatoria (RoPE) |
| **FourierEmbedding** | `FourierEmbedding` | 123 | RoPE + mezcla aprendible de frecuencias |

### Flujo completo (4 Residual Groups con atención semi-compartida)

```
Entrada → conv_first → patch_embed → tokens 1D

  RG1 [share='N'] → genera attn_map y qk_map NUEVOS
  RG2 [share='F'] → REUSA attn_map y qk_map de RG1
  RG3 [share='N'] → genera attn_map y qk_map NUEVOS
  RG4 [share='F'] → REUSA attn_map y qk_map de RG3

  → norm → patch_unembed → conv_after_body + residual global
  → UpsampleOneStep (PixelShuffle ×4) → salida SR
```

### HybridBlock (unidad base del Transformer)

Cada HybridBlock (`ucan_arch.py:854`) contiene:

1. **Parte 1 — Window-MHSA** (ventana 16×16): atención local por ventanas + MLP (SGFN)
2. **Parte 2 — Dual Fusion Layer**: 
   - *Spatial branch*: Hedgehog Attention — `ω(Q)ω(K)ᵀ V` con DWConv, evita colapso de rango
   - *Channel branch*: `softmax(QᵀK)V` — atención sobre canales
3. **Scale + residual connections** con `layer_scale = 1e-4`

### Hedgehog Feature Map

Usada en SDFL/DFRL para linear attention (`ucan_arch.py:15`). Implementa el mapeo:
```
ω(x) = [exp(Wx+b), exp(-Wx-b)]  → softmax
```
Entrenable, inicializado como identidad.

---

## 2. Configuración de hiperparámetros

### Archivos YAML de configuración

`options/Test/test_UCAN_x2.yml`, `test_UCAN_x3.yml`, `test_UCAN_x4.yml`

### Hiperparámetros de red (`network_g`)

| Parámetro | Valor | Línea en YAML | Descripción |
|---|---|---|---|
| `type` | `UCAN` | 56 | Arquitectura registrada en `ARCH_REGISTRY` |
| `upscale` | 2 / 3 / 4 | 57 | Factor de super-resolución |
| `in_chans` | 3 | 58 | Canales de entrada (RGB) |
| `img_size` | 64 | 59 | Tamaño de parche para patch_embed |
| `img_range` | 1.0 | 60 | Rango de imagen [0,1] |
| `window_size` | 16 | 61 | Tamaño de ventana de atención |
| `conv_depth` | 5 | 62 | Bloques LKSA + SGFN por BasicLayer |
| `share` | `['N','F','N','F']` | 63 | Esquema de compartición de atención |
| `embed_dim` | 48 | 64 | Dimensión del embedding (canales) |
| `mlp_ratio` | 1 | 65 | Factor de expansión del MLP |
| `mhsa_num_heads` | 2 | 66 | Cabezas en Window-MHSA |
| `dfl_num_heads` | 1 | 67 | Cabezas en Dual Fusion Layer |
| `use_checkpoint` | `False` | 68 | Gradient checkpointing |
| `upsampler` | `pixelshuffledirect` | 69 | Módulo de upsampling |
| `resi_connection` | `1conv` | 70 | Conexión residual (Conv1x1) |

### Hiperparámetros en el código (`ucan_arch.py:1248-1266`, clase `UCAN`)

```python
def __init__(self,
             img_size=64, patch_size=1, in_chans=3,
             embed_dim=96,        # default, pero en YAML se sobreescribe a 48
             drop_rate=0.,
             window_size=8,       # default, en YAML se sobreescribe a 16
             mlp_ratio=1.5,       # default, en YAML a 1
             conv_depth=5,
             share=None,          # default → ['N','F','N','F']
             mhsa_num_heads=2,
             dfl_num_heads=1,
             ...)
```

### Parámetros de evaluación (`val`)

| Parámetro | Valor | Descripción |
|---|---|---|
| `crop_border` | 2/3/4 (según scale) | Pixeles del borde a ignorar en métricas |
| `test_y_channel` | `True` | Evaluar en canal Y (YCbCr) |
| `save_img` | `True` | Guardar imágenes resultado |

---

## 3. Configuración de datos de entrenamiento y testing

### Datasets de testing

Definidos en `options/Test/test_UCAN_x4.yml:8-52`

| Dataset | GT (HR) | LQ (LR) | Template |
|---|---|---|---|
| **Set5** | `datasets/SR/Set5/HR` | `datasets/SR/Set5/LR_bicubic/X4` | `{}x4` |
| **Set14** | `datasets/SR/Set14/HR` | `datasets/SR/Set14/LR_bicubic/X4` | `{}x4` |
| **B100** | `datasets/SR/B100/HR` | `datasets/SR/B100/LR_bicubic/X4` | `{}x4` |
| **Urban100** | `datasets/SR/Urban100/HR` | `datasets/SR/Urban100/LR_bicubic/X4` | `{}x4` |

### Descarga de datasets

`download_datasets.py` — descarga automática desde Hugging Face:
```bash
python download_datasets.py
```
Crea la estructura: `datasets/SR/{Set5,Set14,B100,Urban100}/HR/` y `LR_bicubic/X4/`.

### PairedImageDataset (`basicsr/data/paired_image_dataset.py:13`)

- Lee pares de imágenes LR (baja resolución) y GT (alta resolución)
- Durante **train**: random crop, flip, rotación
- Durante **test**: solo carga y convierte a tensor (BGR→RGB, HWC→CHW, normalización [0,1])
- Soporta backend: `disk`, `lmdb`, `meta_info_file`

### Pipeline de test (`basicsr/test.py`)

```bash
CUDA_VISIBLE_DEVICES="" python basicsr/test.py -opt options/Test/test_UCAN_x4.yml
```

1. Parsea `test_UCAN_x4.yml`
2. Construye dataset y dataloader para cada benchmark
3. Construye el modelo UCAN con pesos preentrenados
4. Ejecuta validación: inferencia → calcula PSNR/SSIM → guarda imágenes

### Pipeline de inferencia rápida (`infer.py`)

```bash
CUDA_VISIBLE_DEVICES="" python infer.py imagen_lr.jpg resultado.png
```

---

## 4. Información adicional del paper y adecuación a CPU

### Información del paper

- **Título completo:** _UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution_
- **Autores originales:** Hokiyoshi (repositorio GitHub: `hokiyoshi/UCAN`)
- **Innovaciones clave:**
  - **Atención semi-compartida**: 4 Residual Groups donde RG2 reusa atención de RG1, y RG4 reusa de RG3. Reduce cómputo sin perder calidad.
  - **Hedgehog Attention**: Mapeo de features lineal que evita colapso de rango en atención linear.
  - **Large Kernel Distillation (LKD)**: Divide canales en finos (1/4) y gruesos (3/4 bypass). Solo procesa finos con convoluciones dilatadas 23×23.
  - **Dual Fusion Layer**: Combina atención espacial (Hedgehog) + atención de canales en cada HybridBlock.
  - **FourierEmbedding**: RoPE con coeficientes de Fourier aprendibles.

### Adecuación para CPU (trabajo realizado)

#### a) Forzar device CPU en inferencia

**`infer.py:14`** — Dispositivo forzado a CPU:
```python
device = torch.device("cpu")
```

**`infer.py:30`** — Carga de pesos con `map_location=device`:
```python
checkpoint = torch.load("experiments/pretrained_models/weight_final_x4.pth", map_location=device)
```

#### b) Instalación con PyTorch CPU

**`README_CPU.md`** — Guía completa paso a paso:
- Entorno Conda con Python 3.10
- `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- Dependencias: numpy, opencv-python, Pillow, pyyaml, scipy, scikit-image, tqdm, lmdb, tensorboard, timm, einops, PyWavelets, matplotlib
- Instalación como paquete editable: `pip install -e .`

#### c) Evaluación y ejecución

```bash
# Evaluar en datasets benchmark (CPU)
CUDA_VISIBLE_DEVICES="" python basicsr/test.py -opt options/Test/test_UCAN_x4.yml

# Inferencia con imagen propia
CUDA_VISIBLE_DEVICES="" python infer.py imagen_lr.jpg resultado.png
```

`CUDA_VISIBLE_DEVICES=""` deshabilita GPU aunque esté disponible.

#### d) Descarga de pesos preentrenados

```bash
mkdir -p experiments/pretrained_models
curl -L -o experiments/pretrained_models/weight_final_x4.pth \
  "https://github.com/hokiyoshi/UCAN/releases/download/1.0/weight_final_x4.pth"
```

#### e) Parámetros de evaluación (PSNR / SSIM)

| Dataset | PSNR (dB) | SSIM |
|---|---|---|
| Set5 | — | — |
| Set14 | — | — |
| B100 | — | — |
| Urban100 | — | — |

*(Completar con valores obtenidos tras ejecutar `test_UCAN_x4.yml`)*

---

## Referencias rápidas para el video

| Tópico | Archivo | Líneas |
|---|---|---|
| Clase UCAN (modelo completo) | `basicsr/archs/ucan_arch.py` | 1224-1497 |
| WindowsAttention (Transformer) | `basicsr/archs/ucan_arch.py` | 759-820 |
| Dual Fusion Layer (SDFL) | `basicsr/archs/ucan_arch.py` | 546-641 |
| Hedgehog Feature Map | `basicsr/archs/ucan_arch.py` | 15-40 |
| HybridBlock (unidad base) | `basicsr/archs/ucan_arch.py` | 854-968 |
| Hiperparámetros | `options/Test/test_UCAN_x4.yml` | 55-70 |
| Datasets de test | `options/Test/test_UCAN_x4.yml` | 8-52 |
| PairedImageDataset | `basicsr/data/paired_image_dataset.py` | 13-113 |
| Inferencia CPU | `infer.py` | 14, 30 |
| Instalación CPU | `README_CPU.md` | 1-87 |
| Descarga datasets | `download_datasets.py` | 1-45 |
| Modelo + test loop | `basicsr/models/ucan_model.py` | 15-118 |
| Pipeline de test | `basicsr/test.py` | 11-44 |
