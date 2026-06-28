# UCAN — Unified Convolutional Attention Network
## Documentación para evaluación — Video 5 min

---

## 1. Modelo propuesto e implementación del módulo Transformer

### Arquitectura general

UCAN (**U**nified **C**onvolutional **A**ttention **N**etwork) es una red híbrida CNN (**C**onvolutional **N**eural **N**etwork)-Transformer para super-resolución de imágenes ligera (_lightweight_). Combina atención por ventanas (Transformer), atención linear global y bloques convolucionales de kernel grande.

**Archivo principal:** `basicsr/archs/ucan_arch.py` (1497 líneas)

### Componentes Transformer implementados

| Componente | Clase | Línea | Descripción |
|---|---|---|---|
| **W-MHSA (Window Multi-Head Self-Attention)** | `WindowsAttention` | 759 | Atención propia multi-cabeza con ventana 16×16 y sesgo de posición relativa |
| **Shared W-MHSA** | `SharedWindowsAttention` | 822 | Reusa mapa de atención de un RG anterior (ahorra cómputo) |
| **SDFL (Self Dual Fusion Layer)** | `SDFL` | 546 | Atención dual: rama espacial (Hedgehog) + rama canales (softmax) |
| **DFRL (Dual Fusion Reuse Layer)** | `DFRL` | 642 | Versión que reusa QK de un RG anterior |
| **HPA (High Performance Attention)** | `WindowsAttention(flash=True)` | 1003 | Flash Attention con ventana 32×32 |
| **RoPE (Rotary Position Embedding)** | `RotaryEmbedding` | 42 | Codificación posicional rotatoria |
| **FourierEmbedding** | `FourierEmbedding` | 123 | RoPE + mezcla aprendible de frecuencias |

### Flujo completo (4 Residual Groups con atención semi-compartida)

```
Entrada → conv_first → patch_embed → tokens 1D

  RG (Residual Group) 1 [share='N'] → genera attn_map y qk_map NUEVOS
  RG2 [share='F'] → REUSA attn_map y qk_map de RG1
  RG3 [share='N'] → genera attn_map y qk_map NUEVOS
  RG4 [share='F'] → REUSA attn_map y qk_map de RG3

  → norm → patch_unembed → conv_after_body + residual global
  → UpsampleOneStep (PixelShuffle ×4) → salida SR (Super-Resolución)
```

### HybridBlock (unidad base del Transformer)

Cada HybridBlock (`ucan_arch.py:854`) contiene:

1. **Parte 1 — W-MHSA (Window Multi-Head Self-Attention)** (ventana 16×16): atención local por ventanas + MLP (Multi-Layer Perceptron) implementado como **SGFN (Spatial-Gate Feedforward Network)**
2. **Parte 2 — Dual Fusion Layer**: 
   - *Rama espacial*: Hedgehog Attention — `ω(Q)ω(K)ᵀ V` con **DWConv (Depthwise Convolution)**, evita colapso de rango
   - *Rama de canales*: `softmax(QᵀK)V` — atención sobre canales
3. **Scale + residual connections** con `layer_scale = 1e-4`

### ¿Cómo señalar la implementación del Transformer en el código?

Para demostrar al ingeniero dónde está implementado el Transformer, muestra estos 3 puntos en `basicsr/archs/ucan_arch.py`:

**1. `WindowsAttention` (línea 759)** — El núcleo Transformer: QKV projection + attention dentro de ventanas:
```python
class WindowsAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, ...):
        self.to_qkv = nn.Linear(dim, dim * 3)  # genera Q, K, V
        ...
    def forward(self, x, mask=None):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # scaled dot-product attention dentro de cada ventana 16×16
        attn = (q @ k.transpose(-2, -1)) + relative_position_bias
        attn = self.softmax(attn)
        x = (attn @ v)
```
Sigue la fórmula estándar: `Attention(Q,K,V) = softmax(QK^T/√d + bias) V`

**QKV** son las 3 matrices del mecanismo de atención:
- **Q (Query / Consulta):** representa lo que cada token "pregunta"
- **K (Key / Clave):** representa lo que cada token "ofrece" como relevancia
- **V (Value / Valor):** representa la información que cada token aporta

La atención calcula: qué tanto coincide cada Query con cada Key (producto punto), usa softmax para convertir eso en pesos, y pondera los Values con esos pesos.

**MLP (Multi-Layer Perceptron):** red feed-forward de 2 capas que sigue a la atención, implementado en UCAN como `SGFN` (`Spatial-Gate Feed-Forward Network`, línea 435). Su estructura:
```python
fc1 → Linear(in_features, hidden_features)    # expande
sg  → SpatialGate (conv 7×7 depthwise + GELU) # gate espacial
fc2 → Linear(hidden_features//2, out_features) # comprime
```

**2. `HybridBlock` (línea 854)** — Sigue la estructura clásica Transformer → Atención + MLP + residual:
```
                    entrada
                       │
                  ┌────┴────┐
                  │  Norm1  │ ← LayerNorm
                  │ W-MHSA  │ ← Window Multi-Head Self-Attention
                  │  + MLP  │ ← SGFN (Spatial-Gate FFN)
                  └────┬────┘
                  scale + residual
                       │
                  ┌────┴────┐
                  │ L_attn  │ ← Dual Fusion Layer (atención global)
                  └────┬────┘
                  Norm + residual + scale
                       │
                    salida
```

**3. `BasicLayer` (línea 972)** — La capa Transformer completa que orquesta:
- `attn1`: **HPA (High Performance Attention)** con Flash Attention (ventana 32×32)
- `self.blocks`: 4 HybridBlocks (**W-MHSA** + **Dual Fusion** + **MLP**)
- `mbconv`: **MBConv (Mobile Block Convolution)** entre bloques, Conv1×1 → DWConv 3×3 → Conv1×1

**En tu video di:** *"El módulo Transformer se implementa en 3 niveles: `WindowsAttention` hace la atención por ventanas (QKV → softmax → V), `HybridBlock` compone atención + MLP con residual como un bloque Transformer estándar, y `BasicLayer` apila 4 HybridBlocks más atención global Flash."*

### Hedgehog Feature Map

Usada en **SDFL (Self Dual Fusion Layer)** / **DFRL (Dual Fusion Reuse Layer)** para atención lineal (`ucan_arch.py:15`). Implementa el mapeo:
```
ω(x) = [exp(Wx+b), exp(-Wx-b)]  → softmax
```
Entrenable, inicializado como identidad.

---

## 2. Configuración de hiperparámetros

### Archivos YAML (YAML Ain't Markup Language) de configuración

`options/Test/test_UCAN_x2.yml`, `test_UCAN_x3.yml`, `test_UCAN_x4.yml`

### ¿Dónde están las configuraciones de los hiperparámetros?

Hay **2 lugares** donde se definen los hiperparámetros:

**A) Archivos YAML (sobreescriben defaults) → `options/Test/test_UCAN_x4.yml:55-70`**

```yaml
network_g:
  type: UCAN
  upscale: 4           # factor de super-resolución
  window_size: 16      # tamaño de ventana de atención
  embed_dim: 48        # canales del embedding
  mlp_ratio: 1         # factor de expansión MLP
  mhsa_num_heads: 2    # cabezas de atención
  dfl_num_heads: 1     # cabezas del Dual Fusion Layer
  conv_depth: 5        # bloques convolucionales por grupo
  share: ['N','F','N','F']  # esquema de compartición
```

Estos valores se cargan en `basicsr/utils/options.py` y se pasan al constructor de `UCAN`.

**B) Constructor de la clase UCAN (defaults) → `basicsr/archs/ucan_arch.py:1248-1266`**

```python
class UCAN(nn.Module):
    def __init__(self,
                 img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96,        # ← default, YAML lo cambia a 48
                 drop_rate=0.,
                 window_size=8,       # ← default, YAML lo cambia a 16
                 mlp_ratio=1.5,       # ← default, YAML lo cambia a 1
                 conv_depth=5,
                 share=None,          # → ['N','F','N','F']
                 mhsa_num_heads=2,
                 dfl_num_heads=1,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 resi_connection='1conv')
```

**En tu video di:** *"Los hiperparámetros se configuran en `options/Test/test_UCAN_x4.yml` líneas 55 a 70, donde se definen `window_size=16`, `embed_dim=48`, `mlp_ratio=1`, etc. Además, la clase `UCAN` en `ucan_arch.py` línea 1248 tiene los valores por defecto que el YAML sobreescribe."*

### Hiperparámetros de red (`network_g`)

| Parámetro | Valor | Línea en YAML | Descripción |
|---|---|---|---|
| `type` | `UCAN` | 56 | Arquitectura registrada en `ARCH_REGISTRY` |
| `upscale` | 2 / 3 / 4 | 57 | Factor de super-resolución |
| `in_chans` | 3 | 58 | Canales de entrada (RGB) |
| `img_size` | 64 | 59 | Tamaño de parche para patch_embed |
| `img_range` | 1.0 | 60 | Rango de imagen [0,1] |
| `window_size` | 16 | 61 | Tamaño de ventana de atención |
| `conv_depth` | 5 | 62 | Bloques **LKSA (Large-Kernel Spatial Attention)** + **SGFN** por BasicLayer |
| `share` | `['N','F','N','F']` | 63 | Esquema de compartición de atención |
| `embed_dim` | 48 | 64 | Dimensión del embedding (canales) |
| `mlp_ratio` | 1 | 65 | Factor de expansión del MLP |
| `mhsa_num_heads` | 2 | 66 | Cabezas en Window-MHSA |
| `dfl_num_heads` | 1 | 67 | Cabezas en **DFL (Dual Fusion Layer)** |
| `use_checkpoint` | `False` | 68 | Gradient checkpointing |
| `upsampler` | `pixelshuffledirect` | 69 | Módulo de upsampling |
| `resi_connection` | `1conv` | 70 | Conexión residual (**ESA**: Enhanced Spatial Attention + Conv1x1) |

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
| `test_y_channel` | `True` | Evaluar en canal Y (espacio YCbCr — luminancia) |
| `save_img` | `True` | Guardar imágenes resultado |

---

## 3. Configuración de datos de entrenamiento y testing

### Datasets de testing

Definidos en `options/Test/test_UCAN_x4.yml:8-52`

| Dataset | **GT (Ground Truth = verdad absoluta) / HR (High Resolution = alta resolución)** | **LQ (Low Quality = baja calidad) / LR (Low Resolution = baja resolución)** | Template |
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

- Lee pares de imágenes **LR (Low Resolution)** y **GT (Ground Truth / alta resolución)**
- Durante **train**: random crop, flip, rotación
- Durante **test**: solo carga y convierte a tensor (**BGR→RGB**, **HWC→CHW** — reordena dimensiones, normalización [0,1])
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
  - **Atención semi-compartida**: 4 **RG (Residual Groups)** donde RG2 reusa atención de RG1, y RG4 reusa de RG3. Reduce cómputo sin perder calidad.
  - **Hedgehog Attention**: Mapeo de features lineal que evita colapso de rango en atención lineal.
  - **LKD (Large Kernel Distillation)**: Divide canales en finos (1/4) y gruesos (3/4 bypass). Solo procesa finos con convoluciones dilatadas 23×23.
  - **Dual Fusion Layer**: Combina atención espacial (Hedgehog) + atención de canales en cada HybridBlock.
  - **FourierEmbedding**: **RoPE (Rotary Position Embedding)** con coeficientes de Fourier aprendibles.

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

**PSNR (Peak Signal-to-Noise Ratio)** — relación señal-ruido máxima en dB.  
**SSIM (Structural Similarity Index Measure)** — índice de similitud estructural.

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
