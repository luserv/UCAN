# UCAN — CPU + Conda

## Requisitos

- Conda (Miniforge3, Miniconda o Anaconda)
- CPU (no requiere GPU)
- ~500 MB de espacio

## Instalación

```bash
# 1. Crear y activar entorno
conda create -n UCAN python=3.10 -y
conda activate UCAN

# 2. Verificar que estás usando el pip del entorno
which pip
# Debe mostrar: ~/miniforge3/envs/UCAN/bin/pip
# Si muestra /usr/bin/pip, el entorno NO está activo

# 3. Instalar PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar el resto de dependencias
pip install numpy opencv-python Pillow pyyaml scipy scikit-image tqdm lmdb tensorboard timm einops PyWavelets matplotlib

# 5. Instalar UCAN como paquete editable
pip install -e .
```

## Descargar pesos preentrenados

```bash
mkdir -p experiments/pretrained_models

# Peso para upscale x2
curl -L -o experiments/pretrained_models/weight_final_x2.pth \
  "https://github.com/hokiyoshi/UCAN/releases/download/1.0/weight_final_x2.pth"

# Peso para upscale x3
curl -L -o experiments/pretrained_models/weight_final_x3.pth \
  "https://github.com/hokiyoshi/UCAN/releases/download/1.0/weight_final_x3.pth"

# Peso para upscale x4
curl -L -o experiments/pretrained_models/weight_final_x4.pth \
  "https://github.com/hokiyoshi/UCAN/releases/download/1.0/weight_final_x4.pth"
```

## Descargar datasets de prueba

```bash
# Set5, Set14, BSD100, Urban100 desde Hugging Face
python download_datasets.py
```

Esto crea la estructura `datasets/SR/{Set5,Set14,B100,Urban100}/HR/` y `LR_bicubic/X4/`.

> **Manga109**: requiere registro en http://www.manga109.org/ — no se descarga automáticamente.

## Evaluar en datasets benchmark

```bash
CUDA_VISIBLE_DEVICES="" python basicsr/test.py -opt options/Test/test_UCAN_x4.yml
```

> `CUDA_VISIBLE_DEVICES=""` fuerza el uso de CPU aunque tengas GPU.

## Inferencia con imagen propia

Usa el script `infer.py`:

```bash
CUDA_VISIBLE_DEVICES="" python infer.py ruta/de/mi_imagen_lr.jpg resultado.png
```

- La imagen de entrada debe ser **baja resolución** (LR)
- La imagen de salida será upscaleda 4x
- Formatos soportados: PNG, JPG, etc.

## Verificar entorno

```bash
conda activate UCAN
which pip        # Debe mostrar el path del entorno
python --version # Debe mostrar 3.10.x
python -c "import torch; print('OK - torch', torch.__version__)"
```
