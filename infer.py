import argparse
import cv2
import torch
import numpy as np
from basicsr.archs import build_network    # construye la arquitectura desde un dict
from basicsr.utils import img2tensor, tensor2img  # conversiones imagen <-> tensor

# --- Argumentos CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Ruta de la imagen de entrada (LR)")
parser.add_argument("output", nargs="?", default="output.png", help="Ruta de salida")
args = parser.parse_args()

device = torch.device("cpu")

# --- Configuración de la arquitectura UCAN (debe coincidir con el peso) ---
opt_arch = {
    "type": "UCAN", "upscale": 4, "in_chans": 3, "img_size": 64,
    "img_range": 1., "window_size": 16, "conv_depth": 5,
    "share": ["N", "F", "N", "F"], "embed_dim": 48,
    "mlp_ratio": 1, "mhsa_num_heads": 2, "dfl_num_heads": 1,
    "use_checkpoint": False, "upsampler": "pixelshuffledirect",
    "resi_connection": "1conv",
}

# 1. Crear la red UCAN (solo la arquitectura, sin pesos)
net = build_network(opt_arch)

# 2. Cargar los pesos preentrenados (params_ema = exponential moving average)
checkpoint = torch.load("experiments/pretrained_models/weight_final_x4.pth", map_location=device)
net.load_state_dict(checkpoint["params_ema"], strict=True)

# 3. Poner en modo evaluación y mover a CPU
net.eval().to(device)

# 4. Cargar imagen de entrada con OpenCV (BGR, HWC, uint8)
img = cv2.imread(args.input)

# 5. Convertir img a tensor (BGR->RGB, HWC->CHW, uint8->float32) y añadir batch dim
img_t = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)
img_t = img_t / 255.0  # normalizar a [0, 1]

# 6. Inferencia (sin gradientes para ahorrar memoria)
with torch.no_grad():
    output = net(img_t)

# 7. Convertir tensor de salida de vuelta a imagen (RGB->BGR, CHW->HWC)
out_img = tensor2img(output)

# 8. Guardar resultado
cv2.imwrite(args.output, out_img)
print(f"Listo: {args.output}")
