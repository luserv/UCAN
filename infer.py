import argparse
import cv2
import torch
import numpy as np
from basicsr.archs import build_network
from basicsr.utils import img2tensor, tensor2img

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Ruta de la imagen de entrada (LR)")
parser.add_argument("output", nargs="?", default="output.png", help="Ruta de salida")
args = parser.parse_args()

device = torch.device("cpu")

opt_arch = {
    "type": "UCAN", "upscale": 4, "in_chans": 3, "img_size": 64,
    "img_range": 1., "window_size": 16, "conv_depth": 5,
    "share": ["N", "F", "N", "F"], "embed_dim": 48,
    "mlp_ratio": 1, "mhsa_num_heads": 2, "dfl_num_heads": 1,
    "use_checkpoint": False, "upsampler": "pixelshuffledirect",
    "resi_connection": "1conv",
}

net = build_network(opt_arch)
checkpoint = torch.load("experiments/pretrained_models/weight_final_x4.pth", map_location=device)
net.load_state_dict(checkpoint["params_ema"], strict=True)
net.eval().to(device)

img = cv2.imread(args.input)
img_t = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(0).to(device)

with torch.no_grad():
    output = net(img_t)

out_img = tensor2img(output)
cv2.imwrite(args.output, out_img)
print(f"Listo: {args.output}")
