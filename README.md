# UCAN — Unified Convolutional Attention Network

## Flujo de ejecución paso a paso

### 1. Preprocesamiento (`infer.py`)

```
img (HxWx3, uint8, BGR)          ej: 964×1280
  → img2tensor() → RGB, CHW, float32 [0, 255]
  → /255.0 → rango [0, 1]
  → unsqueeze(0) → (1, 3, 964, 1280)
```

### 2. `UCAN.forward()`

```
(A) Padding reflectivo a múltiplo de window_size=16
    964→976, 1280→1280  →  (1, 3, 976, 1280)

(B) Restar media RGB (0.4488, 0.4371, 0.4040) × img_range (1.0)
    x = (x - mean) × 1.0   → rango ≈ [-0.45, 0.55]

(C) conv_first: Conv2d(3 → 48, kernel=3)
    → (1, 48, 976, 1280)
```

### 3. `forward_features()`

```
patch_embed: (1, 48, 976, 1280) → (1, 1249280, 48)   [tokens 1D]
                                         ↑ 976×1280 = 1,249,280 tokens

RG1 (share='N')  → genera attn_map y qk_map nuevos
RG2 (share='F')  → REUSA attn_map y qk_map de RG1
RG3 (share='N')  → genera attn_map y qk_map nuevos
RG4 (share='F')  → REUSA attn_map y qk_map de RG3
```

### 4. BasicLayer.forward() — dentro de cada ResidualGroup

```
(1) High Performance Attention (HPA):
    - ConvMLP → extrae contexto local
    - Flash Attention con ventana 32×32 → atención O(n) en lugar de O(n²)

(2) 4× HybridBlocks (atención semi-compartida):

    ┌─ PARTE 1: Window-MHSA (ventana 16×16)
    │   - share='N': calcula Q, K, V y atención normalmente
    │   - share='F': REUSA attention map del RG anterior
    │   - MLP (SGFN) + residual + scale
    │
    └─ PARTE 2: Dual Fusion Layer (atención global)
        - Spatial branch: Hedgehog Attention
            ω(Q)ω(K)ᵀ V + DWConv
            ω = [exp(Wx+b), exp(-Wx-b)]  → evita colapso de rango
        - Channel branch: softmax(QᵀK) V
        - Concatena ambas → fusión local + global
        - share='F': REUSA QK del RG anterior

(3) Large Kernel Distillation (LKD):
    - Divide canales: finos (C/4) + gruesos (3C/4, bypass)
    - Solo en canales finos: Triple Feature Extraction
      · Rama channel attention
      · Rama local 1×1 → 3×3 → 1×1
      · Rama large kernel (conv dilatada 23×23 depthwise)

(4) ESA (Enhanced Spatial Attention) + Conv1×1 + residual
```

### 5. Post-procesamiento del modelo

```
norm (LayerNorm)
patch_unembed: (1, 1249280, 48) → (1, 48, 976, 1280)

conv_after_body: Conv2d(48 → 48, k=3)

  RESIDUAL GLOBAL:
    salida = conv_after_body(features) + conv_first(x)
    (conexión residual global estilo SwinIR)

UpsampleOneStep: Conv2d(48 → 192, k=3) + PixelShuffle(×4)
                 → (1, 3, 3904, 5120)

x = x / img_range (1.0) + mean  → restaura rango [0, 1]

Crop: quita padding → (1, 3, 3856, 5120)
```

### 6. Postprocesamiento (`infer.py`)

```
output tensor [0, 1] → tensor2img()
  → clamp [0, 1], ×255, uint8, RGB → BGR
  → cv2.imwrite → resultado.png (3856×5120)
```

---

## Diagrama del flujo de datos

```
imagen LR (964×1280)
  │
  ├─ padding + restar mean
  │
  ├─ conv_first (Conv2d 3→48)
  │
  ├─ patch_embed → tokens 1D (1, 1249280, 48)
  │
  ├─ RG1 [share=N]  → HPA + 4×[WMHSA + HgA+CA] + LKD
  │    │              genera attn_map, qk_map
  │    │
  ├─ RG2 [share=F]  → HPA + 4×[WMHSA + HgA+CA] + LKD
  │    │              REUSA attn_map, qk_map de RG1
  │    │
  ├─ RG3 [share=N]  → HPA + 4×[WMHSA + HgA+CA] + LKD
  │    │              genera attn_map, qk_map
  │    │
  ├─ RG4 [share=F]  → HPA + 4×[WMHSA + HgA+CA] + LKD
  │    │              REUSA attn_map, qk_map de RG3
  │    │
  ├─ norm + patch_unembed → (1, 48, 976, 1280)
  │
  ├─ conv_after_body + residual global
  │
  ├─ UpsampleOneStep (PixelShuffle ×4) → (1, 3, 3904, 5120)
  │
  ├─ sumar mean + crop → (1, 3, 3856, 5120)
  │
  └─ tensor2img → resultado.png
```

## Arquitectura del HybridBlock (unidad base)

```
                    entrada (tokens 1D)
                         │
                    ┌────┴────┐
                    │  Norm1  │
                    │ WMHSA   │ ← ventana 16×16 (o reusada si share=F)
                    │  +MLP   │
                    └────┬────┘
                    scale + residual
                         │
                    ┌────┴────┐
                    │ L_attn  │ ← Dual Fusion Layer
                    │  ┌──────┤    · Spatial: Hedgehog Attention
                    │  │ HgA  │    · Channel: CA
                    │  │ CA   │    · Concat
                    │  └──────┤
                    └────┬────┘
                    Norm + residual + scale
                         │
                      salida
```
