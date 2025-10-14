# Minecraft 3D Asset Generator - Architektur-Diagramm

## System-Übersicht

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Minecraft 3D Asset Generator                      │
│                                                                       │
│  Input: "big birch tree with fall leaves"                           │
│                              ↓                                        │
│  Output: generated.schem (16×32×16 Voxel Structure)                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Training Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. DATA PREPROCESSING                                                 │
└──────────────────────────────────────────────────────────────────────┘

out/
├── big_birch_wood_birch_leaves_fall_09_05--uuid.schem
├── beehive_oak_planks_04_01.schem
└── ... (9180 files)
                    ↓
        ┌───────────────────────┐
        │  Filename Parser      │
        │  - Extract tags       │
        │  - Remove UUID        │
        │  - Classify size      │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Schematic Parser     │
        │  - Load NBT           │
        │  - Decode BlockData   │
        │  - Convert to Voxels  │
        └───────────────────────┘
                    ↓
data/processed/
├── big_birch_tree_001.npy  (16×32×16 array)
├── beehive_oak_001.npy     (16×16×16 array)
└── ...
                    +
data/splits/
├── train_metadata.json (7344 samples)
├── val_metadata.json   (918 samples)
└── test_metadata.json  (918 samples)

┌──────────────────────────────────────────────────────────────────────┐
│ 2. MODEL ARCHITECTURE                                                 │
└──────────────────────────────────────────────────────────────────────┘

INPUT
─────
┌──────────────────┐         ┌──────────────────┐
│ Voxel Array      │         │ Text Prompt      │
│ (B,26,16,32,16)  │         │ "big birch tree" │
│ [One-Hot Encoded]│         └──────────────────┘
└──────────────────┘                   ↓
         ↓                   ┌──────────────────┐
         ↓                   │ Sentence-BERT    │
         ↓                   │ (all-MiniLM-L6)  │
         ↓                   └──────────────────┘
         ↓                             ↓
         ↓                   ┌──────────────────┐
         ↓                   │ Text Embedding   │
         ↓                   │ (B, 384)         │
         ↓                   └──────────────────┘
         ↓                             ↓
         ↓                             ↓ (conditioning)
ENCODER                                ↓
───────                                ↓
┌──────────────────┐                   ↓
│ 3D Conv Block 1  │                   ↓
│ 26 → 64 channels │                   ↓
│ 16×32×16         │                   ↓
└──────────────────┘                   ↓
         ↓                             ↓
┌──────────────────┐                   ↓
│ 3D Conv Block 2  │                   ↓
│ 64 → 128 channels│                   ↓
│ 8×16×8           │                   ↓
└──────────────────┘                   ↓
         ↓                             ↓
┌──────────────────┐                   ↓
│ 3D Conv Block 3  │                   ↓
│ 128 → 256        │                   ↓
│ 4×8×4            │                   ↓
└──────────────────┘                   ↓
         ↓                             ↓
┌──────────────────┐                   ↓
│ 3D Conv Block 4  │                   ↓
│ 256 → 512        │                   ↓
│ 2×4×2            │                   ↓
└──────────────────┘                   ↓
         ↓                             ↓
┌──────────────────┐                   ↓
│ Flatten          │                   ↓
│ → 8192           │                   ↓
└──────────────────┘                   ↓
         ↓                             ↓
    ┌────┴────┐                        ↓
    ↓         ↓                        ↓
┌───────┐ ┌─────────┐                 ↓
│  μ    │ │ log σ²  │                 ↓
│ (192) │ │  (192)  │                 ↓
└───────┘ └─────────┘                 ↓
    └────┬────┘                        ↓
         ↓                             ↓
LATENT SPACE                           ↓
────────────                           ↓
┌──────────────────┐                   ↓
│ Reparameterize   │                   ↓
│ z = μ + ε·σ      │                   ↓
│ z ~ N(μ, σ²)     │                   ↓
└──────────────────┘                   ↓
         ↓                             ↓
    ┌────┴────┐                        ↓
    │    z    │◄───────────────────────┘
    │  (192)  │        (concat)
    └─────────┘
         ↓
DECODER (with FiLM Conditioning)
───────
┌──────────────────┐
│ FC Layer         │
│ 192+384 → 8192   │
└──────────────────┘
         ↓
┌──────────────────┐
│ Reshape          │
│ → (512,2,4,2)    │
└──────────────────┘
         ↓
┌──────────────────┐  ┌──────────────┐
│ TransConv3D + FiLM│◄─│ Text (384)   │
│ 512 → 256        │  │ γ, β scaling │
│ 4×8×4            │  └──────────────┘
└──────────────────┘
         ↓
┌──────────────────┐  ┌──────────────┐
│ TransConv3D + FiLM│◄─│ Text (384)   │
│ 256 → 128        │  │ γ, β scaling │
│ 8×16×8           │  └──────────────┘
└──────────────────┘
         ↓
┌──────────────────┐  ┌──────────────┐
│ TransConv3D + FiLM│◄─│ Text (384)   │
│ 128 → 64         │  │ γ, β scaling │
│ 16×32×16         │  └──────────────┘
└──────────────────┘
         ↓
┌──────────────────┐
│ Output Conv3D    │
│ 64 → 26 channels │
│ 16×32×16         │
└──────────────────┘
         ↓
OUTPUT
──────
┌──────────────────┐
│ Block Logits     │
│ (B,26,16,32,16)  │
└──────────────────┘
         ↓
    [Argmax]
         ↓
┌──────────────────┐
│ Reconstructed    │
│ Voxel Array      │
│ (B,16,32,16)     │
└──────────────────┘

LOSS CALCULATION
────────────────
┌──────────────────────────────────┐
│ Total Loss = Recon + β·KL        │
│                                   │
│ Recon Loss:                      │
│   Cross-Entropy(pred, target)    │
│                                   │
│ KL Loss:                         │
│   -0.5·Σ(1 + log(σ²) - μ² - σ²) │
│                                   │
│ β = 0.5 (with annealing)         │
└──────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ 3. TRAINING                                                           │
└──────────────────────────────────────────────────────────────────────┘

for epoch in range(200):
    for batch in train_loader:
        ┌─────────────────┐
        │ Forward Pass    │
        │ - Encode        │
        │ - Sample z      │
        │ - Decode        │
        └─────────────────┘
                ↓
        ┌─────────────────┐
        │ Calculate Loss  │
        │ - Reconstruction│
        │ - KL Divergence │
        └─────────────────┘
                ↓
        ┌─────────────────┐
        │ Backpropagation │
        │ - Update weights│
        └─────────────────┘
                ↓
        ┌─────────────────┐
        │ Log Metrics     │
        │ - TensorBoard   │
        └─────────────────┘
    
    ┌─────────────────┐
    │ Validation      │
    │ - Eval metrics  │
    │ - Save best     │
    └─────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ 4. INFERENCE                                                          │
└──────────────────────────────────────────────────────────────────────┘

User Input: "big oak tree"
         ↓
┌──────────────────┐
│ Parse Size Tag   │
│ "big" → big      │
│ (16×32×16)       │
└──────────────────┘
         ↓
┌──────────────────┐
│ Encode Text      │
│ Sentence-BERT    │
│ → (1, 384)       │
└──────────────────┘
         ↓
┌──────────────────┐
│ Sample z         │
│ z ~ N(0,I)·temp  │
│ → (1, 192)       │
└──────────────────┘
         ↓
┌──────────────────┐
│ Decode           │
│ z + text_emb     │
│ → logits         │
└──────────────────┘
         ↓
┌──────────────────┐
│ Argmax           │
│ → Block IDs      │
│ (16×32×16)       │
└──────────────────┘
         ↓
┌──────────────────┐
│ Create .schem    │
│ - Encode varint  │
│ - Build NBT      │
│ - Save file      │
└──────────────────┘
         ↓
Output: generated.schem
```

## FiLM Conditioning Detail

```
┌─────────────────────────────────────────────────────────────┐
│ Feature-wise Linear Modulation (FiLM)                       │
└─────────────────────────────────────────────────────────────┘

Input Features: x (B, C, D, H, W)
Text Embedding: t (B, 384)
         ↓
    ┌────┴────┐
    ↓         ↓
┌────────┐ ┌────────┐
│ Linear │ │ Linear │
│ 384→C  │ │ 384→C  │
│  (γ)   │ │  (β)   │
└────────┘ └────────┘
    ↓         ↓
    │         │
    └────┬────┘
         ↓
    x' = γ·x + β

[Modulates each channel based on text]
```

## Size Categories

```
┌─────────────────────────────────────────────────────────┐
│ SIZE CATEGORIES                                          │
└─────────────────────────────────────────────────────────┘

Normal (16×16×16)     Big (16×32×16)      Huge (24×64×24)
─────────────────     ──────────────      ───────────────
Latent: 128D          Latent: 192D        Latent: 256D

Examples:             Examples:            Examples:
- Small trees         - Normal trees       - Giant trees
- Bushes             - Big bushes         - Massive oaks
- Bee nests          - Structures         - Landmarks

┌────────┐           ┌────────┐           ┌──────────┐
│        │           │        │           │          │
│  16×16 │           │  16×32 │           │  24×64   │
│        │           │        │           │          │
└────────┘           └────────┘           └──────────┘
```

## Training Metrics Flow

```
┌─────────────────────────────────────────────────────────┐
│ METRICS TRACKING                                         │
└─────────────────────────────────────────────────────────┘

Training Step
├── Loss (Total)
├── Reconstruction Loss
├── KL Loss
├── Accuracy (Overall)
└── Non-Air Accuracy
       ↓
   [Logged to TensorBoard]
       ↓
Validation Step
├── Val Loss
├── Val Reconstruction Loss
├── Val KL Loss
├── Val Accuracy
└── Val Non-Air Accuracy
       ↓
   [Model Checkpointing]
       ↓
Best Model Selection
(based on val_loss)
```

## Data Flow Summary

```
.schem files → NBT Parser → Voxel Arrays → One-Hot Encoding
                                              ↓
Filename → Tags → Text Prompt → Sentence-BERT → Text Embedding
                                                    ↓
                                    ┌───────────────┴───────────────┐
                                    ↓                               ↓
                              [TRAINING]                      [INFERENCE]
                                    ↓                               ↓
                          Voxels + Text Emb                   z + Text Emb
                                    ↓                               ↓
                                  VAE                            Decoder
                                    ↓                               ↓
                            Reconstructed Voxels            Generated Voxels
                                    ↓                               ↓
                              Calculate Loss                   Argmax
                                    ↓                               ↓
                           Update Weights                    Block IDs
                                                                    ↓
                                                              .schem File
```

---

**Architektur:** Conditional 3D VAE mit FiLM  
**Framework:** PyTorch + PyTorch Lightning  
**Text Encoding:** Sentence-BERT  
**Voxel Encoding:** One-Hot (26 classes)  
**Training:** Beta-VAE mit KL Annealing
