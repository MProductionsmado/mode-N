# Minecraft 3D Nature Asset Generator 🌳

Eine KI, die Minecraft Natur-Assets (Bäume, Büsche, etc.) als 3D-Schematics basierend auf Text-Prompts generiert.

## Features

- **Conditional 3D Generation**: Generiert individualisierte Assets basierend auf Text-Beschreibungen
- **Mehrere Größenklassen**:
  - Normal: 16x16x16
  - Big: 16x32x16
  - Huge: 24x64x24
- **Tag-basiertes Lernen**: Versteht Material-Tags (oak, birch, etc.) und Struktur-Tags (bush, tree, etc.)
- **Variational Autoencoder**: Jede Generation ist einzigartig durch Sampling im Latent Space

## Installation

```bash
pip install -r requirements.txt
```

## Projektstruktur

```
model N/
├── out/                    # Original Schematic-Dateien (9180 Assets)
├── data/                   # Preprocessed Daten
│   ├── processed/          # Voxel-Arrays als .npy
│   ├── metadata.json       # Tag-Informationen
│   └── splits/             # Train/Val/Test Split
├── models/                 # Gespeicherte Modelle
├── logs/                   # Training Logs
├── src/
│   ├── data/
│   │   ├── schematic_parser.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── vae_3d.py
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── text_encoder.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   └── inference/
│       ├── generator.py
│       └── visualizer.py
├── scripts/
│   ├── preprocess_data.py
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
├── config/
│   └── config.yaml
└── requirements.txt
```

## Verwendung

### 1. Daten Preprocessing

```bash
python scripts/preprocess_data.py --input out/ --output data/
```

### 2. Training

```bash
python scripts/train.py --config config/config.yaml
```

### 3. Asset Generierung

```bash
python scripts/generate.py --prompt "big birch tree with fall leaves" --output generated.schem
```

### 4. Evaluation

```bash
python scripts/evaluate.py --model_path models/best_model.ckpt
```

## Architektur

Das Modell verwendet einen **Conditional 3D Variational Autoencoder (CVAE)**:

1. **Text Encoder**: Sentence-BERT für Text-Embeddings
2. **3D Encoder**: 3D-CNN zur Kompression von Voxel-Daten
3. **Latent Space**: Variationaler Bottleneck für diverse Generierung
4. **3D Decoder**: Transposed 3D-CNN zur Rekonstruktion
5. **Conditioning**: Text-Features werden in alle Decoder-Layer injiziert

## Beispiel-Prompts

- `"oak tree"`
- `"big birch tree with fall leaves"`
- `"huge oak tree"`
- `"small bush"`
- `"beehive in oak planks"`

## Lizenz

MIT License
