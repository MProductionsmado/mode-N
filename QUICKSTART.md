# Minecraft 3D Nature Asset Generator - Schnellstart 🚀

## Installation

### 1. Python-Umgebung erstellen

```powershell
# Python 3.9+ erforderlich
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Dependencies installieren

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Hinweis**: Für CUDA-Support (GPU-Training):
```powershell
# Für PyTorch mit CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Workflow

### Schritt 1: Daten Preprocessing

Analysiere zunächst den Datensatz:

```powershell
python scripts/preprocess_data.py --analyze-only
```

Dann verarbeite alle Schematics:

```powershell
python scripts/preprocess_data.py --input out --output data
```

**Output**:
- `data/processed/` - Voxel Arrays als .npy Dateien
- `data/splits/` - Train/Val/Test Metadata
- `data/dataset_statistics.json` - Datensatz-Statistiken

⏱️ **Dauer**: ~10-30 Minuten für 9180 Dateien

### Schritt 2: Training

```powershell
python scripts/train.py --config config/config.yaml
```

**Optional**: Debug-Modus für schnellen Test:
```powershell
python scripts/train.py --debug
```

**Optional**: Training fortsetzen:
```powershell
python scripts/train.py --resume models/last.ckpt
```

**Training überwachen**:
```powershell
tensorboard --logdir logs
```

⏱️ **Dauer**: 
- GPU (RTX 3080): ~2-4 Stunden für 200 Epochs
- CPU: ~24-48 Stunden

### Schritt 3: Asset Generierung

Einzelnes Asset generieren:

```powershell
python scripts/generate.py --checkpoint models/best.ckpt --prompt "big oak tree" --output generated.schem
```

Mehrere Variationen:

```powershell
python scripts/generate.py --checkpoint models/best.ckpt --prompt "birch tree with fall leaves" --num-samples 5 --output generated/tree.schem
```

**Interaktiver Modus**:

```powershell
python scripts/generate.py --interactive
```

### Schritt 4: Evaluation

```powershell
python scripts/evaluate.py --checkpoint models/best.ckpt --visualize --output evaluation
```

## Beispiel-Prompts

### Bäume
- `"oak tree"`
- `"big birch tree"`
- `"huge oak tree"`
- `"big birch tree with fall leaves"`
- `"spruce tree"`

### Büsche
- `"bush"`
- `"big bush"`
- `"small oak bush"`

### Spezielle Strukturen
- `"beehive in oak planks"`
- `"bee nest"`

## Konfiguration anpassen

Editiere `config/config.yaml` für:

- **Modellgröße**: `model.encoder.channels`, `model.sizes.*.latent_dim`
- **Batch Size**: `training.batch_size` (reduziere bei GPU-OOM)
- **Learning Rate**: `training.learning_rate`
- **VAE Beta**: `model.vae.beta` (höher = weniger Diversität, bessere Rekonstruktion)

## Tipps & Tricks

### GPU Memory Issues

Falls "CUDA out of memory":

1. Reduziere Batch Size in `config/config.yaml`:
   ```yaml
   training:
     batch_size: 8  # statt 16
   ```

2. Nutze Mixed Precision:
   ```yaml
   hardware:
     precision: 16  # statt 32
   ```

### CPU Training

Ändere in `config/config.yaml`:
```yaml
hardware:
  device: "cpu"
  num_workers: 2
```

### Bessere Diversität

Erhöhe Temperature beim Generieren:
```powershell
python scripts/generate.py --checkpoint models/best.ckpt --prompt "oak tree" --temperature 1.5
```

### Bessere Qualität

- Trainiere länger (mehr Epochs)
- Erhöhe KL-Annealing-Dauer
- Reduziere VAE Beta für bessere Rekonstruktion

## Troubleshooting

### Import Error: nbtlib

```powershell
pip install nbtlib
```

### CUDA not available

Installiere PyTorch mit CUDA-Support:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Datei nicht gefunden

Stelle sicher, dass du im Projektverzeichnis bist:
```powershell
cd "c:\Users\priva\Documents\MProductions\model N"
```

## Projektstruktur

```
model N/
├── out/                       # Original .schem Dateien (9180 Assets)
├── data/                      # Nach Preprocessing
│   ├── processed/             # .npy Voxel-Arrays
│   ├── splits/                # Train/Val/Test Metadata
│   └── dataset_statistics.json
├── models/                    # Gespeicherte Checkpoints
├── logs/                      # TensorBoard Logs
├── generated/                 # Generierte Assets
├── evaluation/                # Evaluation-Ergebnisse
├── src/                       # Source Code
├── scripts/                   # Haupt-Scripts
├── config/                    # Konfiguration
├── requirements.txt
└── README.md
```

## Nächste Schritte

1. **Experimentiere mit Prompts**: Teste verschiedene Beschreibungen
2. **Fine-Tuning**: Trainiere auf spezifischen Asset-Typen
3. **Integration**: Nutze die generierten Assets in deinem Projekt
4. **Erweitere Vocabulary**: Füge mehr Blöcke hinzu in `config/config.yaml`

## Support

- Schau in die Logs: `logs/` und Tensorboard
- Teste mit `--debug` Flag
- Prüfe `data/dataset_statistics.json` für Datensatz-Insights
