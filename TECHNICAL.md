# Minecraft 3D Asset Generator - Technische Dokumentation

## Architekturübersicht

### Modell: Conditional 3D Variational Autoencoder (CVAE)

Das System verwendet einen **3D Conditional VAE**, der speziell für die Generierung von Minecraft-Voxel-Strukturen entwickelt wurde.

#### Komponenten:

1. **Text Encoder**
   - Sentence-BERT (all-MiniLM-L6-v2)
   - Konvertiert Text-Prompts in 384-dimensionale Embeddings
   - Pre-trained auf großem Text-Korpus

2. **3D Encoder**
   - 3D-Convolutional Neural Network
   - Komprimiert Voxel-Daten (16³ bis 24×64×24) in latenten Space
   - Architektur: 4 Conv3D-Layer mit [64, 128, 256, 512] Channels
   - Output: μ (mean) und log σ² (log variance) für Variational Sampling

3. **Latent Space**
   - Variable Dimensionen je nach Größenkategorie:
     - Normal (16³): 128 Dimensionen
     - Big (16×32×16): 192 Dimensionen  
     - Huge (24×64×24): 256 Dimensionen
   - Ermöglicht Sampling für diverse Generierungen

4. **3D Decoder**
   - Transposed 3D-CNN für Rekonstruktion
   - FiLM (Feature-wise Linear Modulation) für Text-Conditioning
   - Jeder Decoder-Block erhält Text-Features als Conditioning
   - Output: Block-ID-Logits für jeden Voxel

#### Training:

**Loss Function:**
```
Total Loss = Reconstruction Loss + β × KL Divergence

Reconstruction Loss: Cross-Entropy zwischen predicted und actual blocks
KL Divergence: KL(q(z|x) || p(z)) für Regularisierung
```

**Beta-VAE:**
- β = 0.5 (konfigurierbar)
- KL Annealing über 50 Epochs für stabiles Training
- Balanciert Rekonstruktionsqualität und Latent Space-Struktur

**Optimizer:**
- Adam mit Learning Rate 1e-4
- Cosine Annealing Schedule mit 10 Epochs Warmup
- Weight Decay 1e-5

**Augmentation:**
- 90° Rotationen um Y-Achse (Minecraft-kompatibel)
- Horizontale Flips (X/Z-Achse)
- Nur geometrische Transformationen (keine Block-Manipulationen)

### Datenpipeline

#### 1. Schematic Parsing (.schem → Voxel Array)

**Format:** Sponge Schematic Format (NBT-basiert)

**Schritte:**
1. NBT-Datei laden
2. Dimensionen extrahieren (Width, Height, Length)
3. Palette-Mapping laden (Block-Namen → IDs)
4. BlockData als Varint-encoded Array dekodieren
5. Von Sponge-Ordering (Y,Z,X) zu (X,Y,Z) transponieren
6. Palette-IDs zu Vocabulary-IDs konvertieren
7. Zu Standard-Größe padden/croppen

**Block Vocabulary:**
- 26 häufigste Natur-Blöcke
- air (0), oak_log (1), oak_leaves (2), etc.
- Erweiterbar durch config.yaml

#### 2. Filename Parsing

**Pattern Recognition:**
```
[size]_[material]_[properties]_[numbers]--[uuid].schem
↓
Tags: [size, material, properties]
Materials: [wood types, leaf types]
Prompt: "size material properties"
```

**Beispiele:**
- `big_birch_wood_birch_leaves_fall_09_05.schem`
  → "big birch wood birch leaves fall"
- `beehive_oak_planks_04_01.schem`
  → "beehive oak planks"

**UUID-Handling:**
- Automatische Erkennung und Entfernung
- Keine Auswirkung auf Training

#### 3. Size Classification

**Auto-Detection:**
```python
if max_dimension <= 16:    → normal (16³)
elif max_dimension <= 32:  → big (16×32×16)
else:                      → huge (24×64×24)
```

**Padding/Cropping:**
- Zentriert auf X/Z-Achse
- Von unten auf Y-Achse (Strukturen wachsen nach oben)
- Erhält Struktur-Integrität

### Generierung

#### Inference-Prozess:

1. **Text Encoding:**
   - Prompt → Sentence-BERT → 384D-Embedding

2. **Size Detection:**
   - Keywords ("big", "huge") → Size Category
   - Default: "normal"

3. **Latent Sampling:**
   - z ~ N(0, I) × temperature
   - Temperature > 1: mehr Variation
   - Temperature < 1: näher an Trainingsdaten

4. **Conditional Decoding:**
   - z + text_embedding → Decoder
   - FiLM modulation mit Text-Features
   - Output: Block-Logits

5. **Block Selection:**
   - Argmax über Logits → Block-IDs
   - Conversion zu .schem Format
   - Export als Minecraft-kompatible Datei

#### Diversität & Kontrolle:

**Für mehr Variation:**
- Höhere Temperature (1.5-2.0)
- Mehrere Samples (--num-samples)
- Niedrigerer β während Training

**Für mehr Kontrolle:**
- Detailliertere Prompts
- Niedrigere Temperature (0.5-0.8)
- Höherer β während Training

### Metriken

#### Training Metrics:

1. **Reconstruction Loss:** Cross-Entropy Loss
   - Misst Block-by-Block Genauigkeit
   
2. **KL Divergence:** Regularisierung
   - Strukturiert Latent Space
   
3. **Block Accuracy:** % korrekt rekonstruierte Blöcke
   - Overall und Non-Air getrennt

4. **Per-Size Metrics:**
   - Separate Tracking für normal/big/huge

#### Evaluation Metrics:

1. **Reconstruction Quality:**
   - Test-Set Accuracy
   - Visueller Vergleich Original vs. Rekonstruktion

2. **Generation Quality:**
   - Prompt-Konsistenz (manuell)
   - Strukturelle Plausibilität
   - Block-Diversität

3. **Latent Space Quality:**
   - Interpolation-Tests
   - Cluster-Analyse

### Performance

#### Hardware Requirements:

**Minimum:**
- CPU: 4 Cores
- RAM: 16 GB
- Storage: 50 GB

**Empfohlen:**
- GPU: NVIDIA RTX 3060+ (8GB VRAM)
- RAM: 32 GB
- Storage: 100 GB SSD

#### Training Zeit:

**RTX 3080 (10GB):**
- ~1-2 Minuten pro Epoch
- 200 Epochs: ~3-4 Stunden
- Batch Size 16

**CPU Only:**
- ~15-20 Minuten pro Epoch
- 200 Epochs: ~2-3 Tage
- Batch Size 8

#### Inference Zeit:

**GPU:**
- <1 Sekunde pro Asset
- Batch-Generation: ~100ms pro Asset

**CPU:**
- ~2-5 Sekunden pro Asset

### Erweiterungen & Verbesserungen

#### Mögliche Erweiterungen:

1. **Größere Block-Vocabulary:**
   - Mehr Materialien (Stein, Erz, etc.)
   - Modded Blocks
   - Block States (Rotation, Properties)

2. **Hierarchisches VAE:**
   - Multi-Scale Generierung
   - Grobe → Feine Details

3. **Diffusion Models:**
   - Alternative zu VAE
   - Potentiell höhere Qualität

4. **ControlNet:**
   - Zusätzliche Steuerung via Sketches
   - Form-Vorgaben

5. **Größere Strukturen:**
   - 64³+ durch Sparse Convolutions
   - Chunk-basierte Generierung

6. **Multi-Modal Conditioning:**
   - Bild + Text
   - Referenz-Strukturen

#### Optimierungen:

1. **Mixed Precision Training:**
   - FP16 statt FP32
   - 2x schneller, weniger VRAM

2. **Gradient Checkpointing:**
   - Mehr Batch Size möglich
   - Trade-off: Speed vs. Memory

3. **Model Quantization:**
   - INT8 Inference
   - 4x schneller auf CPU

4. **Knowledge Distillation:**
   - Kleineres Modell
   - Schnellere Inference

### Code-Struktur

```
src/
├── data/
│   ├── schematic_parser.py    # NBT → Voxel Array
│   ├── preprocessing.py        # Filename parsing, size classification
│   └── dataset.py              # PyTorch Dataset, DataLoader
├── models/
│   └── vae_3d.py              # CVAE Architecture
├── training/
│   ├── losses.py              # Loss functions, metrics
│   └── trainer.py             # PyTorch Lightning Module
└── inference/
    ├── generator.py           # Generation logic
    └── visualizer.py          # Visualization utilities

scripts/
├── preprocess_data.py         # Datenverarbeitung
├── train.py                   # Training
├── generate.py                # Generierung
└── evaluate.py                # Evaluation
```

### Best Practices

#### Training:

1. Start mit kleinem Datensatz (--debug) zum Testen
2. Monitor Tensorboard für Overfitting
3. Nutze Checkpointing (alle 10 Epochs)
4. Early Stopping bei Validation Loss Plateau
5. KL Annealing für stabiles Training

#### Generierung:

1. Experimentiere mit Temperature (0.5-2.0)
2. Generiere mehrere Samples pro Prompt
3. Nutze spezifische Prompts für beste Ergebnisse
4. Teste verschiedene Size-Tags

#### Troubleshooting:

1. **Low Accuracy:** Längeres Training, mehr Daten, größeres Modell
2. **Poor Generation:** Höherer β, mehr Training
3. **No Diversity:** Niedrigerer β, höhere Temperature
4. **OOM Errors:** Kleinere Batch Size, Mixed Precision

---

**Entwickelt für Minecraft 1.19+**  
**Kompatibel mit WorldEdit, FastAsyncWorldEdit, Litematica**
