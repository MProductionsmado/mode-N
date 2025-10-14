# Classifier-Free Guidance (CFG) - Implementation Complete! 🎉

## ✅ Was wurde implementiert

### 1. **Training mit Conditioning Dropout**
- `src/training/trainer_discrete_diffusion.py`:
  - 10% der Trainingsschritte mit leerem Text-Embedding
  - Modell lernt sowohl conditional als auch unconditional generation
  - Ermöglicht CFG während Generation

### 2. **CFG Generation**
- `src/models/discrete_diffusion_3d.py`:
  - Neue `generate()` Methode mit `guidance_scale` Parameter
  - Neue `p_sample_cfg()` Methode für CFG Sampling
  - Formula: `output = uncond + scale × (cond - uncond)`

### 3. **Config Updates**
- `config/config.yaml`:
  - `conditioning_dropout: 0.1` für Training
  - `guidance_scale: 3.0` als Default für Generation

### 4. **Generation Script**
- `scripts/generate_discrete_diffusion.py`:
  - Neues `--guidance-scale` Argument
  - Automatisch verwendet Config-Default wenn nicht angegeben

## 🚀 Wie zu nutzen

### Training (NEU von vorne)

```bash
# Auf RunPod
cd "model N"
git pull

# Training mit CFG
python3 scripts/train_discrete_diffusion.py

# WICHTIG: Training von Epoch 0!
# Altes Checkpoint (Epoch 67) ist INKOMPATIBEL (wurde ohne conditioning dropout trainiert)
```

**Training Logs zeigen jetzt:**
```
Initialized Discrete Diffusion Model
Number of block categories: 50 (oder 172 je nach Config)
Timesteps: 1000
Conditioning Dropout: 10.0% (CFG)  ← NEU!
```

### Generation mit CFG

```bash
# Test 1: No Guidance (wie vorher)
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_X/checkpoints/epoch=XX.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 5 \
  --guidance-scale 1.0

# Test 2: Balanced Guidance (EMPFOHLEN) ⭐
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_X/checkpoints/epoch=XX.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 10 \
  --guidance-scale 3.0

# Test 3: Strong Guidance (sehr konsistent, wenig Variation)
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_X/checkpoints/epoch=XX.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 5 \
  --guidance-scale 7.5

# Test 4: Weak Guidance (viel Variation)
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_X/checkpoints/epoch=XX.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 5 \
  --guidance-scale 1.5
```

## 🎯 Guidance Scale Guide

| Scale | Effekt | Wann nutzen |
|-------|--------|-------------|
| **1.0** | Kein CFG, nur Noise | Maximale Variation (eventuell zu chaotisch) |
| **1.5-2.0** | Schwach | Viel Variation, Prompt hat wenig Einfluss |
| **3.0** ⭐ | Balanced | **EMPFOHLEN: Gute Balance** |
| **5.0** | Mittel-Stark | Text wichtiger, weniger Variation |
| **7.5** | Stark | Wie Stable Diffusion, sehr konsistent |
| **10.0+** | Sehr stark | Fast keine Variation (Overfitting auf Prompt) |

## 📊 Erwartete Ergebnisse

### Nach Epoch 50-100 mit CFG:

**Scale 1.0 (No Guidance):**
```
"oak tree" seed=1 → Irgendein Baum (oak, birch, spruce?)
"oak tree" seed=2 → Komplett anderer Baum
"oak tree" seed=3 → Eventuell gar kein Baum
→ ZU VIEL Variation ❌
```

**Scale 3.0 (Balanced - EMPFOHLEN):** ⭐
```
"oak tree" seed=1 → Kleiner dichter oak tree ✅
"oak tree" seed=2 → Großer lichter oak tree ✅
"oak tree" seed=3 → Mittlerer buschiger oak tree ✅
→ PERFEKTE Balance! 🎯
```

**Scale 7.5 (Strong):**
```
"oak tree" seed=1 → Oak tree Variante A
"oak tree" seed=2 → Oak tree Variante A (90% gleich)
"oak tree" seed=3 → Oak tree Variante A (90% gleich)
→ ZU WENIG Variation ❌
```

## ⚠️ Wichtige Hinweise

### 1. **Altes Checkpoint NICHT kompatibel**
```bash
# ❌ FALSCH - Epoch 67 wurde OHNE conditioning dropout trainiert:
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_11/checkpoints/epoch=67.ckpt \
  --guidance-scale 3.0
# → Funktioniert, aber CFG wirkt NICHT (unconditional wurde nie gelernt!)

# ✅ RICHTIG - Neues Training MIT conditioning dropout:
python3 scripts/train_discrete_diffusion.py  # Von Epoch 0
# Warte bis Epoch 50-100
# Dann teste mit --guidance-scale 3.0
```

### 2. **Training dauert ~50-100 Epochs**
- Epoch 0-20: Modell lernt Basics (mit und ohne Text)
- Epoch 20-50: CFG beginnt zu wirken
- Epoch 50-100: CFG voll funktionsfähig
- **Test frühestens ab Epoch 50!**

### 3. **Batch Size eventuell reduzieren**
- CFG braucht 2× Forward Pass (conditional + unconditional)
- Wenn OOM: Reduziere `batch_size` von 2 auf 1

## 🔬 Testing Strategie

### Phase 1: Training (Epoch 0-50)
```bash
# Einfach trainieren lassen
python3 scripts/train_discrete_diffusion.py
```

### Phase 2: Erste Tests (Epoch 50)
```bash
# Test ohne CFG (Baseline)
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint .../epoch=50.ckpt \
  --prompt "oak tree" \
  --guidance-scale 1.0 \
  --num-samples 5

# Test mit CFG
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint .../epoch=50.ckpt \
  --prompt "oak tree" \
  --guidance-scale 3.0 \
  --num-samples 5

# Vergleich: Sind sie unterschiedlicher?
```

### Phase 3: Scale Tuning (Epoch 100)
```bash
# Teste verschiedene Scales
for scale in 1.5 2.0 3.0 5.0 7.5; do
  python3 scripts/generate_discrete_diffusion.py \
    --checkpoint .../epoch=100.ckpt \
    --prompt "oak tree" \
    --guidance-scale $scale \
    --num-samples 3
done

# Finde besten Scale für dein Use-Case
```

## 📈 Erfolgsmetrik

**Vorher (ohne CFG):**
- 10 Generationen von "oak tree" → 9 fast identisch ❌

**Nachher (mit CFG scale=3.0):**
- 10 Generationen von "oak tree" → 10 deutlich unterschiedlich ✅
- Aber alle erkennbar als "oak tree" ✅

## 🎨 Advanced: Prompt Engineering mit CFG

**Mit CFG kannst du jetzt kreativ werden:**

```bash
# Spezifische Variationen
python3 scripts/generate_discrete_diffusion.py \
  --prompt "small dense oak tree" \
  --guidance-scale 5.0  # Höher = "small dense" wichtiger

python3 scripts/generate_discrete_diffusion.py \
  --prompt "tall oak tree" \
  --guidance-scale 5.0

python3 scripts/generate_discrete_diffusion.py \
  --prompt "oak tree" \
  --guidance-scale 2.0  # Niedriger = mehr Variation

# Kombination mit Temperature
python3 scripts/generate_discrete_diffusion.py \
  --prompt "oak tree" \
  --guidance-scale 3.0 \
  --temperature 1.2 \
  --sample-mode multinomial
```

## 🔧 Troubleshooting

### Problem: CFG wirkt nicht (alle Generationen noch gleich)
**Lösung:** 
- Training noch nicht lange genug (< Epoch 50)
- Oder: Alte Checkpoint ohne conditioning dropout
- → Warte bis Epoch 100 oder trainiere neu

### Problem: CUDA OOM während Training
**Lösung:**
```yaml
# config/config.yaml
training:
  batch_size: 1  # Von 2 reduziert
```

### Problem: Generationen zu chaotisch
**Lösung:**
- Erhöhe `guidance_scale` von 3.0 auf 5.0 oder 7.5
- Oder: Weiter trainieren (Epoch 150+)

### Problem: Generationen noch zu ähnlich
**Lösung:**
- Reduziere `guidance_scale` von 3.0 auf 1.5 oder 2.0
- Oder: Erhöhe `conditioning_dropout` von 0.1 auf 0.15

---

## 🎉 Zusammenfassung

**CFG ist jetzt implementiert!**

✅ Training mit 10% conditioning dropout
✅ Generation mit guidance_scale Parameter
✅ Config mit sinnvollen Defaults (scale=3.0)
✅ Generation Script mit --guidance-scale Argument

**Nächste Schritte:**
1. `git pull` auf RunPod
2. Training starten (von Epoch 0)
3. Ab Epoch 50: Teste verschiedene guidance_scales
4. Finde optimalen Scale (vermutlich 3.0-5.0)
5. Genieße vielfältige Generationen! 🌳🎲

**Training Zeit:** ~50-100 Epochs = ~20-40 Stunden
**Lohnt sich?** JA! Industrie-Standard für Diffusion Models! 🚀
