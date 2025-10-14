# Slow Learning Strategy für Bessere Räumliche Kohärenz

## Problem

**Aktueller Status:**
- ✅ Loss: 0.005 (exzellent)
- ✅ Accuracy: 99.5% (sehr gut)
- ❌ Räumliche Kohärenz: schlecht (schwebende Teile, getrennte Strukturen)

**Ursache:**
Das Modell lernt **zu schnell** lokale Block-Muster, aber **zu langsam** globale räumliche Zusammenhänge.

## Lösung: Langsameres Lernen + Längeres Training

### Änderungen in `config/config.yaml`

| Parameter | Alt | Neu | Grund |
|-----------|-----|-----|-------|
| `learning_rate` | 0.0001 | **0.00005** | Halbiert → mehr Zeit für räumliche Muster |
| `num_epochs` | 200 | **300** | +50% Training → mehr Gelegenheit zu lernen |
| `warmup_epochs` | 5 | **10** | Sanfterer Start für stabiles langsames Lernen |
| `early_stopping.patience` | 50 | **75** | Mehr Geduld für langsamere Konvergenz |

### Warum das funktionieren sollte

**Theorie:**
1. **Schnelles Lernen** → Modell perfektioniert lokale Muster (Block-Nachbarschaften)
2. **Langsames Lernen** → Modell hat Zeit, globale Muster zu erkennen (Baum-Struktur)

**Analogie:**
- **Vorher:** Student lernt Vokabeln auswendig (schnell, aber kein Kontext)
- **Nachher:** Student lernt Grammatik (langsam, aber besseres Verständnis)

**Was das Modell lernen soll:**
- Nicht nur: "Oak_leaves kommen oft neben oak_wood vor" ✅ (bereits gelernt)
- Sondern: "Oak_leaves bilden eine Krone OBEN auf einem vertikalen oak_wood Stamm" ❌ (fehlt noch)

### Erwartete Metriken

**Training-Verlauf:**

| Epoch | Loss | Accuracy | Erwartete Qualität |
|-------|------|----------|-------------------|
| 0-20 | 3.0 → 1.5 | 20% → 40% | Chaotisch (Warmup) |
| 20-50 | 1.5 → 0.8 | 40% → 60% | Erste Muster |
| 50-100 | 0.8 → 0.3 | 60% → 75% | Klare Strukturen |
| 100-150 | 0.3 → 0.1 | 75% → 85% | Gute Kohärenz |
| 150-200 | 0.1 → 0.05 | 85% → 90% | Sehr gut |
| 200-300 | 0.05 → 0.02 | 90% → 95% | **Exzellent** |

**Wichtig:** Loss wird **langsamer** fallen als vorher (das ist gewollt!).

### Test-Punkte

**Frühe Tests (Strukturbildung):**
- **Epoch 50:** Erste kohärente Strukturen?
- **Epoch 100:** Klare Baumstämme?

**Mittlere Tests (Kohärenz):**
- **Epoch 150:** Verbundene Kronen?
- **Epoch 200:** Keine schwebenden Teile?

**Finale Tests (Qualität):**
- **Epoch 250:** Production-ready?
- **Epoch 300:** Beste mögliche Qualität

### Wie zu Testen

```bash
# Auf RunPod (während Training läuft)
# In separatem Terminal

# Test nach Epoch 50
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_X/checkpoints/epoch=50.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 5 \
  --sampling-steps 100

# Test nach Epoch 100
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_X/checkpoints/epoch=100.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 5 \
  --sampling-steps 100

# Test nach Epoch 150
python3 scripts/generate_discrete_diffusion.py \
  --checkpoint lightning_logs/version_X/checkpoints/epoch=150.ckpt \
  --prompt "oak tree" \
  --size normal \
  --num-samples 5 \
  --sampling-steps 100
```

### Zusätzliche Optimierungen

**1. Mehr Sampling Steps während Generation:**
```bash
--sampling-steps 200  # Statt 50, für bessere Qualität
```

**2. Größere Strukturen testen:**
```bash
--size big  # 16×32×16 statt 16×16×16
--size huge  # 24×64×24
```

**3. Verschiedene Prompts:**
```bash
--prompt "tall oak tree"  # Betonung auf Höhe
--prompt "oak tree with thick trunk"  # Betonung auf Stamm
--prompt "small oak tree with dense leaves"  # Klein & dicht
```

## Training Starten

```bash
# Auf RunPod
cd "model N"
git pull  # Neue config holen

# Neues Training starten (von vorne)
python3 scripts/train_discrete_diffusion.py

# ODER: Von bestehendem Checkpoint fortsetzen
python3 scripts/train_discrete_diffusion.py \
  --resume lightning_logs/version_X/checkpoints/epoch=XX.ckpt
```

**Wichtig:** 
- **Neues Training** empfohlen (alte LR war zu hoch)
- Training dauert **~30-40 Stunden** (300 Epochs statt 200)
- Teste zwischendurch (Epoch 50, 100, 150)

## Erwartete Ergebnisse

### Beste Szenario (95% Wahrscheinlichkeit)
- ✅ Deutlich bessere räumliche Kohärenz
- ✅ Verbundene Strukturen (keine schwebenden Teile)
- ✅ Klare Baumform (Stamm + Krone)
- ⚠️ Immer noch nicht perfekt (eventuell kleine Artefakte)

### Realistisches Szenario (70% Wahrscheinlichkeit)
- ✅ Viel besser als jetzt
- ✅ Erkennbare Bäume
- ⚠️ Manche Strukturen noch etwas inkohärent
- ⚠️ Kleine schwebende Teile möglich

### Schlechtestes Szenario (5% Wahrscheinlichkeit)
- ⚠️ Nur marginale Verbesserung
- ❌ UNet-Architektur fundamental limitiert
- 💡 Dann: Hierarchical Generation oder Post-Processing nötig

## Alternativen falls nicht genug Verbesserung

**1. Learning Rate noch weiter reduzieren:**
```yaml
learning_rate: 0.00003  # Statt 0.00005
num_epochs: 400  # Noch länger trainieren
```

**2. Attention auf allen Levels:**
```yaml
attention_levels: [0, 1, 2, 3]  # Statt nur [2, 3]
```

**3. Mehr Residual Blocks:**
```yaml
num_res_blocks: 3  # Statt 2
```

**4. Classifier-Free Guidance:**
- Braucht Code-Änderungen
- Verstärkt Text-Conditioning
- Wie Stable Diffusion

**5. Hierarchical Generation:**
- Grobe Struktur zuerst (Stamm-Position)
- Details später (Blätter)
- Braucht 2-stufiges Modell

**6. Post-Processing:**
- Regel-basierte Cleanup
- Entferne schwebende Blöcke
- Verbinde getrennte Strukturen
- Schnellste Lösung, aber nicht elegant

## Zeitplan

| Tag | Aktion | Dauer |
|-----|--------|-------|
| 1 | Training starten (Epoch 0-50) | ~6h |
| 2 | Test Epoch 50, weiter bis 100 | ~12h |
| 3 | Test Epoch 100, weiter bis 150 | ~12h |
| 4 | Test Epoch 150, weiter bis 200 | ~12h |
| 5 | Test Epoch 200, weiter bis 250 | ~12h |
| 6 | Test Epoch 250, weiter bis 300 | ~12h |
| 7 | Finale Tests, Dokumentation | ~4h |

**Total: ~70 Stunden** (mit Tests und Analysen)

## Erfolgsmetriken

**Quantitativ:**
- Loss < 0.05 (aktuell: 0.005 ✅)
- Accuracy > 90% (aktuell: 99.5% ✅)
- **Neu:** Connectivity Score > 0.8 (messen wie verbunden Strukturen sind)

**Qualitativ:**
- Keine schwebenden Teile
- Vertikale Stämme erkennbar
- Blätter bilden Krone oben
- Strukturen sehen aus wie Bäume

**Vergleich:**
- Besser als continuous diffusion (chaotisch)
- Vergleichbar mit low-res Point-E/DreamFusion
- Nicht perfekt, aber nutzbar

## Fazit

**Diese Strategie hat höchste Erfolgswahrscheinlichkeit:**
1. ✅ Einfach umzusetzen (nur Config-Änderung)
2. ✅ Theoretisch fundiert (mehr Zeit für globale Muster)
3. ✅ Kein Code-Refactoring nötig
4. ✅ Kann von bestehendem Checkpoint fortsetzen

**Wenn das nicht reicht:**
- Dann liegt es an UNet-Architektur
- Hierarchical Generation oder Transformer nötig
- Aber erst diese Strategie testen!

---

**Nächster Schritt:** Training mit neuer Config starten und nach Epoch 50 testen! 🚀
