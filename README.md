# FGMDTI

**FGMDTI: Fine-grained Multimodal Learning for Drug–Target Interaction Prediction via Bidirectional Cross-Attention**

---

## 🔬 Overview

FGMDTI is a fine-grained multimodal framework for drug–target interaction (DTI) prediction.  
Instead of modeling whole molecules or sequences, FGMDTI operates on biologically meaningful functional units, including drug fragments and protein domains.

The framework integrates:
- Structural representations from pretrained foundation models
- Semantic representations from textual descriptions
- Bidirectional cross-attention to model mutual adaptation between drug and protein substructures

---

## ✨ Key Features

- **Fine-grained modeling** via BRICS fragmentation and HMMER domain decomposition  
- **Multimodal fusion** of structural and semantic information  
- **Bidirectional cross-attention** for interaction modeling  
- **Parameter-efficient design** using lightweight Transformer adapters  
- **Interpretability** via substructure-level interaction analysis  

---

## 📁 Project Structure

```bash
FGMDTI/
│
├── data/                  # Dataset or sample data
├── ablation/              # Ablation study scripts / results
│
├── train.py               # Main training script
├── text_embedding.py      # Semantic feature extraction
├── autoencoder.py         # Representation learning module
├── heat_map.py            # Visualization & interpretability
│
├── README.md
