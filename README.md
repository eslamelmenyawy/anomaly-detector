<div align="center">
  <img src="se_logo.svg" width="120"/>
  <br><br>
  <strong>Sam Elmenyawy</strong><br>
  AI · Deep Learning · Computer Vision · NLP
</div>

---

# Anomaly Detection with Anomalib — Hazelnut Dataset

Industrial anomaly detection using two models from the [Anomalib](https://github.com/openvinotoolkit/anomalib) library, trained on a custom hazelnut dataset and deployed via two inference backends: **OpenVINO** and **PyTorch (TorchInferencer)**.

---

## What This Project Does

Trains unsupervised anomaly detection models to classify product images as **Normal** or **Anomalous** (crack defects) — no labels needed during training, only normal images. Two separate pipelines are implemented:

| Notebook | Model | Backend | Epochs | Params |
|---|---|---|---|---|
| `anomly_openvino.ipynb` | CFlow | OpenVINO (`.xml`) | 25 | 236M |
| `anomly2_lightrch.ipynb` | PaDiM | TorchInferencer (`.pt`) | 1 | 2.8M |

---

## Dataset

**Hazelnut Toy Dataset** — custom folder structure:

```
hazelnut_toy/
├── good/        ← normal images (training + validation)
│   ├── 00.jpg
│   ├── 01.jpg
│   └── ...
└── crack/       ← anomalous images (evaluation only)
    ├── 00.jpg
    └── ...
```

> Only `good/` images are used during training. `crack/` images are used only at test/inference time — this is the core principle of unsupervised anomaly detection.

---

## Models

### CFlow (Conditional Normalizing Flow)
- **Architecture:** Normalizing flow on top of a feature extractor
- **Parameters:** 236M
- **Training epochs:** 25
- **Image size:** 512 × 512
- **Task:** Classification (image-level anomaly score)
- **Metrics:** AUROC, F1Score (image-level)
- **Normalization:** MinMax
- **Threshold:** F1AdaptiveThreshold
- **Export:** OpenVINO format (`model.xml` + `model.bin`) for optimised CPU inference

### PaDiM (Patch Distribution Modelling)
- **Architecture:** Multivariate Gaussian distribution fitted on ResNet-18 patch features
- **Backbone:** ResNet-18 (pretrained)
- **Feature layers:** `layer1`, `layer2`, `layer3`
- **Parameters:** 2.8M
- **Training epochs:** 1 (distribution fitting — no gradient updates needed)
- **Export:** PyTorch format (`model.pt`) via TorchInferencer
- **Custom threshold:** 0.4 — score > 0.4 → Anomalous

---

## Pipeline

```
hazelnut_toy/
        │
        ▼  Anomalib Folder datamodule
   Normal images (good/)  →  Train  →  Fit anomaly model
                                              │
                                    Export weights
                               ┌──────────────┴──────────────┐
                               ▼                             ▼
                        OpenVINO .xml                  PyTorch .pt
                        (CFlow — CPU)           (PaDiM — TorchInferencer)
                               │                             │
                          Test image                   Test image
                               │                             │
                        Anomaly Score                 Anomaly Score
                               │                             │
                         Normal / Anomalous          Normal / Anomalous
```

---

## Results

### CFlow — OpenVINO Inference

| Image | Prediction Score | Label |
|---|---|---|
| `good/00.jpg` | 2.1502 | ✅ Normal |

### PaDiM — TorchInferencer

| Image | Raw Score | Threshold | Label |
|---|---|---|---|
| `good/15.jpg` | — | 0.4 | ✅ Normal |
| `crack/04.jpg` | 0.4988 | 0.4 | ❌ Anomalous |

PaDiM correctly flagged the crack image with a score above the threshold.

---

## Setup

### Environment

```bash
conda create -n anomalib-gpu python=3.10
conda activate anomalib-gpu
pip install anomalib==1.0.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install openvino
```

### Dataset Structure

Set your dataset path in Cell 2 of each notebook:

```python
root = r"path\to\hazelnut_toy"
normal_dir  = "good"
abnormal_dir = "crack"
```

---

## Notebook Walkthrough

### `anomly_openvino.ipynb` — CFlow + OpenVINO

| Cell | Description |
|---|---|
| 1 | Imports, seed, GPU setup |
| 2 | Folder datamodule — 512×512, batch 16 |
| 3 | CFlow model init |
| 4 | Engine config — 25 epochs, AUROC + F1 metrics |
| 5 | `engine.fit()` — training |
| 6 | Export to OpenVINO (`.xml` + `.bin`) |
| 7 | Load test image, preprocess |
| 8 | OpenVINOInferencer — predict + visualise |

### `anomly2_lightrch.ipynb` — PaDiM + TorchInferencer

| Cell | Description |
|---|---|
| 1 | Imports, seed, GPU setup |
| 2 | Folder datamodule |
| 3 | PaDiM model — ResNet-18, layers 1–3, 100 features |
| 4 | Engine — 1 epoch (distribution fitting only) |
| 5 | `engine.fit()` — training |
| 6 | Export to PyTorch `.pt` |
| 7 | `engine.test()` — evaluation |
| 8 | TorchInferencer — normal image prediction |
| 9 | TorchInferencer — anomalous image prediction + custom threshold |

---

## Why Two Models?

| | CFlow | PaDiM |
|---|---|---|
| Approach | Normalizing flow (generative) | Gaussian distribution on patch features |
| Training time | Slow (25 epochs, 236M params) | Fast (1 epoch, 2.8M params) |
| Deployment | OpenVINO — optimised for Intel CPU | PyTorch — flexible, easy to integrate |
| Best for | High accuracy, production on Intel hardware | Fast prototyping, low resource environments |
| Score interpretation | Higher score = more anomalous | Higher score = more anomalous (threshold: 0.4) |

---

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Framework | PyTorch + CUDA 12.1 |
| Anomalib | v1.0.1 |
| OS | Windows 11, Anaconda |

---

## References

- [Anomalib — OpenVINO Toolkit](https://github.com/openvinotoolkit/anomalib)
- [CFlow-AD: Real-World Anomaly Detection with Normalizing Flows](https://arxiv.org/abs/2107.12571)
- [PaDiM: A Patch Distribution Modelling Framework for Anomaly Detection](https://arxiv.org/abs/2011.08785)
- [OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)
