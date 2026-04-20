[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19664827.svg)](https://doi.org/10.5281/zenodo.19664827)
# FusionWeightnet: A Four-View CNN Framework for Non-Invasion Body Weight Estimation of Tenyi Vo Pigs

> 📄 **Manuscript Notice:**
> This repository contains the **official source code** for the paper:
>
> **"FusionWeightnet: a Four-View Convolutional Neural Network Framework for Non-Invasion Body Weight Estimation of Tenyi Vo Pigs Under Smallholder Farms conditions"**
> Submitted to *The Visual Computer* — Springer, 2026
>
> ⭐ **If you use this code, please cite our manuscript** (see Citation section below).

---

## 📌 Overview

This repository implements FusionWeightnet, a deep learning framework that estimates the body weight of Tenyi Vo pigs non-invasively using multi-view images combined with morphometric tabular features. The method fuses ResNet50-extracted image features with polynomial-expanded morphometric measurements through a fully connected regression network. This code allows full reproduction of all experiments reported in the manuscript, including training, testing, validation, and Grad-CAM visualizations.

---

## 🖥️ System Requirements

- Operating System: Windows 10 / Ubuntu 20.04 / macOS 11+
- Python: 3.10 or higher
- GPU: NVIDIA GPU with CUDA (optional but recommended; CPU mode also supported)
- RAM: 16 GB recommended

---

## ⚙️ Installation & Dependencies

**Step 1 — Clone this repository:**
```bash
git clone https://github.com/YOURUSERNAME/velato-web.git
cd velato-web
```

**Step 2 — Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**Step 3 — Install all dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

```
tensorflow
numpy
pandas
scikit-learn
opencv-python
matplotlib
tqdm
```

All versions are listed in `requirements.txt`.

---

## 📁 Repository Structure

```
velato-web/
│
├── cnn_model.py          # Main training script — ResNet50 + tabular fusion model
├── testing_code.py       # Testing/evaluation script on new data
├── validation.py         # 5-Fold cross-validation script
├── requirements.txt      # Python dependencies
├── CITATION.cff          # Citation metadata
└── README.md             # This file
```

---

## 🗂️ Dataset Structure Required

Your data directory should be organized as follows:

```
Velato Data/
├── pig001/
│   ├── front.jpg
│   ├── side.jpg
│   └── top.jpg
├── pig002/
│   └── ...
MORPHOMETRIC.csv          # Contains animal_no, weight, and morphometric columns
```

The CSV file must include the following columns:
- `animal_no` — matches folder names
- `weight` — target variable (kg)
- Optional: `sex`, `age_in_months`, `face_length`, `height_at_wither`, `heart_girth`, `middle_girth`, `body_length`, `neck_length`, `face_width`, `ear_length`, `shoulder_width`, `hip_width`, `tail_length`, `punch_girth`

---

## 🚀 Usage

### 1. Train the model

Update the `IMG_DIR` and `CSV_PATH` variables in `cnn_model.py` to point to your data, then run:

```bash
python cnn_model.py
```

This will:
- Load and preprocess images and tabular features
- Extract ResNet50 features
- Train the fusion regression model
- Save the best model to `paper_outputs/best_model.keras`
- Generate Figures 1–4 and Table 1 CSV in `paper_outputs/`

### 2. Test the model

```bash
python testing_code.py
```

Loads the saved model and evaluates it, printing R², RMSE, MAE, and MAPE.

### 3. Run cross-validation

```bash
python validation.py
```

Runs 5-fold cross-validation and reports mean ± std R², RMSE, and MAE.

---

## 🔬 Reproducing Paper Results

To reproduce **Table 1** (Train/Validation/Test performance):
```bash
python cnn_model.py
# Results saved to paper_outputs/Table_1_Performance.csv
```

To reproduce **Figure 1** (Actual vs Predicted scatter plot):
```bash
# Generated automatically by cnn_model.py
# Output: paper_outputs/Figure_1.png
```

To reproduce **Figures 3–4** (Grad-CAM visualizations):
```bash
# Generated automatically by cnn_model.py
# Output: paper_outputs/Figure_3_Validation_*.png
#         paper_outputs/Figure_4_Test_*.png
```

---

## 📖 Key Algorithms

| File | Description | Paper Section |
|------|-------------|---------------|
| `cnn_model.py` | ResNet50 feature extraction + tabular fusion + regression | Section 3 |
| `cnn_model.py` | Grad-CAM heatmap generation | Section 3.4 |
| `validation.py` | 5-fold cross-validation | Section 4 |
| `testing_code.py` | Independent test set evaluation | Section 4 |

---

## 📝 Citation

If you use this code or find it helpful in your research, please cite our paper:

```bibtex
@article{nyekha2026fusionweightnet,
  author    = {Nyekha, Velato and others},
  title     = {FusionWeightnet: a Four-View Convolutional Neural Network Framework
               for Non-Invasion Body Weight Estimation of Tenyi Vo Pigs Under
               Smallholder Farms conditions},
  journal   = {The Visual Computer},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.5281/zenodo.19662480},
  url       = {https://doi.org/10.5281/zenodo.19662480}
}
```


---
@article{nyekha2026fusionweightnet,
  author    = {Nyekha, Velato},
  title     = {FusionWeightnet: a Four-View Convolutional Neural Network 
               Framework for Non-Invasion Body Weight Estimation of Tenyi 
               Vo Pigs Under Smallholder Farms conditions},
  journal   = {The Visual Computer},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.5281/zenodo.19664827},
  url       = {https://doi.org/10.5281/zenodo.19664827}
}
## 📧 Contact

For questions about the code or paper, please contact:

- **Velato Nyekha** — [velatonyekha@gmail.com]
- Or open an **Issue** on this GitHub repository

---

## 📜 License

This project is licensed under the MIT License.
