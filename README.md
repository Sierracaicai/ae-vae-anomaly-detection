# 🔍 AE-VAE Anomaly Detection

This project evaluates and compares anomaly detection approaches on network traffic data using:

- 🧠 Deep learning methods: Autoencoder (AE), Variational Autoencoder (VAE)
- 🧪 Classical methods: Isolation Forest, One-Class SVM

> Built as a portfolio-grade, modular and reproducible data science project.

---

## 📁 Project Structure

```
ae-vae-anomaly-detection/
├── notebooks/                 # Step-by-step notebooks for exploration and training
│   └── 0_exploration_and_baseline.ipynb
├── src/                       # Source code modules
│   ├── utils/
│   │   ├── eda_tools.py       # Reusable EDA functions
│   │   ├── preprocess.py      # Data cleaning, transformation, encoding, scaling
│   │   └── ...
│   └── models/                # (Planned) AE/VAE architectures and trainers
├── data/                      # Raw and processed data
│   ├── raw/
│   └── processed/
├── docs/                      # Module-level documentation (EDA, Preprocessing, Experiments)
│   └── feature_review.md
├── README.md
```

---

## 📊 Dataset

**UNSW-NB15** — A modern labeled network intrusion detection dataset.

- 49 features + `attack_cat` + binary `label`
- Mixture of normal traffic and multiple attack types
- Used only the first 100,000 rows for initial development (scalable)

---

## 🧰 Core Modules

### 🔹 EDA Tools (`src/utils/eda_tools.py`)

Reusable, production-ready utilities for data inspection and visualization.

- Distribution histograms, label counts
- Outlier detection (Z-score)
- Correlation heatmaps
- Distribution drift analysis (KS-test)

📘 See [docs/EDA Tools](src/utils/eda_tools.md)

---

### 🔹 Data Preprocessing (`src/utils/preprocess.py`)

- Categorical encoding (one-hot)
- MinMax/Standard scaling
- Optional log1p transformation
- Auto correlation filtering
- Cleaned output saved to CSV

---

## 🔬 Models and Experiments

| Model            | Type         | Status  | Notes                                   |
|------------------|--------------|---------|-----------------------------------------|
| Autoencoder      | Deep Learning | ✅ Done | Shallow AE with Dropout + BN            |
| VAE              | Deep Learning | ✅ Done | Custom training loop + KL-div loss      |
| Isolation Forest | Traditional  | ✅ Done | Baseline comparison                      |
| One-Class SVM    | Traditional  | ✅ Done | Baseline comparison                      |

📈 Metrics: MSE reconstruction error, AUC, precision, recall

---

## 🧪 Experimental Design

- Trained AE/VAE on **normal samples only**
- Tested on **mixed samples (normal + anomalous)**
- Compared anomaly scores across methods
- Evaluated impact of:
  - Sample size
  - Thresholding method
  - Feature engineering
  - Model complexity

---

## 🔍 Future Work

- Add interpretability via SHAP / LIME
- Hyperparameter tuning and AE/VAE deepening
- Build modular `train_runner.py`
- Add CLI + logging support for reproducible runs

---

## 👩‍💻 Author

Created by [Your Name] · Portfolio Project  
> *Feel free to fork, study or reuse parts of this repo in your own data science learning path.*