# 🔍 AE-VAE Anomaly Detection

This project evaluates and compares anomaly detection approaches on network traffic data using:

- 🧠 Deep learning methods: Autoencoder (AE), Variational Autoencoder (VAE)
- 🧪 Classical methods: Isolation Forest, One-Class SVM

> Built as a portfolio-grade, modular and reproducible data science project.

---

## 📁 Project Structure

```
ae-vae-anomaly-detection/
├── notebooks/
│   ├── 0_exploration_and_baseline.ipynb
│   └── EDA.ipynb
├── src/
│   ├── utils/
│   │   ├── eda_tools.py
│   │   ├── preprocess.py
│   │   ├── load_data.py
│   │   └── reduce_mem.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
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

- Distribution histograms, label counts
- Outlier detection (Z-score)
- Correlation heatmaps
- Distribution drift analysis (KS-test)

📘 See [docs/EDA Tools](src/utils/eda_tools.md)

### 🔹 Data Preprocessing (`src/utils/preprocess.py`)

- Categorical encoding (one-hot)
- MinMax/Standard scaling
- Optional log1p transformation
- Auto correlation filtering
- Cleaned output saved to CSV

### 🔹 Data Loader (`src/utils/load_data.py`)

- Loads raw CSV with default UNSW column names
- Used to decouple EDA from preprocessing

📘 See [docs/Data Loader](src/utils/load_data.md)

### 🔹 Memory Optimizer (`src/utils/reduce_mem.py`)

- Downcasts numerical columns to reduce memory usage
- Especially useful in Colab or large dataset scenarios
- Optional float16 support
- Automatically prints memory reduction summary

📘 See [docs/Memory Optimizer](src/utils/reduce_mem.md)

---

### 🔹 Colab Module Hot Reload (`src/utils/module_reload.py`)

- ⚡ Reloads all utility `.py` modules (preprocess, EDA, t-SNE, etc.) **without restarting the Colab runtime**
- Designed for **modular workflows** in Google Colab
- Automatically reloads:
  - `preprocess.py`
  - `eda_tools.py`
  - `tsne_vis.py`
  - `load_data.py`
  - `reduce_mem.py`

🧪 Run in Colab anytime after modifying a `.py`:

```python
%run /content/src/utils/module_reload.py
```

### 📌 t-SNE Projection Tool (`src/utils/tsne_vis.py`)

- Visualizes high-dimensional data (e.g., features after preprocessing or latent encodings) in 2D space
- Helps assess the separability of normal vs anomaly samples
- Automatically samples from both classes and supports standardization toggle

🧪 Example:

```python
from utils.tsne_vis import plot_tsne_projection

plot_tsne_projection(df, label_col='label', sample_size=5000, standardize=False)
```

## 📓 Notebooks

- `EDA.ipynb`: Full exploratory data analysis (EDA) including:
  - Shape, types, nulls, duplicates
  - Label & category distributions
  - Feature histograms, skewness, correlation
  - Optional concept drift check (KS-test, normal-only)
- Located in [`notebooks/`](notebooks/)

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