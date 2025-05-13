# 🔍 AE-VAE Anomaly Detection

This project evaluates and compares anomaly detection approaches on network traffic data using:

- 🧠 Deep learning methods: Autoencoder (AE), Variational Autoencoder (VAE)
- 🧪 Classical methods: Isolation Forest, One-Class SVM, Local Outlier Factor, Elliptic Envelope

> Built as a portfolio-grade, modular and reproducible data science project.

---

## 📁 Project Structure

```
ae-vae-anomaly-detection/
├── data/
│   └── README.md           # Dataset Description
├── docs/
│   ├── feature_review.md  
│   ├── experiments/
│   │     ├── ae_eval_report.md      
│   │     ├── ae_depth_comparison_report.md
│   │     ├── bottleneck_report.md
│   │     ├── ae_activation_experiment.md
│   │     ├── ae_mixed_loss_experiment.md
│   │     ├── optimizer_experiment.md
│   │     └── thresholding_comparison.md
│   ├── final_ae_experiment_report.md  
│   ├── VAE_report.md       
│   ├── AE_vs_Traditional_Report.md 
│   └── SHAP_Interpretability_Report.md 
├── notebooks/              # Jupyter notebooks by stage
│   ├── 0_draft_experiments.ipynb 
│   ├── 0_exploration_and_baseline.ipynb 
│   ├── EDA.ipynb
│   ├── Preprocess_and_TSNE.ipynb
│   ├── AE_base_model.ipynb
│   ├── experiments/
│   │    ├── AE_Depth_Comparison.ipynb
│   │    ├── Ae_Bottleneck_Experiment.ipynb
│   │    ├── Ae_activation_Experiment.ipynb
│   │    ├── ae_mixed_loss_experiment.ipynb
│   │    ├── AE_optimizer_experiment.ipynb
│   │    └── AE_Adaptive_Thresholding_experiment.ipynb
│   ├── VAE_model.ipynb
│   ├── traditional_models_with_tuning.ipy
│   ├── Shap_Interpretability.ipynb
│   └── README.md
├── src/
│   ├── utils/              # Reusable utility modules
│   │   ├── load_data.py    # Load raw CSV with default column names
│   │   ├── load_data.md
│   │   ├── reduce_mem.py   # Downcast dtypes to reduce memory
│   │   ├── reduce_mem.md
│   │   ├── preprocess.py   # Full preprocessing pipeline
│   │   ├── preprocess.md
│   │   ├── eda_tools.py    # EDA plotting & statistics utilities
│   │   ├── eda_tools.md
│   │   ├── tsne_vis.py     # t-SNE 2D projection tool
│   │   ├── tsne_vis.md
│   │   └── module_reload.py# Hot-reload Python modules in Colab
│   └──models/
│       ├── ae_model.py           
│       ├── ae_model.md 
│       ├── ae_evaluation.py   
│       ├── ae_evaluation.md  
│       ├── thresholding.py
│       ├── vae_model.py
│       ├── beta_vae_model.py
│       └── best_ae.h5    
│
├── requirements.txt        # List required Python packages & version
├── .gitignore              # Exclude raw/processed data, env files
└── README.md               # Project overview & usage
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
- Rare category detection

📘 See [docs/EDA Tools](src/utils/eda_tools.md)

### 🔹 Data Preprocessing (`src/utils/preprocess.py`)

- Categorical encoding (one-hot)
- MinMax/Standard scaling
- Optional log1p transformation
- Auto correlation filtering
- Cleaned output saved to CSV

📘 See [docs/Data Preprocessing](src/utils/preprocess.md)

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

### 🔹 t-SNE Projection Tool (`src/utils/tsne_vis.py`)

- Visualizes high-dimensional data (e.g., features after preprocessing or latent encodings) in 2D space
- Helps assess the separability of normal vs anomaly samples
- Automatically samples from both classes and supports standardization toggle

📘 See [docs/t-SNE Projection Tool](src/utils/tsne_vis.md)

## 📓 Notebooks

- `0_draft_experiments.ipynb`: The very first draft version  
  - Located in [`notebooks/`](notebooks/)

- `0_exploration_and_baseline.ipynb`: A more structured and organized version of the original baseline
  - Located in [`notebooks/`](notebooks/)

- `EDA.ipynb`: Full exploratory data analysis (EDA) including:
  - Shape, types, nulls, duplicates
  - Label & category distributions
  - Feature histograms, skewness, correlation
  - Optional concept drift check (KS-test, normal-only)
  - Located in [`notebooks/`](notebooks/)

- `Preprocess_and_TSNE.ipynb`: To be completed


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


## 🔭 Future Work
- Add interpretability via SHAP / LIME
- Hyperparameter tuning and AE/VAE deepening
- Build modular `train_runner.py`
- Add CLI + logging support for reproducible runs
- **Strict “Train-Only” Feature Engineering**  
  Implement a fully pipeline-based workflow (e.g. custom `RareCategoryGrouper`, correlation filters, PCA) that is **fitted on the training set only** and then applied to validation/test sets. This will eliminate any potential leakage from unsupervised transforms.

- **Compare Leakage-Aware vs. Pragmatic Approaches**  
  Empirically evaluate the impact on AE/VAE and traditional model performance when grouping rare categories and dropping correlations using (a) full-data rules vs. (b) train-only rules.

- **Production-Ready SKLearn Pipeline**  
  Wrap the entire preprocess → train → evaluate steps into an `sklearn.Pipeline` (or TensorFlow Transform) to support reproducible CI/CD retraining and deployment.

- **Automated CI/CD Tests**  
  Add unit/integration tests that assert no data-leakage—e.g. confirm that transformers fitted on train data yield identical outputs on unseen samples.

- **Monitoring & Drift Detection in Deployment**  
  In a deployed setting, continuously monitor feature distributions and retrain the “train-only” pipeline when concept drift is detected.


---

## 👩‍💻 Author

Created by [Your Name] · Portfolio Project  
> *Feel free to fork, study or reuse parts of this repo in your own data science learning path.*