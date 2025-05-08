# 📓 Notebooks

This folder contains all step-by-step Jupyter notebooks used in this project.

---

## 📓 Notebook List

### `0_draft_experiments.ipynb`
- The very first draft version of this project.
- Includes initial data loading, simple AE/VAE models, and traditional method comparison.
- Retained for reference and idea evolution.

### `0_exploration_and_baseline.ipynb`
- A more structured and organized version of the original baseline.
- Contains:
  - Cleaned and labeled data import
  - Initial AE/VAE vs Isolation Forest/OcSVM performance comparison
  - Basic evaluation metrics
- Serves as the foundation for all future iterations.

### `EDA.ipynb`

- Exploratory Data Analysis (EDA) notebook executed before any modeling or preprocessing.

- Contains:
  - Dataset overview: shape, data types, missing values, duplicates
  - Label distribution and attack category frequencies
  - Feature histograms (with log-suggested and high variance/skew detection)
  - Correlation heatmap
  - Outlier detection using Z-score
  - Distribution drift check using KS-test (normal-only train/test comparison)

📍 Purpose: Understand data structure, inform preprocessing strategy, and guide model design.

### `2_model_train_ae_vae.ipynb`
- (To be created) Final training of AE and VAE models with callbacks, checkpoints, and hyperparameter tuning.

### `3_explainability.ipynb`
- (To be created) Interpretability using SHAP or LIME to explain AE/VAE predictions.

---

## ✅ Tip

You can run these notebooks directly in Google Colab by opening them via GitHub and selecting "Open in Colab".