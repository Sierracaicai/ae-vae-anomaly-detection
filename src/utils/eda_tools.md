# 🧰 EDA Tools Module

**Path**: `src/utils/eda_tools.py`

This module provides reusable, interpretable, and production-ready utilities for exploratory data analysis (EDA) on structured tabular data. It supports deep learning and traditional pipelines alike.

---

## 🔧 Functions Overview

### Data Overview & Summary
- `overview_dataframe(df)` – Prints structure, types, duplicates, missing value stats
- `plot_missing_heatmap(df)` – Visual heatmap of missing values
- `plot_dtype_distribution(df)` – Column data type breakdown

### Label & Category Insights
- `plot_label_distribution(df)` – Anomaly vs. normal count barplot
- `plot_attack_category_distribution(df)` – Attack category frequency chart

### Feature Distributions & Correlations
- `plot_feature_hist(df, features)` – Histograms for numerical features
- `plot_correlation_heatmap(df)` – Pearson correlation heatmap

### Outlier Detection
- `detect_outliers_zscore(df)` – Detects columns with extreme z-score outliers

### Drift & Stability Checks
- `compare_feature_distributions(df1, df2, features)` – Visualize normal-normal drift
- `compare_ks_test(df1, df2, features)` – KS-test for detecting feature drift

### Auto Feature Selection
- `select_visualization_candidates(df)` – Suggests skewed/high-variance features to visualize

---

## 📌 Usage Guidelines

- Designed for tabular network traffic / log / structured data
- Works well with pandas DataFrames
- Especially helpful in anomaly detection (AE/VAE), classification, preprocessing

---

## ✅ Best Practices

- Run `overview_dataframe()` before modeling
- Use `plot_feature_hist()` to explore normalization/skew
- Run `compare_ks_test()` for concept drift analysis between two normal subsets