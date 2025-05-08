import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ks_2samp

# === OVERVIEW ===
def overview_dataframe(df):
    """
    Overview dataframe.
    """
    print("\n🔎 Basic Info:")
    print(df.info())
    print("\n📊 Numeric Summary:")
    display(df.describe(include=[np.number]))
    print("\n📝 Categorical Summary:")
    display(df.describe(include=['object']))
    print(f"\n🧱 Total Rows: {len(df)}")
    print(f"🔁 Duplicate Rows: {df.duplicated().sum()}")
    print(f"🚫 Columns with Missing Values: {(df.isnull().sum() > 0).sum()}")
    print("\n🧩 Missing Values Summary:")
    display(df.isnull().sum()[df.isnull().sum() > 0])


# === MISSING VALUES ===
def plot_missing_heatmap(df, figsize=(12, 6)):
    """
    Plot missing heatmap.
    """
    if df.isnull().sum().sum() == 0:
        print("✅ No missing values detected, skip heatmap.")
        return
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Value Heatmap")
    plt.tight_layout()
    plt.show()


# === DATA TYPES ===
def plot_dtype_distribution(df):
    """
    Plot dtype distribution.
    """
    dtype_counts = df.dtypes.value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=dtype_counts.index.astype(str), y=dtype_counts.values)
    plt.title("Data Type Distribution")
    plt.ylabel("Number of Columns")
    plt.tight_layout()
    plt.show()


# === LABEL DISTRIBUTION ===
def plot_label_distribution(df, label_col='label'):
    """
    Plot label distribution.
    """
    counts = df[label_col].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks([0, 1], ['Normal', 'Anomaly'])
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.tight_layout()
    plt.show()


def plot_attack_category_distribution(df, cat_col='attack_cat'):
    """
    Plot attack category distribution.
    """
    if cat_col not in df.columns:
        print(f"❌ Column {cat_col} not found in DataFrame.")
        return
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=cat_col, order=df[cat_col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Attack Category Distribution')
    plt.tight_layout()
    plt.show()


# === FEATURE DISTRIBUTIONS ===
def plot_feature_hist(df, features, bins=50):
    """
    Plot feature hist.
    """
    for feature in features:
        if feature not in df.columns:
            print(f"⚠️ Feature {feature} not found in DataFrame.")
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[feature], bins=bins, kde=False)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df, method='pearson', figsize=(12, 10)):
    """
    Plot correlation heatmap.
    """
    df = df.select_dtypes(include='number')
    corr = df.corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, fmt='.2f')
    plt.title(f'{method.title()} Correlation Heatmap')
    plt.tight_layout()
    plt.show()


# === OUTLIERS ===
def detect_outliers_zscore(df, threshold=3.0, top_k=10):
    """
    Detect outliers zscore.
    """
    df_numeric = df.select_dtypes(include=np.number)
    z_scores = np.abs((df_numeric - df_numeric.mean()) / df_numeric.std())
    outlier_flags = (z_scores > threshold)
    outlier_summary = outlier_flags.sum().sort_values(ascending=False)
    print(f"🔍 Z-Score > {threshold} — Number of outliers per column:")
    print(outlier_summary[outlier_summary > 0])
    top_outliers = outlier_summary[outlier_summary > 0].head(top_k)
    if not top_outliers.empty:
        plt.figure(figsize=(10, 4))
        sns.barplot(x=top_outliers.index, y=top_outliers.values)
        plt.xticks(rotation=45)
        plt.ylabel("# of Outliers")
        plt.title(f"Top {top_k} Columns with Z-Score Outliers (> {threshold})")
        plt.tight_layout()
        plt.show()
    return outlier_summary[outlier_summary > 0]


# === DISTRIBUTION STABILITY ===
"""
Distribution Stability Check (Advanced Utility)

Recommended Usage:
- Verify distribution consistency between training and test sets containing only normal samples (label == 0)
- Assess quality of new data before deployment by comparing it to historical data
- ⚠️ Not recommended for direct comparison between training and test sets in AE/VAE, 
  since the test set includes anomalies and is expected to differ in distribution
"""
def compare_feature_distributions(df_train, df_test, features, bins=50):
    """
    Compare feature distributions.
    """
    for feature in features:
        if feature not in df_train.columns or feature not in df_test.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df_train[feature], bins=bins, color='blue', label='Train', stat='density', kde=True, alpha=0.5)
        sns.histplot(df_test[feature], bins=bins, color='orange', label='Test', stat='density', kde=True, alpha=0.5)
        plt.title(f'Distribution: {feature}')
        plt.legend()
        plt.tight_layout()
        plt.show()


"""
Kolmogorov–Smirnov Test:
Evaluate whether feature distributions differ significantly between training and test sets (only normal samples)

Recommended Usage:
- Comparing normal samples across time periods or different systems
- Checking for concept drift before deploying new data
"""
def compare_ks_test(df_train, df_test, features, alpha=0.05):
    """
    Compare ks test.
    """
    print("Kolmogorov-Smirnov Test for Distribution Drift:")
    drift_results = []
    for feature in features:
        if feature not in df_train.columns or feature not in df_test.columns:
            continue
        stat, p = ks_2samp(df_train[feature].dropna(), df_test[feature].dropna())
        drift = '✅ Same Dist' if p > alpha else '⚠️ Drifted'
        drift_results.append((feature, round(stat, 3), round(p, 4), drift))
    df_drift = pd.DataFrame(drift_results, columns=["Feature", "KS_Statistic", "p_value", "Drift"])
    display(df_drift.sort_values("p_value"))


# === AUTO FEATURE SELECTION ===
def select_visualization_candidates(df, skew_thresh=3.0, var_thresh=None, range_thresh=None, top_k=10):
    """
    Select visualization candidates.
    """
    df_num = df.select_dtypes(include='number')
    skewed_cols = df_num.skew().abs()
    skewed = skewed_cols[skewed_cols > skew_thresh]

    var_cols = df_num.var()
    ranged_cols = df_num.max() - df_num.min()

    selected_parts = []

    if var_thresh:
        selected_parts.append(var_cols[var_cols > var_thresh])
    if range_thresh:
        selected_parts.append(ranged_cols[ranged_cols > range_thresh])
    selected_parts.append(skewed)

    selected_cols = pd.concat(selected_parts)
    selected_cols = selected_cols[~selected_cols.index.duplicated(keep='first')]
    return selected_cols.sort_values(ascending=False).head(top_k).index.tolist()