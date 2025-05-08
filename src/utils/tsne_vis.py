import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def plot_tsne_projection(df, label_col='label', sample_size=5000, perplexity=30, random_state=42, standardize=True):
    """
    Visualize high-dimensional data with t-SNE in 2D space.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and a label column.
        label_col (str): Name of the binary label column (0 = normal, 1 = anomaly).
        sample_size (int): Number of total samples to visualize (split equally by class).
        perplexity (int): Perplexity parameter for t-SNE (typically between 5–50).
        random_state (int): Random seed for reproducibility.
        standardize (bool): Whether to apply StandardScaler before t-SNE. Set to False if data is pre-scaled.
    """
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column not found in DataFrame")

    normal_df = df[df[label_col] == 0]
    anomaly_df = df[df[label_col] == 1]

    n_normal = min(sample_size // 2, len(normal_df))
    n_anomaly = min(sample_size // 2, len(anomaly_df))

    sampled_df = pd.concat([
        normal_df.sample(n=n_normal, random_state=random_state),
        anomaly_df.sample(n=n_anomaly, random_state=random_state)
    ])

    X = sampled_df.drop(columns=[label_col])
    y = sampled_df[label_col]

    if standardize:
        X = StandardScaler().fit_transform(X)
    else:
        X = X.values if hasattr(X, 'values') else X

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.title(f't-SNE projection ({n_normal} normal + {n_anomaly} anomaly)')
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    cbar = plt.colorbar(scatter)
    cbar.set_label("Label (0: Normal, 1: Anomaly)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()