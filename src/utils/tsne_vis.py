import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting
from sklearn.manifold import TSNE  # t-SNE algorithm for dimensionality reduction
from sklearn.preprocessing import StandardScaler  # Standardization utility

def plot_tsne_projection(
    df: pd.DataFrame,
    label_col: str = 'label',
    sample_size: int = 5000,
    perplexity: int = 30,
    random_state: int = 42,
    standardize: bool = True
) -> None:
    """
    Visualize high-dimensional features in 2D using t-SNE.

    Parameters:
        df (pd.DataFrame): DataFrame containing feature columns plus a binary label column.
        label_col (str): Column name for binary label (0=normal, 1=anomaly).
        sample_size (int): Total number of samples to plot (split proportional to classes).
        perplexity (int): t-SNE perplexity parameter (recommended range: 5-50).
        random_state (int): Seed for reproducibility.
        standardize (bool): If True, apply StandardScaler to features; set False if already scaled.

    Returns:
        None: Displays a scatter plot of the 2D t-SNE embedding.
    """
    # Ensure the label column exists
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' column not found in DataFrame")

    # Split into normal and anomaly subsets
    normal_df = df[df[label_col] == 0]
    anomaly_df = df[df[label_col] == 1]

    # Determine sample counts proportional to dataset composition
    total = len(df)
    normal_ratio = len(normal_df) / total
    n_normal = int(sample_size * normal_ratio)
    n_anomaly = sample_size - n_normal

    # Sample the data for plotting
    sampled_normal = normal_df.sample(n=n_normal, random_state=random_state)
    sampled_anomaly = anomaly_df.sample(n=n_anomaly, random_state=random_state)
    sampled_df = pd.concat([sampled_normal, sampled_anomaly])

    # Separate features and labels
    X = sampled_df.drop(columns=[label_col])
    y = sampled_df[label_col]

    # Optionally standardize features before t-SNE
    if standardize:
        X = StandardScaler().fit_transform(X)
    else:
        X = X.values if hasattr(X, 'values') else X

    # Compute 2D t-SNE embedding
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    # Plot t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.6
    )
    plt.title(f"t-SNE projection ({n_normal} normal + {n_anomaly} anomaly)")
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    cbar = plt.colorbar(scatter)
    cbar.set_label("Label (0=Normal, 1=Anomaly)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
