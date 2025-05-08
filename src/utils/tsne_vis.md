# 📌 tsne_vis.py

This module provides a reusable t-SNE projection utility for visualizing high-dimensional feature data.

## 🔍 Function: `plot_tsne_projection`

### Purpose:
Visualize normal vs anomaly samples in 2D space using t-SNE. Useful during:
- EDA phase (e.g., after preprocessing)
- AE/VAE bottleneck visualization
- Anomaly separation inspection

### Signature:
```python
plot_tsne_projection(
    df,
    label_col='label',
    sample_size=5000,
    perplexity=30,
    random_state=42,
    standardize=True
)
```

### Parameters:
- **df**: DataFrame that includes features + label
- **label_col**: Name of binary label column
- **sample_size**: Number of total samples (split between classes)
- **standardize**: Whether to apply StandardScaler before t-SNE (disable if already scaled)

### Example:

```python
from utils.tsne_vis import plot_tsne_projection

# For preprocessed data:
plot_tsne_projection(processed_df, label_col='label', standardize=False)

# For raw feature data:
plot_tsne_projection(raw_df, label_col='label', standardize=True)
```