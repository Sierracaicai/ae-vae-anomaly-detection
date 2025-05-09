# 🛠️ Data Preprocessing Module

`src/utils/preprocess.py`

This document describes the functionality and usage of the **preprocess** module, which implements an end-to-end pipeline for cleaning, encoding, transforming, and saving network traffic data.

---

## 📦 Module Overview

* **Purpose**: Convert raw CSV data into a clean, numeric feature matrix suitable for anomaly detection models (AE, VAE, Isolation Forest, etc.).
* **Key Steps**:

  1. Load raw data with default or custom column names (`load_raw_data`)
  2. Clean: remove duplicates, merge rare categories, and drop unneeded columns (`clean_data`)
  3. Encode: one-hot encode specified categorical features (`encode_categoricals`)
  4. Transform: log1p on skewed numeric columns
  5. Scale: MinMax or Standard scaling on all numeric columns (`scale_features`)
  6. Filter: drop highly correlated numeric features (`drop_highly_correlated_features`)
  7. Optimize: reduce DataFrame memory usage (`reduce_mem_usage`)
  8. Save: write cleaned DataFrame to CSV or pickle

---

## 🧩 File Structure

```
src/utils/
├── load_data.py           # Raw CSV loader with default UNSW-NB15 column names
├── reduce_mem.py          # Downcast numeric columns to save memory
└── preprocess.py          # Full preprocessing pipeline
```

---

## 🔧 Functions

### `group_rare_categories`

```python
group_rare_categories(
    df: pd.DataFrame,
    categorical_cols: list,
    freq_thresh: float = 0.01,
    anomaly_col: str = 'label',
    class_values: tuple = (0, 1)
) -> pd.DataFrame
```

* **Description**: Merge low-frequency categories into `'Other'`, but retain categories that appear more often in anomalies than normals.
* **Parameters**:

  * `df`: Input DataFrame
  * `categorical_cols`: List of columns to process
  * `freq_thresh`: Overall frequency threshold
  * `anomaly_col`: Label column name
  * `class_values`: (normal, anomaly) values
* **Returns**: Modified DataFrame

### `clean_data`

```python
def clean_data(
    df: pd.DataFrame,
    drop_cols: list = None,
    drop_duplicates: bool = True,
    group_rare: bool = False,
    categorical_cols: list = None,
    rare_freq_thresh: float = 0.01
) -> pd.DataFrame
```

* **Description**: Initial cleaning step: drop duplicates, optional rare-category grouping, drop unwanted columns.
* **Parameters**:

  * `drop_cols`: Columns to drop (defaults to internal `recommended_drop_cols`)
  * `drop_duplicates`: Remove duplicate rows
  * `group_rare`: Enable rare-category grouping
  * `categorical_cols`: Columns to consider for grouping
  * `rare_freq_thresh`: Frequency threshold for grouping
* **Returns**: Cleaned DataFrame

### `encode_categoricals`

```python
def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list
) -> pd.DataFrame
```

* **Description**: One-hot encode specified categorical columns.
* **Parameters**:

  * `categorical_cols`: List of columns to encode
* **Returns**: DataFrame with original columns replaced by one-hot features

### `scale_features`

```python
def scale_features(
    df: pd.DataFrame,
    method: str = 'minmax'
) -> pd.DataFrame
```

* **Description**: Scale numeric features using MinMax or Standard scaler.
* **Parameters**:

  * `method`: `'minmax'` or `'standard'`
* **Returns**: DataFrame with scaled features

### `drop_highly_correlated_features`

```python
def drop_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95
) -> (pd.DataFrame, list)
```

* **Description**: Remove numeric features whose pairwise correlation exceeds the threshold.
* **Parameters**:

  * `threshold`: Absolute correlation cutoff
* **Returns**:

  * Tuple of cleaned DataFrame and list of dropped columns

### `preprocess_pipeline`

```python
def preprocess_pipeline(
    path: str,
    col_names: list = None,
    drop_cols: list = None,
    drop_duplicates: bool = True,
    group_rare: bool = False,
    rare_freq_thresh: float = 0.01,
    categorical_cols: list = None,
    scale_method: str = 'minmax',
    encode_categorical: bool = True,
    log_transform_cols: list = None,
    drop_corr_features: bool = False,
    save_path: str = None,
    save_as_pickle: bool = False
) -> pd.DataFrame
```

* **Description**: Orchestrates the entire preprocessing workflow.
* **Parameters**:

  * `path`: Path to raw CSV file
  * `col_names`: Optional custom column names
  * Other flags and lists corresponding to each processing step
  * `save_path`: If provided, writes output to this location
  * `save_as_pickle`: Write as pickle instead of CSV
* **Returns**: Fully processed DataFrame

---

## 🚀 Usage Example

```python
from utils.preprocess import preprocess_pipeline

processed_df = preprocess_pipeline(
    path="data/raw/UNSW-NB15_1.csv",
    categorical_cols=["proto","state","service"],
    group_rare=True,
    log_transform_cols=["dur","sbytes","sloss"],
    drop_corr_features=False,
    save_path="data/processed/cleaned.csv"
)
```

---

## 📄 Notes

* **Do not commit large processed CSVs** to the repository; use `.gitignore` to exclude `data/processed/*`.
* Adjust `rare_freq_thresh` based on EDA insights to balance dimensionality vs. information loss.
* Ensure `load_data.py` and `reduce_mem.py` are also present in `src/utils/` for full pipeline functionality.

```
```
