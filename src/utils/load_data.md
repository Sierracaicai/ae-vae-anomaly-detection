# 📂 Data Loader Module

**Path**: `src/utils/load_data.py`

This module handles standardized loading of the raw UNSW-NB15 dataset.

---

## 🔧 Functions

### `load_raw_data(path, col_names=None, verbose=True)`

Load raw CSV into a DataFrame, optionally applying custom column names.  
If `col_names` is not specified, the built-in `default_col_names` will be used.

#### Parameters:
- `path` (str): Full path to CSV file
- `col_names` (list, optional): Custom list of column names
- `verbose` (bool): Print dataset shape summary (default: True)

#### Returns:
- `pd.DataFrame`: Loaded dataset

---

## 📌 Built-in Column Names

`default_col_names` contains the 49 feature names + `attack_cat` + `label`, matching the UNSW-NB15 dataset documentation.

No need to manually provide headers during raw loading.

---

## 🧪 Example Usage

```python
from utils.load_data import load_raw_data

df = load_raw_data("data/raw/UNSW-NB15_1.csv")
```

> ✅ This function is especially useful during early EDA without triggering preprocessing.