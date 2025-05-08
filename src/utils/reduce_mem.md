# 🧮 Memory Optimization Utility

**Path**: `src/utils/reduce_mem.py`

This module contains a utility function to reduce memory usage of pandas DataFrames by downcasting numeric types.

---

## 🔧 Function: `reduce_mem_usage(df, use_float16=False)`

Automatically scans all numeric columns and attempts to convert their data types to smaller equivalents:

- `int64` → `int32` / `int16` / `int8`
- `float64` → `float32` / `float16` (optional)
- Leaves object/categorical/date columns unchanged

### Parameters:
- `df` (`pd.DataFrame`): Input DataFrame to optimize
- `use_float16` (`bool`): Whether to attempt float16 compression (default: False)

### Returns:
- `pd.DataFrame`: Memory-optimized DataFrame

---

## 📊 Benefits

- Lower memory footprint (especially in Colab/low-RAM environments)
- Faster data loading and model training for large datasets
- Helpful in preprocessing pipelines and batch training scripts

---

## 🧪 Example Usage

```python
from utils.reduce_mem import reduce_mem_usage

df = reduce_mem_usage(df)
```

✅ Memory usage stats will be printed before and after optimization.