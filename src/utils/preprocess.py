# src/utils/preprocess.py
import os
import pandas as pd  # For DataFrame operations
import numpy as np   # For numerical computations
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from utils.load_data import load_raw_data
from utils.reduce_mem import reduce_mem_usage

# Default list of columns to drop during cleaning
recommended_drop_cols = [
    "srcip", "dstip", "sport", "dsport", "Stime", "Ltime",
    "is_sm_ips_ports", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "attack_cat"
]

def group_rare_categories(
    df: pd.DataFrame,
    categorical_cols: list,
    freq_thresh: float = 0.01,
    anomaly_col: str = 'label',
    class_values: tuple = (0, 1)
) -> pd.DataFrame:
    """
    Merge infrequent categories into an 'Other' label, while preserving categories
    that occur more frequently in anomaly samples than in normal ones.

    This helps reduce one-hot dimensionality by combining truly insignificant categories,
    but retains rare categories that may signal anomalies.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    categorical_cols : list
        List of column names to evaluate and group.
    freq_thresh : float, default=0.01
        Overall frequency threshold below which values are considered 'rare'.
    anomaly_col : str, default='label'
        Column name for binary anomaly labels (0=normal, 1=anomaly).
    class_values : tuple, default=(0,1)
        Integer values representing the normal and anomaly classes.

    Returns:
    --------
    pd.DataFrame
        DataFrame with rare, non-informative categories replaced by 'Other'.
    """
    df = df.copy()

    normal_class, anomaly_class = class_values

    for col in categorical_cols:
        # Calculate overall value frequencies
        vc = df[col].value_counts(normalize=True)
        rare_vals = vc[vc < freq_thresh].index.tolist()
        if not rare_vals:
            continue
        # Frequency in normal vs. anomaly classes
        normal_freq = df[df[anomaly_col] == normal_class][col].value_counts(normalize=True)
        anomaly_freq = df[df[anomaly_col] == anomaly_class][col].value_counts(normalize=True)
        # Determine which rare values to group (if not more frequent in anomalies)
        to_group = [v for v in rare_vals if anomaly_freq.get(v, 0) <= normal_freq.get(v, 0)]
        if to_group:
            mask = df[col].isin(to_group)
            df.loc[mask, col] = 'Other'
    return df


def clean_data(
    df: pd.DataFrame,
    drop_cols: list = None,
    drop_duplicates: bool = True,
    group_rare: bool = False,
    categorical_cols: list = None,
    rare_freq_thresh: float = 0.01
) -> pd.DataFrame:
    """
    Perform initial data cleaning steps:
      1. Remove duplicate rows if desired.
      2. Merge rare categories using group_rare_categories if enabled.
      3. Drop specified columns that are not useful for modeling.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame loaded from source.
    drop_cols : list, optional
        Columns to drop; defaults to recommended_drop_cols if None.
    drop_duplicates : bool, default=True
        Whether to remove exact duplicate rows.
    group_rare : bool, default=False
        Whether to merge rare categories into 'Other'.
    categorical_cols : list, optional
        Columns to consider for rare category grouping.
    rare_freq_thresh : float, default=0.01
        Threshold for considering a category as rare.

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame ready for further processing.
    """
    df = df.copy()

    if drop_duplicates:
        before = len(df)
        df = df.loc[~df.duplicated()].copy()
        print(f"🗑 Removed {before - len(df)} duplicate rows")
    if group_rare and categorical_cols:
        df = group_rare_categories(df, categorical_cols, freq_thresh=rare_freq_thresh)
    if drop_cols is None:
        drop_cols = recommended_drop_cols
    df = df.drop(columns=drop_cols, errors='ignore')
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list
) -> pd.DataFrame:
    """
    Convert categorical columns into one-hot encoded features.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing categorical columns.
    categorical_cols : list
        List of column names to encode.

    Returns:
    --------
    pd.DataFrame
        DataFrame with original categorical columns replaced by one-hot columns.
    """
    if not categorical_cols:
        return df
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    arr = encoder.fit_transform(df[categorical_cols])
    cols = encoder.get_feature_names_out(categorical_cols)
    df_enc = pd.DataFrame(arr, columns=cols, index=df.index)
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, df_enc], axis=1)
    print(f"🧠 Encoded {len(categorical_cols)} columns into {len(cols)} features")
    return df


def scale_features(
    df: pd.DataFrame,
    method: str = 'minmax'
) -> pd.DataFrame:
    """
    Scale all numeric features using specified method.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with numeric columns to scale.
    method : str, default='minmax'
        Scaling method: 'minmax' for MinMaxScaler, otherwise StandardScaler.

    Returns:
    --------
    pd.DataFrame
        DataFrame with scaled numeric features.
    """
    num_cols = df.select_dtypes(include='number').columns
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"📏 Scaled {len(num_cols)} numeric columns using {method}")
    return df


def drop_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95
) -> tuple:
    """
    Remove numeric features with correlation above a threshold to reduce multicollinearity.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing numeric features.
    threshold : float, default=0.95
        Correlation threshold; features with abs(corr) > threshold are dropped.

    Returns:
    --------
    tuple:
        - DataFrame with dropped features removed.
        - List of dropped feature names.
    """
    corr_matrix = df.select_dtypes(include='number').corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"🧹 Dropped {len(to_drop)} highly correlated features: {to_drop}")
    return df.drop(columns=to_drop), to_drop


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
) -> pd.DataFrame:
    """
    Complete data preprocessing pipeline combining cleaning, encoding, scaling,
    correlation filtering, memory optimization, and file saving.

    Parameters:
    -----------
    path : str
        File path to raw CSV data.
    col_names : list, optional
        Column names to assign when loading data.
    drop_cols : list, optional
        Columns to drop during cleaning.
    drop_duplicates : bool, default=True
        Whether to remove duplicate rows.
    group_rare : bool, default=False
        Whether to group rare categories into 'Other'.
    rare_freq_thresh : float, default=0.01
        Frequency threshold for rare category grouping.
    categorical_cols : list, optional
        Columns to one-hot encode.
    scale_method : str, default='minmax'
        Scaling method for numeric features.
    encode_categorical : bool, default=True
        Whether to apply one-hot encoding to categorical columns.
    log_transform_cols : list, optional
        Numeric columns to apply log1p transform.
    drop_corr_features : bool, default=False
        Whether to drop highly correlated numeric features.
    save_path : str, optional
        File path to save processed data (CSV or pickle).
    save_as_pickle : bool, default=False
        Save as pickle if True, else CSV.

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame ready for modeling.
    """
    # Step 1: Load raw data
    df = load_raw_data(path, col_names)
    # Step 2: Initial cleaning
    df = clean_data(
        df,
        drop_cols=drop_cols,
        drop_duplicates=drop_duplicates,
        group_rare=group_rare,
        categorical_cols=categorical_cols,
        rare_freq_thresh=rare_freq_thresh
    )
    # Step 3: One-hot encoding
    if encode_categorical and categorical_cols:
        df = encode_categoricals(df, categorical_cols)
    # Step 4: Log-transform selected columns
    if log_transform_cols:
        for col in log_transform_cols:
            if col in df:
                df[col] = np.log1p(df[col])
                print(f"🔁 Log1p applied to {col}")
    # Step 5: Scale numeric features
    df = scale_features(df, method=scale_method)
    # Step 6: Drop highly correlated features
    if drop_corr_features:
        df, dropped = drop_highly_correlated_features(df)
    # Step 7: Memory optimization
    df = reduce_mem_usage(df)
    # Step 8: Save processed data
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_as_pickle:
            df.to_pickle(save_path)
        else:
            df.to_csv(save_path, index=False)
        print(f"💾 Saved processed data to {save_path}")
    return df

