import pandas as pd

# Default UNSW-NB15 column names
default_col_names = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt",
    "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd",
    "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label"
]

def load_raw_data(path: str, col_names: list = None, verbose=True) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame with optional column names.

    Parameters:
        path (str): Path to the CSV file.
        col_names (list, optional): Column names to apply to the dataframe. If None, default_col_names is used.
        verbose (bool): If True, prints dataset shape.

    Returns:
        pd.DataFrame: Loaded raw data.
    """
    df = pd.read_csv(path, header=None, low_memory=False, dtype={1: str, 3: str, 47: str})
    if col_names is None:
        col_names = default_col_names
    df.columns = col_names
    if verbose:
        print(f"✅ Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    return df