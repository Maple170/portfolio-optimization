import pandas as pd


def load_dummy(csv_path: str) -> pd.DataFrame:
    """ダミー株価CSVを読み込む。
    列: 銘柄名, 行: 日付 (DatetimeIndex), 値: 終値
    """
    return pd.read_csv(csv_path, index_col="Date", parse_dates=True)
