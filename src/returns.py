"""
株価データから収益率・共分散行列を計算する。

ノートブックの calculate_return_rates() に準拠:
    収益率 = (運用終了時の株価 - 運用開始時の株価) / 運用開始時の株価
"""

import numpy as np
import pandas as pd


def calc_return_rates(prices: pd.DataFrame, num_days_operation: int) -> np.ndarray:
    """num_days_operation 日間運用したときの収益率を計算する。

    Parameters
    ----------
    prices              : 株価 DataFrame (行=営業日, 列=銘柄)
    num_days_operation  : 運用日数 (例: 20)

    Returns
    -------
    np.ndarray  shape = (len(prices) - num_days_operation, n_stocks)
    行 t は「t 日目に買って t+num_days_operation 日目に売ったときの収益率」
    """
    p = prices.to_numpy()
    return (p[num_days_operation:] - p[:-num_days_operation]) / p[:-num_days_operation]


def calc_mean_returns(return_rates: np.ndarray) -> np.ndarray:
    """各銘柄の平均収益率 (期待収益率) を返す。shape = (n_stocks,)"""
    return return_rates.mean(axis=0)


def calc_cov_matrix(return_rates: np.ndarray) -> np.ndarray:
    """銘柄間の共分散行列を返す。shape = (n_stocks, n_stocks)"""
    return np.cov(return_rates, rowvar=False)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_dummy

    prices = load_dummy("data/stock_prices.csv")
    rates = calc_return_rates(prices, num_days_operation=20)
    mean = calc_mean_returns(rates)
    cov = calc_cov_matrix(rates)

    print(f"収益率の形状: {rates.shape}")
    print(f"期待収益率 (先頭5銘柄):\n{mean[:5]}")
    print(f"\n共分散行列の形状: {cov.shape}")
