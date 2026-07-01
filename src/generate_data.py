"""
幾何ブラウン運動 (GBM) による株価ダミーデータ生成。

GBM の離散近似:
    S(t+1) = S(t) * exp((mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
    Z ~ N(0, 1),  dt = 1/252 (1営業日)
"""

import numpy as np
import pandas as pd

N_STOCKS = 50
N_DAYS = 500
START_DATE = "2023-01-01"


def generate(
    n_stocks: int = N_STOCKS,
    n_days: int = N_DAYS,
    start_date: str = START_DATE,
    seed: int = 42,
    output_path: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 1営業日 = 1/252 年
    dt = 1 / 252

    # 銘柄ごとに mu (期待収益率) と sigma (ボラティリティ) をランダムに設定
    # → 銘柄間に差をつけることで、最適化が特定の銘柄を選ぶようになる
    mus = rng.uniform(-0.05, 0.30, size=n_stocks)   # 年率期待収益率: -5% 〜 30%
    sigmas = rng.uniform(0.10, 0.50, size=n_stocks)  # 年率ボラティリティ: 10% 〜 50%

    # 株価の2次元配列を用意 (行=日付, 列=銘柄)
    prices = np.zeros((n_days, n_stocks))

    # 初日の株価を銘柄ごとにランダムに設定 (100 〜 1000)
    prices[0] = rng.uniform(100, 1000, size=n_stocks)

    # 2日目以降を GBM で1日ずつ計算
    for t in range(1, n_days):
        # 標準正規乱数 Z (銘柄ごとに独立)
        z = rng.standard_normal(n_stocks)

        # GBM の対数収益率
        # (mu - sigma^2/2) * dt : トレンド項
        # sigma * sqrt(dt) * Z  : ランダムブレ項
        log_return = (mus - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * z

        # 前日の株価に掛け算して今日の株価を求める
        prices[t] = prices[t - 1] * np.exp(log_return)

    # 銘柄名と日付のラベルを付けて DataFrame に変換
    tickers = [f"stock_{i:02d}" for i in range(n_stocks)]
    dates = pd.bdate_range(start=start_date, periods=n_days)  # 営業日のみ
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    df.index.name = "Date"

    if output_path:
        df.to_csv(output_path)
        print(f"Saved: {output_path}  ({n_days} days x {n_stocks} stocks)")

    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = generate(output_path="data/stock_prices.csv")

    plt.figure(figsize=(12, 5))
    for col in df.columns:
        plt.plot(df.index, df[col], linewidth=0.8, alpha=0.6)

    plt.title("GBM Simulated Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()
