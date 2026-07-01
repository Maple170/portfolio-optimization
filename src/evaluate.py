"""
古典ソルバーと量子ソルバーのポートフォリオを未来データで評価し、
収益率のヒストグラムで比較する。

ノートブック cell 20-22 に相当。
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, "src")

from data_loader import load_dummy
from returns import calc_return_rates
from optimizer_classical import optimize as classical
from optimizer_quantum import optimize as quantum


def calc_portfolio_return_rates(
    portfolio: dict,
    future_prices,
    num_days_operation: int,
) -> np.ndarray:
    """未来データに対してポートフォリオの収益率を計算する。"""
    stock_names = list(future_prices.columns)

    # 各銘柄の投資比率の配列に変換
    weights = np.array([
        portfolio.get(name, 0) / 100 for name in stock_names
    ])

    # 未来データの収益率 (行=運用開始日, 列=銘柄)
    return_rates = calc_return_rates(future_prices, num_days_operation)

    # 各運用開始日ごとのポートフォリオ全体の収益率
    return (weights * return_rates).sum(axis=1)


def evaluate(prices, train_ratio=0.6, num_days_operation=20, save_path="output/return_histogram.png"):
    """
    データを訓練期間と評価期間に分割する。
    訓練期間でポートフォリオを最適化し、評価期間で収益率を比較する。

    train_ratio: 訓練データの割合 (残りが評価用)
    """
    split = int(len(prices) * train_ratio)
    train = prices.iloc[:split]   # 最適化に使う過去データ
    future = prices.iloc[split:]  # 評価に使う未来データ

    print(f"訓練期間: {train.index[0].date()} 〜 {train.index[-1].date()} ({len(train)}日)")
    print(f"評価期間: {future.index[0].date()} 〜 {future.index[-1].date()} ({len(future)}日)\n")

    print("最適化中 (古典)...")
    p_c, _, _ = classical(train, num_days_operation)

    print("最適化中 (量子)...")
    p_q, _, _ = quantum(train, num_days_operation)

    # 評価期間で各ポートフォリオの収益率を計算
    r_c = calc_portfolio_return_rates(p_c, future, num_days_operation)
    r_q = calc_portfolio_return_rates(p_q, future, num_days_operation)

    # 数値比較
    print(f"\n=== 評価結果 ===")
    print(f"{'':20} {'Classical':>12} {'Quantum':>12}")
    print(f"{'平均収益率':20} {np.mean(r_c)*100:>11.2f}% {np.mean(r_q)*100:>11.2f}%")
    print(f"{'標準偏差':20} {np.std(r_c)*100:>11.2f}% {np.std(r_q)*100:>11.2f}%")
    print(f"{'最小収益率':20} {np.min(r_c)*100:>11.2f}% {np.min(r_q)*100:>11.2f}%")
    print(f"{'最大収益率':20} {np.max(r_c)*100:>11.2f}% {np.max(r_q)*100:>11.2f}%")

    # ヒストグラムで比較
    bins = np.linspace(
        min(r_c.min(), r_q.min()) * 100 - 1,
        max(r_c.max(), r_q.max()) * 100 + 1,
        40,
    )

    plt.figure(figsize=(10, 5))
    plt.hist(r_c * 100, bins=bins, label="Classical", color="royalblue", alpha=0.7)
    plt.hist(r_q * 100, bins=bins, label="Quantum",   color="coral",     alpha=0.7)
    plt.axvline(np.mean(r_c) * 100, color="royalblue", linestyle="--", linewidth=1.5)
    plt.axvline(np.mean(r_q) * 100, color="coral",     linestyle="--", linewidth=1.5)
    plt.xlabel("Return Rate (%)")
    plt.ylabel("Frequency")
    plt.title("Portfolio Return Rate Comparison (Classical vs Quantum)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存: {save_path}")
    plt.show()


if __name__ == "__main__":
    prices = load_dummy("data/stock_prices.csv")
    evaluate(prices)
