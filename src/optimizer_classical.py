"""
scipy を用いた古典的ポートフォリオ最適化。

ノートブックの optimize_portfolio() と同じ目的関数・制約条件を使う:
    目的関数: f(w) = -r_p + (gamma/2) * sigma_p^2
    制約条件: sum(w_i) = 100
    変数範囲: 0 <= w_i <= 20
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from returns import calc_return_rates, calc_mean_returns, calc_cov_matrix


def optimize(
    historical_data: pd.DataFrame,
    num_days_operation: int,
    gamma: float = 20,
    max_w: int = 20,
) -> tuple[dict[str, float], float, float]:
    """
    Parameters
    ----------
    historical_data    : 過去の株価 DataFrame
    num_days_operation : 運用日数
    gamma              : リスク回避度 (大きいほどリスク重視)
    max_w              : 1銘柄への投資上限 (%)

    Returns
    -------
    portfolio   : {銘柄名: 投資比率(%)} の辞書 (比率 > 0 の銘柄のみ)
    return_rate : ポートフォリオの期待収益率
    variance    : ポートフォリオの分散 (リスク)
    """
    stock_names = list(historical_data.columns)
    n = len(stock_names)

    return_rates = calc_return_rates(historical_data, num_days_operation)
    mean_r = calc_mean_returns(return_rates)   # shape (n,)
    cov = calc_cov_matrix(return_rates)        # shape (n, n)

    def objective(w):
        w_ratio = w / 100
        portfolio_return = np.dot(w_ratio, mean_r)
        portfolio_variance = w_ratio @ cov @ w_ratio
        return -portfolio_return + 0.5 * gamma * portfolio_variance

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 100}
    bounds = [(0, max_w)] * n
    w0 = np.full(n, 100 / n)  # 均等配分を初期値にする

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000})

    if not result.success:
        raise RuntimeError(f"最適化失敗: {result.message}")

    w_values = result.x
    portfolio = {
        name: round(float(w), 2)
        for name, w in zip(stock_names, w_values)
        if w > 0.5
    }

    w_ratio = w_values / 100
    return_rate = float(np.dot(w_ratio, mean_r))
    variance = float(w_ratio @ cov @ w_ratio)

    return portfolio, return_rate, variance


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.insert(0, "src")
    from data_loader import load_dummy

    prices = load_dummy("data/stock_prices.csv")
    portfolio, return_rate, variance = optimize(prices, num_days_operation=20)

    print("=== 最適化結果 ===")
    for name, w in sorted(portfolio.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w:.1f}%")
    print(f"\n期待収益率: {return_rate * 100:.2f}%")
    print(f"リスク(分散): {variance:.6f}")

    labels = list(portfolio.keys())
    values = list(portfolio.values())

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct="%.1f%%", startangle=90)
    plt.title(f"Portfolio (classical)\nReturn: {return_rate*100:.2f}%  Variance: {variance:.6f}")
    plt.tight_layout()
    plt.show()
