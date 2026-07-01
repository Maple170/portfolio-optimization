"""
Fixstars Amplify (量子アニーリング) を用いたポートフォリオ最適化。

ノートブックの optimize_portfolio() をほぼそのまま移植。
古典ソルバー (optimizer_classical.py) との違い:
    - 変数 w_i が整数 (0〜max_w) → より現実的な投資比率
    - QUBO 形式に変換して量子アニーリングマシンで解く
"""

import datetime
import os
import numpy as np
import pandas as pd
import amplify

from returns import calc_return_rates, calc_mean_returns, calc_cov_matrix

def optimize(
    historical_data: pd.DataFrame,
    num_days_operation: int,
    gamma: float = 20,
    max_w: int = 20,
    time_limit: datetime.timedelta = datetime.timedelta(seconds=5),
) -> tuple[dict[str, int], float, float]:
    """
    Parameters
    ----------
    historical_data    : 過去の株価 DataFrame
    num_days_operation : 運用日数
    gamma              : リスク回避度 (大きいほどリスク重視)
    max_w              : 1銘柄への投資上限 (%)
    time_limit         : Amplify AE の実行時間制限

    Returns
    -------
    portfolio   : {銘柄名: 投資比率(%)} の辞書 (比率 > 0 の銘柄のみ)
    return_rate : ポートフォリオの期待収益率
    variance    : ポートフォリオの分散 (リスク)
    """
    stock_names = list(historical_data.columns)

    # 投資比率を表す整数変数 w_i (0 以上 max_w 以下)
    gen = amplify.VariableGenerator()
    w = gen.array("Integer", len(stock_names), bounds=(0, max_w))

    # 制約条件: w の総和 = 100
    constraint = amplify.equal_to(w.sum(), 100)

    # w (%) を実数の投資比率に変換
    w_ratio = w / 100

    # 収益率・共分散の計算
    return_rates = calc_return_rates(historical_data, num_days_operation)
    mean_r = calc_mean_returns(return_rates)
    cov = calc_cov_matrix(return_rates)

    # ポートフォリオの期待収益率 (各銘柄の収益率 × 投資比率の合計)
    portfolio_return = (w_ratio * mean_r).sum()

    # ポートフォリオのリスク (収益率の分散)
    portfolio_variance = w_ratio @ cov @ w_ratio  # type: ignore

    # 目的関数: 収益率を最大化しつつリスクを最小化
    objective = -portfolio_return + 0.5 * gamma * portfolio_variance

    # 最適化モデルを作成
    model = amplify.Model(objective, constraint)

    # Amplify AE クライアントの設定
    token = os.environ.get("AMPLIFY_TOKEN", "")
    if not token:
        raise RuntimeError("AMPLIFY_TOKEN が設定されていません")
    client = amplify.AmplifyAEClient()
    client.parameters.time_limit_ms = time_limit
    client.token = token

    # 最適化を実行
    result = amplify.solve(model, client)

    if len(result) == 0:
        raise RuntimeError("実行可能解が見つかりませんでした")

    # 最良解から投資比率を取得
    w_values = w.evaluate(result.best.values)

    # 投資比率 > 0 の銘柄のみ辞書に格納
    portfolio = {
        name: int(w_val)
        for name, w_val in zip(stock_names, w_values)
        if w_val > 0
    }

    # 得られたポートフォリオの収益率とリスクを計算
    return_rate = float(portfolio_return.evaluate(result.best.values))
    variance = float(portfolio_variance.evaluate(result.best.values))

    return portfolio, return_rate, variance


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    sys.path.insert(0, "src")
    from data_loader import load_dummy

    prices = load_dummy("data/stock_prices.csv")
    portfolio, return_rate, variance = optimize(prices, num_days_operation=20)

    print("=== 最適化結果 (量子) ===")
    for name, w in sorted(portfolio.items(), key=lambda x: -x[1]):
        print(f"  {name}: {w}%")
    print(f"\n期待収益率: {return_rate * 100:.2f}%")
    print(f"リスク(分散): {variance:.6f}")

    plt.figure(figsize=(8, 8))
    plt.pie(
        list(portfolio.values()),
        labels=list(portfolio.keys()),
        autopct="%.1f%%",
        startangle=90,
    )
    plt.title(f"Portfolio (quantum)\nReturn: {return_rate*100:.2f}%  Variance: {variance:.6f}")
    plt.tight_layout()
    plt.show()
