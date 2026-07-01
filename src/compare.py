"""
古典ソルバーと量子ソルバーの結果を並べて比較する。
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
sys.path.insert(0, "src")

from data_loader import load_dummy
from optimizer_classical import optimize as classical
from optimizer_quantum import optimize as quantum


def plot_comparison(prices, num_days_operation=20, save_path="output/portfolio_comparison.png"):
    p_c, r_c, v_c = classical(prices, num_days_operation)
    p_q, r_q, v_q = quantum(prices, num_days_operation)

    # 両方に登場する全銘柄に共通の色を割り当てる
    all_stocks = sorted(set(p_c.keys()) | set(p_q.keys()))
    colors = cm.tab20(np.linspace(0, 1, len(all_stocks)))
    color_map = {stock: colors[i] for i, stock in enumerate(all_stocks)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, portfolio, return_rate, variance, title in [
        (axes[0], p_c, r_c, v_c, "Classical (scipy)"),
        (axes[1], p_q, r_q, v_q, "Quantum (Amplify)"),
    ]:
        labels = [s.replace("stock_", "") for s in portfolio.keys()]
        values = list(portfolio.values())
        stock_colors = [color_map[s] for s in portfolio.keys()]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%.1f%%",
            startangle=90,
            colors=stock_colors,
            labeldistance=0.6,   # ラベルを内側に
            pctdistance=0.35,    # パーセントをさらに内側に
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )

        # ラベルのフォントサイズを調整
        for text in texts:
            text.set_fontsize(8)
            text.set_horizontalalignment("center")
        for autotext in autotexts:
            autotext.set_fontsize(7)

        ax.set_title(
            f"{title}\nReturn: {return_rate*100:.2f}%  Variance: {variance:.6f}  Stocks: {len(portfolio)}",
            fontsize=11,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"保存: {save_path}")
    plt.show()

    # 数値比較
    print("\n=== ポートフォリオ構成の比較 ===")
    print(f"{'':20} {'Classical':>12} {'Quantum':>12}")
    print(f"{'期待収益率':20} {r_c*100:>11.2f}% {r_q*100:>11.2f}%")
    print(f"{'分散(リスク)':20} {v_c:>12.6f} {v_q:>12.6f}")
    print(f"{'選択銘柄数':20} {len(p_c):>12} {len(p_q):>12}")

    return p_c, r_c, v_c, p_q, r_q, v_q


if __name__ == "__main__":
    prices = load_dummy("data/stock_prices.csv")
    plot_comparison(prices)
