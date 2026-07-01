"""
ポートフォリオ最適化 比較デモ

実行方法:
    export AMPLIFY_TOKEN="your_token"
    python main.py
"""

import sys
sys.path.insert(0, "src")

from generate_data import generate
from compare import plot_comparison
from evaluate import evaluate


if __name__ == "__main__":
    # ダミー株価データを生成
    print("株価データを生成中...")
    prices = generate(output_path="data/stock_prices.csv")
    print(f"生成完了: {prices.shape[0]}日 × {prices.shape[1]}銘柄\n")

    # 1. 古典 vs 量子の円グラフ比較
    print("=== ポートフォリオ構成の比較 ===")
    plot_comparison(prices)

    # 2. 未来データでの収益率ヒストグラム比較
    print("\n=== 収益率の評価 ===")
    evaluate(prices)
