# ポートフォリオ最適化：古典 vs 量子

幾何ブラウン運動 (GBM) で生成したダミー株価データを用いて、**古典ソルバー (scipy)** と**量子アニーリング (Fixstars Amplify)** のポートフォリオ最適化結果を比較する Streamlit アプリです。

## 概要

### 最適化問題

投資比率 $w_i$ (%) を決定変数として、以下の平均・分散モデルを最小化します。

```
f(w) = -r_p + (γ/2) * σ_p^2
```

- `r_p = Σ wi * ri` : ポートフォリオ期待収益率
- `σ_p^2 = w^T Σ w` : ポートフォリオ分散（リスク）
- 制約: Σwi = 100、0 ≤ wi ≤ 20

### 古典 vs 量子の違い

| | 古典 (scipy) | 量子 (Amplify) |
|---|---|---|
| 変数の型 | 連続値 | 整数 |
| 手法 | SLSQP 法 | QUBO → 量子アニーリング |
| トークン | 不要 | Fixstars Amplify AE |

## 機能

- γ（リスク回避度）と運用日数をスライダーで調整して古典ソルバーをリアルタイム実行
- 量子ソルバーの結果は事前計算済みを表示（γ=20、運用日数=20）
- 円グラフによるポートフォリオ構成の比較
- 数値比較テーブル（期待収益率・分散・銘柄数・実行時間）
- 収益率ヒストグラムによる評価期間での比較

## ディレクトリ構成

```
portfolio-optimization/
├── app.py                  # Streamlit アプリ
├── main.py                 # ローカル実行用
├── requirements.txt
├── assets/
│   └── quantum_result.json # 量子ソルバーの事前計算済み結果
├── data/
│   └── stock_prices.csv    # GBM で生成したダミー株価データ
└── src/
    ├── generate_data.py    # GBM による株価データ生成
    ├── data_loader.py      # データ読み込み
    ├── returns.py          # 収益率・共分散の計算
    ├── optimizer_classical.py  # scipy による最適化
    ├── optimizer_quantum.py    # Amplify による最適化
    ├── compare.py          # 円グラフ比較
    └── evaluate.py         # ヒストグラム評価
```

## ローカル実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# Streamlit アプリの起動
streamlit run app.py
```

量子ソルバーを実行する場合（事前計算の更新）：

```bash
export AMPLIFY_TOKEN="your_token"
python main.py
```

## 参考

- [Fixstars Amplify 公式チュートリアル](https://amplify.fixstars.com/ja/docs/)
- [幾何ブラウン運動 (Wikipedia)](https://ja.wikipedia.org/wiki/%E5%B9%BE%E4%BD%95%E3%83%96%E3%83%A9%E3%82%A6%E3%83%B3%E9%81%8B%E5%8B%95)
