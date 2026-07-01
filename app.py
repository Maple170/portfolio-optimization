"""
ポートフォリオ最適化 比較アプリ (Streamlit)

実行方法:
    export AMPLIFY_TOKEN="your_token"
    streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

sys.path.insert(0, "src")
from generate_data import generate
from optimizer_classical import optimize as classical
from optimizer_quantum import optimize as quantum
from evaluate import calc_portfolio_return_rates

try:
    AMPLIFY_TOKEN = st.secrets["AMPLIFY_TOKEN"]
except Exception:
    AMPLIFY_TOKEN = os.environ.get("AMPLIFY_TOKEN", "")

st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 300px; max-width: 300px; }
[data-testid="stSidebarResizeHandle"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.title("ポートフォリオ最適化: 古典 vs 量子")
st.caption("幾何ブラウン運動で生成したダミー株価データを用いて、古典ソルバー (scipy) と量子アニーリング (Amplify) の最適化結果を比較します。")

# ── サイドバー ──────────────────────────────────────
with st.sidebar:
    st.header("パラメータ設定")

    gamma = st.slider("γ (リスク回避度)", min_value=1, max_value=50, value=20,
                      help="大きいほどリスク低減を重視。小さいほどリターン重視。")
    num_days = st.slider("運用日数", min_value=5, max_value=60, value=20,
                         help="ポートフォリオを何営業日保有するか")
    run = st.button("最適化を実行", type="primary", use_container_width=True)

# ── データ生成 ──────────────────────────────────────
@st.cache_data
def get_prices():
    return generate(seed=42)

prices = get_prices()
split = int(len(prices) * 0.6)
train, future = prices.iloc[:split], prices.iloc[split:]

# ── 株価チャート ────────────────────────────────────
st.subheader("生成された株価データ")
st.line_chart(prices, height=250)

# ── 最適化 ─────────────────────────────────────────
if run:
    if not AMPLIFY_TOKEN:
        st.error("AMPLIFY_TOKEN が設定されていません。")
        st.stop()

    os.environ["AMPLIFY_TOKEN"] = AMPLIFY_TOKEN

    with st.spinner("最適化中... (約10秒)"):
        t0 = time.time()
        p_c, r_c, v_c = classical(train, num_days, gamma=gamma)
        t_c = time.time() - t0

        t0 = time.time()
        p_q, r_q, v_q = quantum(train, num_days, gamma=gamma)
        t_q = time.time() - t0

    r_c_eval = calc_portfolio_return_rates(p_c, future, num_days)
    r_q_eval = calc_portfolio_return_rates(p_q, future, num_days)

    # ── 円グラフ比較 ────────────────────────────────
    st.subheader("ポートフォリオ構成")

    all_stocks = sorted(set(p_c.keys()) | set(p_q.keys()))
    colors = cm.tab20(np.linspace(0, 1, max(len(all_stocks), 1)))
    color_map = {s: colors[i] for i, s in enumerate(all_stocks)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, portfolio, return_rate, variance, title in [
        (axes[0], p_c, r_c, v_c, "Classical (scipy)"),
        (axes[1], p_q, r_q, v_q, "Quantum (Amplify)"),
    ]:
        labels = [s.replace("stock_", "") for s in portfolio.keys()]
        values = list(portfolio.values())
        stock_colors = [color_map[s] for s in portfolio.keys()]
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%.1f%%", startangle=90,
            colors=stock_colors, labeldistance=0.6, pctdistance=0.35,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        for text in texts:
            text.set_fontsize(8)
            text.set_horizontalalignment("center")
        for autotext in autotexts:
            autotext.set_fontsize(7)
        ax.set_title(f"{title}\nReturn: {return_rate*100:.2f}%  Variance: {variance:.6f}  Stocks: {len(portfolio)}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── 数値比較テーブル ────────────────────────────
    st.subheader("数値比較")
    table = pd.DataFrame({
        "Classical": [f"{r_c*100:.2f}%", f"{v_c:.6f}", len(p_c), f"{t_c:.1f}秒"],
        "Quantum":   [f"{r_q*100:.2f}%", f"{v_q:.6f}", len(p_q), f"{t_q:.1f}秒"],
    }, index=["期待収益率", "分散(リスク)", "選択銘柄数", "実行時間"])
    st.table(table)

    # ── 収益率ヒストグラム ──────────────────────────
    st.subheader("収益率の分布 (評価期間)")

    bins = np.linspace(
        min(r_c_eval.min(), r_q_eval.min()) * 100 - 1,
        max(r_c_eval.max(), r_q_eval.max()) * 100 + 1,
        40,
    )
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(r_c_eval * 100, bins=bins, label="Classical", color="royalblue", alpha=0.7)
    ax2.hist(r_q_eval * 100, bins=bins, label="Quantum",   color="coral",     alpha=0.7)
    ax2.axvline(np.mean(r_c_eval) * 100, color="royalblue", linestyle="--", linewidth=1.5)
    ax2.axvline(np.mean(r_q_eval) * 100, color="coral",     linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Return Rate (%)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Return Rate Comparison (Classical vs Quantum)")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    eval_table = pd.DataFrame({
        "Classical": [f"{np.mean(r_c_eval)*100:.2f}%", f"{np.std(r_c_eval)*100:.2f}%",
                      f"{np.min(r_c_eval)*100:.2f}%",  f"{np.max(r_c_eval)*100:.2f}%"],
        "Quantum":   [f"{np.mean(r_q_eval)*100:.2f}%", f"{np.std(r_q_eval)*100:.2f}%",
                      f"{np.min(r_q_eval)*100:.2f}%",  f"{np.max(r_q_eval)*100:.2f}%"],
    }, index=["平均収益率", "標準偏差", "最小収益率", "最大収益率"])
    st.table(eval_table)

else:
    st.info("サイドバーでパラメータを設定して「最適化を実行」を押してください。")
