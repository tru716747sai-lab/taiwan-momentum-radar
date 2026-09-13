from pathlib import Path

from .universe import get_universe
from .market import download_prices
from .features import latest_features
from .scoring import (
    apply_strong_gate,
    apply_breakout_gate,
    apply_ignition_gate,
    score_strong,
    score_breakout,
    score_ignition,
)
from .tracking import update_ignition_tracking


def _merge_universe(
    df,
    universe,
):

    return df.merge(
        universe,
        on="ticker",
        how="left",
    )


def _add_trade_cost_columns(
    df,
):

    x = df.copy()

    if "price" in x.columns:

        x["lot_cost"] = (
            x["price"]
            * 1000
        )

    return x


def _percent_for_display(
    df,
    columns,
):

    x = df.copy()

    for column in columns:

        if column in x.columns:

            x[column] = (
                x[column]
                * 100
            )

    return x


def _markdown_table(
    df,
    columns,
    percent_columns=None,
):

    columns = [
        c
        for c in columns
        if c in df.columns
    ]

    shown = df[
        columns
    ].copy()

    if percent_columns:

        shown = _percent_for_display(
            shown,
            percent_columns,
        )

    if shown.empty:

        return (
            "_今日無符合條件標的。_"
        )

    return shown.to_markdown(
        index=False,
        floatfmt=".2f",
    )


def run(cfg):

    # =========================================================
    # Universe + Market Data
    # =========================================================

    universe = get_universe(
        cfg["universe"]["markets"],
        cfg["universe"]["exclude_etf"],
    )

    close, volume = download_prices(
        universe["ticker"],
        cfg["download"]["period"],
        cfg["download"]["batch_size"],
    )

    features = latest_features(
        close,
        volume,
    )

    top_n = int(
        cfg["output"]["top_n"]
    )

    # =========================================================
    # 1. STRONG
    # =========================================================

    strong_pool = apply_strong_gate(
        features,
        cfg,
    )

    strong_ranked = score_strong(
        strong_pool,
        cfg,
    )

    strong_ranked = _merge_universe(
        strong_ranked,
        universe,
    )

    strong_ranked = (
        _add_trade_cost_columns(
            strong_ranked
        )
    )

    strong = (
        strong_ranked
        .head(top_n)
        .copy()
    )

    # =========================================================
    # 2. BREAKOUT
    # =========================================================

    breakout_pool = apply_breakout_gate(
        features,
        cfg,
    )

    breakout_ranked = score_breakout(
        breakout_pool,
    )

    breakout_ranked = _merge_universe(
        breakout_ranked,
        universe,
    )

    breakout_ranked = (
        _add_trade_cost_columns(
            breakout_ranked
        )
    )

    breakout = (
        breakout_ranked
        .head(top_n)
        .copy()
    )

    # =========================================================
    # 3. IGNITION
    # =========================================================

    ignition_pool = apply_ignition_gate(
        features,
        cfg,
    )

    ignition_ranked_raw = score_ignition(
        ignition_pool,
    )

    # =========================================================
    # HARD FILTER
    #
    # 不做空榜救援。
    # 沒有合格股票就是空榜。
    # =========================================================

    ignition_qualified_raw = (
        ignition_ranked_raw[
            ignition_ranked_raw[
                "ignition_qualified"
            ]
        ]
        .copy()
    )

    ignition_ranked = _merge_universe(
        ignition_ranked_raw,
        universe,
    )

    ignition_qualified = _merge_universe(
        ignition_qualified_raw,
        universe,
    )

    ignition_ranked = (
        _add_trade_cost_columns(
            ignition_ranked
        )
    )

    ignition_qualified = (
        _add_trade_cost_columns(
            ignition_qualified
        )
    )

    ignition = (
        ignition_qualified
        .head(top_n)
        .copy()
    )

    # =========================================================
    # Reports
    # =========================================================

    reports = Path(
        "reports"
    )

    reports.mkdir(
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Full rankings
    # ---------------------------------------------------------

    strong_ranked.to_csv(
        reports / "strong_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    breakout_ranked.to_csv(
        reports / "breakout_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ignition_ranked.to_csv(
        reports / "ignition_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # Top lists
    # ---------------------------------------------------------

    strong.to_csv(
        reports / "strong.csv",
        index=False,
        encoding="utf-8-sig",
    )

    breakout.to_csv(
        reports / "breakout.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ignition.to_csv(
        reports / "ignition.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # Backward compatibility
    # ---------------------------------------------------------

    strong_ranked.to_csv(
        reports / "latest_full.csv",
        index=False,
        encoding="utf-8-sig",
    )

    strong.to_csv(
        reports / "latest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================
    # Data Date
    # =========================================================

    data_date = (
        close.index.max().date()
    )

    # =========================================================
    # v2.3 Ignition Tracking
    #
    # 第一次進 Ignition = Day 0
    #
    # 未來 Daily 自動更新：
    # D+1 / D+3 / D+5 / D+10 / D+20
    # =========================================================

    ignition_tracking = (
        update_ignition_tracking(
            ignition=ignition,
            close=close,
            data_date=data_date,
            reports_dir=reports,
        )
    )

    # =========================================================
    # Markdown report
    # =========================================================

    md = (
        "# Taiwan Momentum Radar — v2.3\n\n"
    )

    md += (
        f"資料日：{data_date}\n\n"
    )

    md += (
        "三層 Radar："
        "**Strong / Breakout / Ignition**。\n\n"
    )

    md += (
        "v2.3 新增："
        "**目前價格、整張約需資金、"
        "Ignition Day 0 與後續績效追蹤。**\n\n"
    )

    # =========================================================
    # Strong
    # =========================================================

    md += (
        "## 1. Strong — 穩健強勢榜\n\n"
    )

    md += (
        "目前已形成較成熟、"
        "相對均衡強勢趨勢的股票。\n\n"
    )

    md += _markdown_table(
        strong,
        [
            "ticker",
            "code",
            "name",
            "market",
            "score",
            "price",
            "lot_cost",
            "momentum_score",
            "trend_score",
            "liquidity_score",
            "risk_score",
            "ret_20",
            "ret_60",
            "ret_252",
        ],
        [
            "ret_20",
            "ret_60",
            "ret_252",
        ],
    )

    # =========================================================
    # Breakout
    # =========================================================

    md += (
        "\n\n"
        "## 2. Breakout — 爆發榜\n\n"
    )

    md += (
        "目前已開始出現強動能、"
        "爆量、突破或 Momentum shift 的股票。\n\n"
    )

    md += _markdown_table(
        breakout,
        [
            "ticker",
            "code",
            "name",
            "market",
            "breakout_score",
            "price",
            "lot_cost",
            "ret_5",
            "ret_20",
            "ret_60",
            "volume_ratio_5",
            "breakout_60",
            "momentum_shift_5_20",
        ],
        [
            "ret_5",
            "ret_20",
            "ret_60",
            "breakout_60",
        ],
    )

    # =========================================================
    # Ignition
    # =========================================================

    md += (
        "\n\n"
        "## 3. Ignition — "
        "早期異常啟動 Watchlist\n\n"
    )

    md += (
        "價格尚未明顯噴出，"
        "但量價開始出現不尋常變化的股票。\n\n"
    )

    md += (
        "**v2.2 硬條件：** "
        "-5% ≤ 5日報酬 < +20%、"
        "20日漲幅 < +35%、"
        "60日漲幅 < +70%、"
        "最近5日均量 ≥ 前20日均量1.5倍、"
        "不得高於前60日高點10%以上，"
        "且接近前高或短期 Momentum shift 轉強。\n\n"
    )

    md += _markdown_table(
        ignition,
        [
            "ticker",
            "code",
            "name",
            "market",
            "ignition_score",
            "ignition_signal_count",
            "price",
            "lot_cost",
            "ret_5",
            "ret_20",
            "ret_60",
            "volume_ratio_5",
            "distance_to_high_60",
            "momentum_shift_5_20",
        ],
        [
            "ret_5",
            "ret_20",
            "ret_60",
            "distance_to_high_60",
        ],
    )

    # =========================================================
    # Ignition Tracking
    # =========================================================

    md += (
        "\n\n"
        "## 4. Ignition — Day 0 績效追蹤\n\n"
    )

    md += (
        "第一次進入 Ignition 的交易日定義為 "
        "**Day 0**。同一股票連續入榜不重設 Day 0。\n\n"
    )

    md += (
        "D+1 / D+3 / D+5 / D+10 / D+20 "
        "均指後續**交易日**，不是曆日。\n\n"
    )

    md += _markdown_table(
        ignition_tracking,
        [
            "signal_date",
            "ticker",
            "code",
            "name",
            "day0_price",
            "lot_cost",
            "ignition_score",
            "day0_ret_5",
            "day0_volume_ratio_5",
            "d1_return",
            "d3_return",
            "d5_return",
            "d10_return",
            "d20_return",
            "latest_return",
            "trading_days_since_signal",
        ],
        [
            "day0_ret_5",
            "d1_return",
            "d3_return",
            "d5_return",
            "d10_return",
            "d20_return",
            "latest_return",
        ],
    )

    # =========================================================
    # Interpretation
    # =========================================================

    md += (
        "\n\n---\n\n"
        "## 分數與追蹤解讀\n\n"
        "**Strong Score 高：** "
        "代表在 Strong 候選池中，"
        "目前中長期強勢程度相對較高。\n\n"
        "**Breakout Score 高：** "
        "代表在 Breakout 候選池中，"
        "目前爆發／突破特徵相對較強。\n\n"
        "**Ignition Score 高：** "
        "代表通過早期啟動硬條件後，"
        "短期量價異常程度相對較高。\n\n"
        "**不同 Radar 的分數不可直接互相比較，"
        "也不是未來上漲機率或勝率。**\n\n"
        "Day 0 追蹤的目的，是驗證 "
        "Ignition 首次發出訊號後，"
        "未來究竟還有沒有後續報酬。\n\n"
        "目前價格與整張資金只供實際操作判讀，"
        "不影響 Radar 排名。\n\n"
        "> 研究排名，不構成投資建議。"
    )

    (
        reports / "latest.md"
    ).write_text(
        md,
        encoding="utf-8",
    )

    return {
        "strong": strong,
        "breakout": breakout,
        "ignition": ignition,
        "ignition_tracking": ignition_tracking,
    }
