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


def _merge_universe(
    df,
    universe,
):

    return df.merge(
        universe,
        on="ticker",
        how="left",
    )


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
    # 不做任何空榜救援。
    #
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
    # Markdown report
    # =========================================================

    data_date = (
        close.index.max().date()
    )

    md = (
        "# Taiwan Momentum Radar — v2.1\n\n"
    )

    md += (
        f"資料日：{data_date}\n\n"
    )

    md += (
        "三層 Radar："
        "**Strong / Breakout / Ignition**。\n\n"
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
            "momentum_score",
            "trend_score",
            "liquidity_score",
            "risk_score",
            "price",
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
        "異常啟動 Watchlist\n\n"
    )

    md += (
        "尚未明顯過熱，"
        "但量價開始出現不尋常變化的股票。\n\n"
    )

    md += (
        "**硬條件：** "
        "20日漲幅 < 60%、"
        "60日漲幅 < 120%、"
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
    # Interpretation
    # =========================================================

    md += (
        "\n\n---\n\n"
        "## 分數解讀\n\n"
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
        "Momentum shift 僅供橫斷面排序，"
        "不可解讀為實際價格加速度百分比。\n\n"
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
    }
