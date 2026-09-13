import pandas as pd


def _pct(s, higher=True):

    return (
        s.rank(
            pct=True,
            method="average",
            ascending=higher,
            na_option="keep",
        )
        * 100
    )


# =============================================================
# Base Gate
# =============================================================

def _base_gate(df, cfg):

    u = cfg["universe"]

    return df[
        (df["price"] >= u["min_price"])
        &
        (
            df["avg_turnover_20"]
            >= u["min_avg_turnover_twd"]
        )
    ].copy()


# =============================================================
# 1. STRONG GATE
#
# 成熟、中長期強勢股
# =============================================================

def apply_strong_gate(df, cfg):

    x = _base_gate(
        df,
        cfg,
    )

    min_history = max(
        int(
            cfg["universe"].get(
                "min_history_days",
                280,
            )
        ),
        253,
    )

    x = x[
        x["history_days"]
        >= min_history
    ]

    required = [
        "ret_20",
        "ret_60",
        "ret_120",
        "ret_252",
        "above_ma_20",
        "above_ma_60",
        "above_ma_120",
        "volatility_20",
        "max_drawdown_60",
    ]

    return x.dropna(
        subset=required
    ).copy()


# =============================================================
# 2. BREAKOUT GATE
#
# 已經開始發動
# 不要求一年歷史
# =============================================================

def apply_breakout_gate(df, cfg):

    x = _base_gate(
        df,
        cfg,
    )

    x = x[
        x["history_days"] >= 121
    ]

    required = [
        "ret_5",
        "ret_20",
        "ret_60",
        "ret_120",
        "volume_ratio_5",
        "volume_ratio_20",
        "breakout_60",
        "breakout_120",
        "momentum_shift_5_20",
        "momentum_shift_20_60",
    ]

    return x.dropna(
        subset=required
    ).copy()


# =============================================================
# 3. IGNITION GATE
#
# 早期異常啟動
# =============================================================

def apply_ignition_gate(df, cfg):

    x = _base_gate(
        df,
        cfg,
    )

    # 需要前 60 日 + 今日
    x = x[
        x["history_days"] >= 61
    ]

    required = [
        "ret_5",
        "ret_10",
        "ret_20",
        "ret_60",
        "volume_ratio_5",
        "breakout_60",
        "distance_to_high_60",
        "momentum_shift_5_20",
        "momentum_shift_20_60",
    ]

    return x.dropna(
        subset=required
    ).copy()


# =============================================================
# STRONG SCORE
# =============================================================

def score_strong(df, cfg):

    x = df.copy()

    mh = cfg["score"][
        "momentum_horizons"
    ]

    x["momentum_score"] = sum(
        _pct(
            x[f"ret_{h}"]
        )
        * float(w)
        for h, w in mh.items()
    )

    x["trend_score"] = (
        _pct(
            x["above_ma_20"]
        )
        +
        _pct(
            x["above_ma_60"]
        )
        +
        _pct(
            x["above_ma_120"]
        )
    ) / 3

    x["liquidity_score"] = _pct(
        x["avg_turnover_20"]
    )

    # 低波動、低回撤較佳
    x["risk_score"] = (
        _pct(
            x["volatility_20"],
            higher=False,
        )
        +
        _pct(
            x["max_drawdown_60"],
            higher=True,
        )
    ) / 2

    w = cfg["score"]

    x["score"] = (
        x["momentum_score"]
        * w["momentum_weight"]
        +
        x["trend_score"]
        * w["trend_weight"]
        +
        x["liquidity_score"]
        * w["liquidity_weight"]
        +
        x["risk_score"]
        * w["risk_weight"]
    )

    return x.sort_values(
        [
            "score",
            "momentum_score",
        ],
        ascending=False,
    )


# =============================================================
# BREAKOUT SCORE
# =============================================================

def score_breakout(df):

    x = df.copy()

    # 短中期動能
    x["breakout_momentum_score"] = (
        _pct(
            x["ret_5"]
        ) * 0.15
        +
        _pct(
            x["ret_20"]
        ) * 0.40
        +
        _pct(
            x["ret_60"]
        ) * 0.30
        +
        _pct(
            x["ret_120"]
        ) * 0.15
    )

    # 爆量
    x["volume_surge_score"] = (
        _pct(
            x["volume_ratio_5"]
        ) * 0.65
        +
        _pct(
            x["volume_ratio_20"]
        ) * 0.35
    )

    # 突破位置
    x["breakout_price_score"] = (
        _pct(
            x["breakout_60"]
        ) * 0.60
        +
        _pct(
            x["breakout_120"]
        ) * 0.40
    )

    # Momentum shift
    x["momentum_shift_score"] = (
        _pct(
            x["momentum_shift_5_20"]
        ) * 0.55
        +
        _pct(
            x["momentum_shift_20_60"]
        ) * 0.45
    )

    x["breakout_score"] = (
        x[
            "breakout_momentum_score"
        ] * 0.40
        +
        x[
            "volume_surge_score"
        ] * 0.25
        +
        x[
            "breakout_price_score"
        ] * 0.20
        +
        x[
            "momentum_shift_score"
        ] * 0.15
    )

    return x.sort_values(
        [
            "breakout_score",
            "volume_surge_score",
        ],
        ascending=False,
    )


# =============================================================
# IGNITION SCORE
# =============================================================

def score_ignition(df):

    x = df.copy()

    # ---------------------------------------------------------
    # 短期 Momentum
    # ---------------------------------------------------------

    x["ignition_short_momentum"] = (
        _pct(
            x["ret_5"]
        ) * 0.45
        +
        _pct(
            x["ret_10"]
        ) * 0.35
        +
        _pct(
            x["ret_20"]
        ) * 0.20
    )

    # ---------------------------------------------------------
    # Volume
    #
    # Ignition 特別重視最近5日爆量
    # ---------------------------------------------------------

    x["ignition_volume"] = _pct(
        x["volume_ratio_5"]
    )

    # ---------------------------------------------------------
    # Momentum shift
    # ---------------------------------------------------------

    x["ignition_shift"] = (
        _pct(
            x["momentum_shift_5_20"]
        ) * 0.65
        +
        _pct(
            x["momentum_shift_20_60"]
        ) * 0.35
    )

    # ---------------------------------------------------------
    # Price position
    # ---------------------------------------------------------

    x["ignition_price_position"] = _pct(
        x["distance_to_high_60"]
    )

    raw_score = (
        x[
            "ignition_short_momentum"
        ] * 0.30
        +
        x[
            "ignition_volume"
        ] * 0.30
        +
        x[
            "ignition_shift"
        ] * 0.25
        +
        x[
            "ignition_price_position"
        ] * 0.15
    )

    # =========================================================
    # Late chase penalty
    #
    # 這只是排序上的額外保護。
    # 真正是否允許入榜由 Hard Gate 決定。
    # =========================================================

    penalty_20 = (
        (
            x["ret_20"]
            - 0.60
        )
        .clip(lower=0)
        * 35
    )

    penalty_60 = (
        (
            x["ret_60"]
            - 1.20
        )
        .clip(lower=0)
        * 15
    )

    x["late_chase_penalty"] = (
        penalty_20
        + penalty_60
    ).clip(
        upper=30
    )

    x["ignition_score"] = (
        raw_score
        - x["late_chase_penalty"]
    ).clip(
        lower=0,
        upper=100,
    )

    # =========================================================
    # Hard signals
    # =========================================================

    # 必須爆量：
    # 最近5日平均量 >= 前20日平均量 1.5倍
    x["volume_surge_flag"] = (
        x["volume_ratio_5"]
        >= 1.50
    )

    # 接近前60日高點：
    # -3% ~ +10%
    x["near_high_flag"] = (
        (
            x["distance_to_high_60"]
            >= -0.03
        )
        &
        (
            x["distance_to_high_60"]
            < 0.10
        )
    )

    # 真正突破前高
    x["breakout_flag"] = (
        x["breakout_60"]
        >= 0
    )

    # 最近5日動能速度高於20日平均
    x["momentum_shift_flag"] = (
        x["momentum_shift_5_20"]
        > 0
    )

    # 尚未過熱
    x["early_stage_flag"] = (
        (x["ret_20"] < 0.60)
        &
        (x["ret_60"] < 1.20)
    )

    # =========================================================
    # Ignition absolute high-distance cap
    #
    # 即使 Momentum shift 很強，
    # 如果已經比前60日高點高超過10%，
    # 也不再視為「早期啟動」。
    # =========================================================

    x["not_extended_flag"] = (
        x["distance_to_high_60"]
        < 0.10
    )

    # =========================================================
    # FINAL IGNITION QUALIFICATION
    #
    # 必須同時：
    #
    # 1. 未過熱
    # 2. 不得已遠離前高 >10%
    # 3. 最近5日量 >= 1.5x
    # 4. 接近前高 OR 短期 Momentum shift
    # =========================================================

    x["ignition_qualified"] = (
        x["early_stage_flag"]
        &
        x["not_extended_flag"]
        &
        x["volume_surge_flag"]
        &
        (
            x["near_high_flag"]
            |
            x["momentum_shift_flag"]
        )
    )

    # 僅供報告觀察
    x["ignition_signal_count"] = (
        x[
            [
                "volume_surge_flag",
                "near_high_flag",
                "breakout_flag",
                "momentum_shift_flag",
                "early_stage_flag",
                "not_extended_flag",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )

    return x.sort_values(
        [
            "ignition_score",
            "ignition_signal_count",
        ],
        ascending=False,
    )


# =============================================================
# Backward compatibility
#
# 舊版 backtest.py 若使用 apply_gates / score，
# 繼續走原本 Strong 模型。
# =============================================================

def apply_gates(df, cfg):

    return apply_strong_gate(
        df,
        cfg,
    )


def score(df, cfg):

    return score_strong(
        df,
        cfg,
    )
