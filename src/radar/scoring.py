import pandas as pd

def _pct(s, higher=True):
    return s.rank(pct=True, method="average", ascending=higher) * 100

def apply_gates(df, cfg):
    u = cfg["universe"]
    return df[
        (df["price"] >= u["min_price"]) &
        (df["history_days"] >= u["min_history_days"]) &
        (df["avg_turnover_20"] >= u["min_avg_turnover_twd"])
    ].dropna(subset=["ret_20","ret_60","ret_120","ret_252",
                     "above_ma_20","above_ma_60","above_ma_120",
                     "volatility_20","max_drawdown_60"]).copy()

def score(df, cfg):
    x = df.copy()
    mh = cfg["score"]["momentum_horizons"]
    x["momentum_score"] = sum(
        _pct(x[f"ret_{h}"]) * float(w) for h, w in mh.items()
    )
    x["trend_score"] = (
        _pct(x["above_ma_20"]) + _pct(x["above_ma_60"]) + _pct(x["above_ma_120"])
    ) / 3
    x["liquidity_score"] = _pct(x["avg_turnover_20"])
    # Lower volatility is better; drawdown closer to zero is better.
    x["risk_score"] = (_pct(x["volatility_20"], higher=False) +
                       _pct(x["max_drawdown_60"], higher=True)) / 2
    w = cfg["score"]
    x["score"] = (
        x["momentum_score"] * w["momentum_weight"] +
        x["trend_score"] * w["trend_weight"] +
        x["liquidity_score"] * w["liquidity_weight"] +
        x["risk_score"] * w["risk_weight"]
    )
    return x.sort_values(["score","momentum_score"], ascending=False)
