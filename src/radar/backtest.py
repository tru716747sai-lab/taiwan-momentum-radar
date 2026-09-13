from pathlib import Path
import json
import numpy as np
import pandas as pd
from .universe import get_universe
from .market import download_prices
from .features import features_on_date
from .scoring import apply_gates, score

def run_backtest(cfg):
    uni = get_universe(cfg["universe"]["markets"], cfg["universe"]["exclude_etf"])
    close, volume = download_prices(uni["ticker"], period="5y",
                                    batch_size=cfg["download"]["batch_size"])
    # Month-end dates available in actual price index.
    month = pd.Series(close.index, index=close.index).groupby(close.index.to_period("M")).max()
    dates = list(month.values)
    n = cfg["backtest"]["top_n"]
    cost = cfg["backtest"]["transaction_cost_bps"] / 10000
    records, prev = [], set()
    for i in range(len(dates)-1):
        d0, d1 = pd.Timestamp(dates[i]), pd.Timestamp(dates[i+1])
        feats = features_on_date(close, volume, d0)
        gated = apply_gates(feats, cfg)
        if gated.empty:
            continue
        picks = list(score(gated, cfg).head(n)["ticker"])
        p0 = close.loc[:d0, picks].iloc[-1]
        p1 = close.loc[:d1, picks].iloc[-1]
        ret = (p1/p0 - 1).dropna()
        if ret.empty:
            continue
        current = set(ret.index)
        turnover = 1.0 if not prev else len(current.symmetric_difference(prev))/(2*max(len(current),1))
        gross = ret.mean()
        net = gross - turnover*cost
        records.append({"date": d1, "gross_return": gross, "turnover": turnover,
                        "net_return": net, "n": len(ret)})
        prev = current
    out = pd.DataFrame(records).set_index("date")
    if out.empty:
        raise RuntimeError("Backtest produced no periods.")
    out["equity"] = (1+out["net_return"]).cumprod()
    ann = (out["equity"].iloc[-1] ** (12/len(out)) - 1)
    vol = out["net_return"].std() * np.sqrt(12)
    dd = out["equity"]/out["equity"].cummax()-1
    summary = {
        "periods": int(len(out)), "annualized_return": float(ann),
        "annualized_volatility": float(vol),
        "sharpe_zero_rf": float(ann/vol) if vol else None,
        "max_drawdown": float(dd.min()),
        "final_equity": float(out["equity"].iloc[-1])
    }
    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/backtest_equity.csv")
    Path("reports/backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
