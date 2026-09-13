import numpy as np
import pandas as pd

def latest_features(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < 20:
            continue
        v = volume[t].reindex(s.index).fillna(0)
        r = s.pct_change()
        row = {
            "ticker": t, "price": s.iloc[-1], "history_days": len(s),
            "avg_turnover_20": (s*v).tail(20).mean(),
            "volatility_20": r.tail(20).std() * np.sqrt(252),
        }
        for h in (20,60,120,252):
            row[f"ret_{h}"] = s.iloc[-1] / s.iloc[-h-1] - 1 if len(s) > h else np.nan
        for h in (20,60,120):
            ma = s.tail(h).mean() if len(s) >= h else np.nan
            row[f"above_ma_{h}"] = s.iloc[-1] / ma - 1 if pd.notna(ma) else np.nan
        if len(s) >= 60:
            peak = s.tail(60).cummax()
            row["max_drawdown_60"] = (s.tail(60)/peak - 1).min()
        else:
            row["max_drawdown_60"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def features_on_date(close, volume, date):
    c = close.loc[:date]
    v = volume.loc[:date]
    return latest_features(c, v)
