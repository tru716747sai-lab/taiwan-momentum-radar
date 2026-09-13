from pathlib import Path
import pandas as pd
from .universe import get_universe
from .market import download_prices
from .features import latest_features
from .scoring import apply_gates, score

def run(cfg):
    uni = get_universe(cfg["universe"]["markets"], cfg["universe"]["exclude_etf"])
    close, volume = download_prices(
        uni["ticker"], cfg["download"]["period"], cfg["download"]["batch_size"]
    )
    feats = latest_features(close, volume)
    ranked = score(apply_gates(feats, cfg), cfg).merge(uni, on="ticker", how="left")
    top = ranked.head(cfg["output"]["top_n"]).copy()
    Path("reports").mkdir(exist_ok=True)
    ranked.to_csv("reports/latest_full.csv", index=False, encoding="utf-8-sig")
    top.to_csv("reports/latest.csv", index=False, encoding="utf-8-sig")
    cols = ["ticker","code","name","market","score","momentum_score","trend_score",
            "liquidity_score","risk_score","price","ret_20","ret_60","ret_120","ret_252"]
    shown = top[[c for c in cols if c in top.columns]].copy()
    md = "# Taiwan Momentum Radar — Latest\n\n"
    md += f"資料日：{close.index.max().date()}\n\n"
    md += shown.to_markdown(index=False, floatfmt=".2f")
    md += "\n\n> 研究排名，不構成投資建議。\n"
    Path("reports/latest.md").write_text(md, encoding="utf-8")
    return top
