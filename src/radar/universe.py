import pandas as pd

URLS = {
    "TWSE": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
    "TPEx": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
}

def _read_market(market: str) -> pd.DataFrame:
    tables = pd.read_html(URLS[market])
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    first = df.columns[0]
    parts = df[first].astype(str).str.strip().str.split(r"\s+", n=1, expand=True)
    out = pd.DataFrame({"code": parts[0], "name": parts[1] if parts.shape[1] > 1 else ""})
    out["market"] = market
    out = out[out["code"].str.fullmatch(r"\d{4}", na=False)]
    return out.drop_duplicates("code")

def get_universe(markets=("TWSE","TPEx"), exclude_etf=True) -> pd.DataFrame:
    frames = [_read_market(m) for m in markets]
    df = pd.concat(frames, ignore_index=True)
    # Four-digit common shares are retained. ISIN pages also contain securities
    # whose codes are not exactly four digits; those are excluded by construction.
    suffix = df["market"].map({"TWSE": ".TW", "TPEx": ".TWO"})
    df["ticker"] = df["code"] + suffix
    return df[["ticker","code","name","market"]].drop_duplicates("ticker")
