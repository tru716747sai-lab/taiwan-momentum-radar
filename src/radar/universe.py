from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd

URLS = {
    "TWSE": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
    "TPEx": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
}


def _fetch_html(url: str) -> str:
    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 TaiwanMomentumRadar/1.0"},
    )
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()

    for encoding in ("cp950", "big5", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("cp950", errors="replace")


def _read_market(market: str) -> pd.DataFrame:
    html = _fetch_html(URLS[market])
    tables = pd.read_html(StringIO(html))

    if not tables:
        raise RuntimeError(f"No ISIN table returned for {market}")

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    first = df.columns[0]
    parts = (
        df[first]
        .astype(str)
        .str.strip()
        .str.split(r"\s+", n=1, expand=True)
    )

    out = pd.DataFrame({
        "code": parts[0],
        "name": parts[1] if parts.shape[1] > 1 else "",
    })

    out["market"] = market
    out = out[out["code"].str.fullmatch(r"\d{4}", na=False)]

    if out.empty:
        raise RuntimeError(f"No four-digit securities parsed for {market}")

    return out.drop_duplicates("code")


def get_universe(
    markets=("TWSE", "TPEx"),
    exclude_etf=True,
) -> pd.DataFrame:

    frames = [_read_market(m) for m in markets]
    df = pd.concat(frames, ignore_index=True)

    suffix = df["market"].map({
        "TWSE": ".TW",
        "TPEx": ".TWO",
    })

    df["ticker"] = df["code"] + suffix

    return df[
        ["ticker", "code", "name", "market"]
    ].drop_duplicates("ticker")
