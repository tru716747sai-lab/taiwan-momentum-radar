import time
import pandas as pd
import yfinance as yf

def download_prices(tickers, period="2y", batch_size=100):
    tickers = list(dict.fromkeys(tickers))
    close_parts, vol_parts = [], []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        raw = yf.download(
            batch, period=period, auto_adjust=True, group_by="column",
            progress=False, threads=True
        )
        if raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy()
            vol = raw["Volume"].copy()
        else:
            # single ticker
            close = raw[["Close"]].rename(columns={"Close": batch[0]})
            vol = raw[["Volume"]].rename(columns={"Volume": batch[0]})
        close_parts.append(close)
        vol_parts.append(vol)
        time.sleep(0.2)
    if not close_parts:
        raise RuntimeError("No market data downloaded.")
    close = pd.concat(close_parts, axis=1)
    volume = pd.concat(vol_parts, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    volume = volume.reindex(index=close.index, columns=close.columns)
    return close, volume
