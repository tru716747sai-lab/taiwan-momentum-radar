import time

import pandas as pd
import yfinance as yf


def _download_batch(tickers, period):
    """
    Fast path:
    download a batch of tickers in parallel.
    """
    return yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,          # 恢復平行下載
        group_by="column",
    )


def _extract_batch(data, batch):
    """
    Convert yfinance batch output into:
    close DataFrame
    volume DataFrame
    """

    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ---------------------------------------------------------
    # Multiple ticker format
    # ---------------------------------------------------------
    if isinstance(data.columns, pd.MultiIndex):

        level0 = data.columns.get_level_values(0)

        if "Close" not in level0:
            return pd.DataFrame(), pd.DataFrame()

        close = data["Close"].copy()

        if isinstance(close, pd.Series):
            close = close.to_frame()

        if "Volume" in level0:
            volume = data["Volume"].copy()

            if isinstance(volume, pd.Series):
                volume = volume.to_frame()

        else:
            volume = pd.DataFrame(
                index=close.index,
                columns=close.columns,
                dtype=float,
            )

        return close, volume

    # ---------------------------------------------------------
    # Single ticker batch
    # ---------------------------------------------------------
    if "Close" not in data.columns:
        return pd.DataFrame(), pd.DataFrame()

    ticker = batch[0]

    close = data[["Close"]].copy()
    close.columns = [ticker]

    if "Volume" in data.columns:
        volume = data[["Volume"]].copy()
        volume.columns = [ticker]
    else:
        volume = pd.DataFrame(
            index=data.index,
            columns=[ticker],
            dtype=float,
        )

    return close, volume


def _download_single(ticker, period, max_retries=3):
    """
    Slow recovery path.

    Only tickers missing from the parallel batch download
    come here.

    Retry individually with threads disabled.
    """

    last_error = None

    for attempt in range(1, max_retries + 1):

        try:

            data = yf.download(
                tickers=ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
                timeout=10,
            )

            if data is not None and not data.empty:

                # Single ticker may still return MultiIndex
                if isinstance(data.columns, pd.MultiIndex):

                    if "Close" not in data.columns.get_level_values(0):
                        raise ValueError("Close column missing")

                    close_obj = data["Close"]

                    if isinstance(close_obj, pd.DataFrame):
                        if close_obj.shape[1] == 0:
                            raise ValueError("Close data empty")
                        close = close_obj.iloc[:, 0]
                    else:
                        close = close_obj

                    if "Volume" in data.columns.get_level_values(0):

                        volume_obj = data["Volume"]

                        if isinstance(volume_obj, pd.DataFrame):
                            if volume_obj.shape[1] > 0:
                                volume = volume_obj.iloc[:, 0]
                            else:
                                volume = pd.Series(
                                    index=data.index,
                                    dtype=float,
                                )
                        else:
                            volume = volume_obj

                    else:
                        volume = pd.Series(
                            index=data.index,
                            dtype=float,
                        )

                else:

                    if "Close" not in data.columns:
                        raise ValueError("Close column missing")

                    close = data["Close"]

                    if "Volume" in data.columns:
                        volume = data["Volume"]
                    else:
                        volume = pd.Series(
                            index=data.index,
                            dtype=float,
                        )

                close = close.dropna()

                if not close.empty:

                    close.name = ticker
                    volume.name = ticker

                    return close, volume, None

                last_error = "empty Close data"

            else:
                last_error = "empty response"

        except Exception as exc:

            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

        if attempt < max_retries:

            wait_seconds = 2 ** (attempt - 1)

            print(
                f"[retry] {ticker}: "
                f"attempt {attempt}/{max_retries} failed "
                f"({last_error}); "
                f"retry in {wait_seconds}s"
            )

            time.sleep(wait_seconds)

    return None, None, last_error


def download_prices(tickers, period, batch_size):
    """
    Production download strategy:

    1. Parallel batch download for speed.
    2. Detect missing/invalid tickers.
    3. Retry ONLY those tickers individually.
    4. Keep NaN volume as NaN.
    5. A few failed tickers do not kill the radar.
    """

    tickers = list(dict.fromkeys(tickers))

    all_close = []
    all_volume = []

    failed_candidates = set()

    # =========================================================
    # FAST PATH
    # Parallel batch downloads
    # =========================================================

    for start in range(
        0,
        len(tickers),
        batch_size,
    ):

        batch = tickers[
            start:start + batch_size
        ]

        try:

            data = _download_batch(
                batch,
                period,
            )

            close, volume = _extract_batch(
                data,
                batch,
            )

        except Exception as exc:

            print(
                f"[batch error] "
                f"{batch[0]} ... {batch[-1]}: "
                f"{type(exc).__name__}: {exc}"
            )

            failed_candidates.update(batch)
            continue

        if close.empty:

            failed_candidates.update(batch)
            continue

        valid_tickers = []

        for ticker in batch:

            if ticker not in close.columns:

                failed_candidates.add(ticker)
                continue

            if close[ticker].dropna().empty:

                failed_candidates.add(ticker)
                continue

            valid_tickers.append(ticker)

        if valid_tickers:

            close_out = close[
                valid_tickers
            ].copy()

            volume_out = pd.DataFrame(
                index=close_out.index,
                columns=valid_tickers,
                dtype=float,
            )

            available_volume = [
                ticker
                for ticker in valid_tickers
                if ticker in volume.columns
            ]

            if available_volume:

                volume_out[
                    available_volume
                ] = volume[
                    available_volume
                ]

            all_close.append(close_out)
            all_volume.append(volume_out)

    # =========================================================
    # Combine fast-path results
    # =========================================================

    if all_close:

        close_all = pd.concat(
            all_close,
            axis=1,
        )

    else:

        close_all = pd.DataFrame()

    if all_volume:

        volume_all = pd.concat(
            all_volume,
            axis=1,
        )

    else:

        volume_all = pd.DataFrame()

    # Remove duplicate columns defensively
    if not close_all.empty:

        close_all = close_all.loc[
            :,
            ~close_all.columns.duplicated(
                keep="last"
            ),
        ]

    if not volume_all.empty:

        volume_all = volume_all.loc[
            :,
            ~volume_all.columns.duplicated(
                keep="last"
            ),
        ]

    # =========================================================
    # Detect anything batch download silently missed
    # =========================================================

    successful = set(close_all.columns)

    failed_candidates.update(
        set(tickers) - successful
    )

    # =========================================================
    # SLOW PATH
    # Retry ONLY failed tickers
    # =========================================================

    final_failed = []

    if failed_candidates:

        print(
            f"\n[recovery] "
            f"{len(failed_candidates)} ticker(s) "
            "missing from batch download; "
            "starting individual retry..."
        )

    for ticker in sorted(failed_candidates):

        close_series, volume_series, error = (
            _download_single(
                ticker,
                period,
                max_retries=3,
            )
        )

        if close_series is None:

            final_failed.append(
                (ticker, error)
            )

            continue

        # Remove stale/partial version if present
        if ticker in close_all.columns:

            close_all = close_all.drop(
                columns=[ticker]
            )

        if ticker in volume_all.columns:

            volume_all = volume_all.drop(
                columns=[ticker]
            )

        close_all = pd.concat(
            [
                close_all,
                close_series.to_frame(),
            ],
            axis=1,
        )

        volume_all = pd.concat(
            [
                volume_all,
                volume_series.to_frame(),
            ],
            axis=1,
        )

        print(
            f"[recovered] {ticker}"
        )

    # =========================================================
    # Final validation
    # =========================================================

    if close_all.empty:

        raise RuntimeError(
            "No market price data could be downloaded."
        )

    close_all = close_all.dropna(
        axis=1,
        how="all",
    )

    # Align volume with Close.
    # IMPORTANT: missing volume remains NaN.
    volume_all = volume_all.reindex(
        index=close_all.index,
        columns=close_all.columns,
    )

    # Stable ordering
    close_all = close_all.sort_index(
        axis=1
    )

    volume_all = volume_all.reindex(
        columns=close_all.columns
    )

    # =========================================================
    # Download quality report
    # =========================================================

    downloaded_count = len(
        close_all.columns
    )

    total_count = len(tickers)

    print(
        f"\n[market] "
        f"downloaded {downloaded_count}/"
        f"{total_count} tickers"
    )

    if final_failed:

        print(
            f"[market warning] "
            f"{len(final_failed)} ticker(s) "
            "failed after all retries:"
        )

        for ticker, error in final_failed:

            print(
                f"  - {ticker}: {error}"
            )

    elif failed_candidates:

        print(
            "[market] "
            "all missing tickers recovered."
        )

    else:

        print(
            "[market] "
            "batch download complete; "
            "no recovery needed."
        )

    return close_all, volume_all
