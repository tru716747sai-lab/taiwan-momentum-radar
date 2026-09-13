import time

import pandas as pd
import yfinance as yf


def _normalize_download(data):
    """
    Normalize yfinance output into close / volume DataFrames.
    """

    if data is None or data.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 多股票下載
    if isinstance(data.columns, pd.MultiIndex):

        level0 = data.columns.get_level_values(0)

        if "Close" not in level0:
            return pd.DataFrame(), pd.DataFrame()

        close = data["Close"].copy()

        if "Volume" in level0:
            volume = data["Volume"].copy()
        else:
            volume = pd.DataFrame(
                index=close.index,
                columns=close.columns,
                dtype=float,
            )

        # 單一 ticker 有時仍會變 Series
        if isinstance(close, pd.Series):
            close = close.to_frame()

        if isinstance(volume, pd.Series):
            volume = volume.to_frame()

        return close, volume

    # 單股票格式
    if "Close" not in data.columns:
        return pd.DataFrame(), pd.DataFrame()

    close = data[["Close"]].copy()
    close.columns = ["Close"]

    if "Volume" in data.columns:
        volume = data[["Volume"]].copy()
        volume.columns = ["Volume"]
    else:
        volume = pd.DataFrame(
            index=data.index,
            columns=["Volume"],
            dtype=float,
        )

    return close, volume


def _download_batch(tickers, period):
    """
    Normal batch download.
    """

    return yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )


def _download_single(
    ticker,
    period,
    max_retries=3,
):
    """
    Retry a failed ticker individually.

    This handles transient yfinance / SQLite / network errors
    without failing the whole radar run.
    """

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            data = yf.download(
                tickers=ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="column",
            )

            if (
                data is not None
                and not data.empty
                and "Close" in data.columns
                and data["Close"].notna().any()
            ):

                close = (
                    data["Close"]
                    .rename(ticker)
                )

                if "Volume" in data.columns:

                    volume = (
                        data["Volume"]
                        .rename(ticker)
                    )

                else:

                    volume = pd.Series(
                        index=data.index,
                        name=ticker,
                        dtype=float,
                    )

                return (
                    close,
                    volume,
                    None,
                )

            last_error = (
                "empty or invalid response"
            )

        except Exception as exc:

            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

        if attempt < max_retries:

            wait_seconds = 2 ** (
                attempt - 1
            )

            print(
                f"[retry] {ticker}: "
                f"attempt {attempt} failed; "
                f"retrying in {wait_seconds}s"
            )

            time.sleep(
                wait_seconds
            )

    return (
        None,
        None,
        last_error,
    )


def download_prices(
    tickers,
    period,
    batch_size,
):
    """
    Download market data with graceful recovery.

    Flow:
    1. Download in batches.
    2. Detect tickers with missing Close data.
    3. Retry failed tickers individually.
    4. Keep successfully recovered tickers.
    5. Report final failures without killing the full run.
    """

    tickers = list(
        dict.fromkeys(tickers)
    )

    all_close = []
    all_volume = []

    failed_candidates = set()

    # =========================================================
    # Batch downloads
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

            close, volume = (
                _normalize_download(data)
            )

        except Exception as exc:

            print(
                "[batch error] "
                f"{batch[0]} ... "
                f"{batch[-1]}: "
                f"{type(exc).__name__}: {exc}"
            )

            failed_candidates.update(
                batch
            )

            continue

        # -----------------------------------------------------
        # yfinance column names
        # -----------------------------------------------------

        if not close.empty:

            # Multi-ticker output should already have ticker names.
            # For one-item batch, normalize the generic column name.
            if (
                len(batch) == 1
                and len(close.columns) == 1
            ):

                close.columns = [
                    batch[0]
                ]

                volume.columns = [
                    batch[0]
                ]

            valid_close = []

            for ticker in batch:

                if ticker not in close.columns:
                    failed_candidates.add(
                        ticker
                    )
                    continue

                series = (
                    close[ticker]
                    .dropna()
                )

                if series.empty:
                    failed_candidates.add(
                        ticker
                    )
                    continue

                valid_close.append(
                    ticker
                )

            if valid_close:

                all_close.append(
                    close[
                        valid_close
                    ]
                )

                available_volume = [
                    ticker
                    for ticker in valid_close
                    if ticker
                    in volume.columns
                ]

                volume_out = pd.DataFrame(
                    index=close.index,
                    columns=valid_close,
                    dtype=float,
                )

                if available_volume:

                    volume_out[
                        available_volume
                    ] = volume[
                        available_volume
                    ]

                all_volume.append(
                    volume_out
                )

        else:

            failed_candidates.update(
                batch
            )

    # =========================================================
    # Combine successful batch data
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

    # =========================================================
    # Detect any missing ticker even if yfinance did not raise
    # =========================================================

    successful = set(
        close_all.columns
    )

    failed_candidates.update(
        set(tickers) - successful
    )

    # =========================================================
    # Individual retries
    # =========================================================

    final_failed = []

    if failed_candidates:

        print(
            "\n[recovery] "
            f"retrying {len(failed_candidates)} "
            "ticker(s) individually..."
        )

    for ticker in sorted(
        failed_candidates
    ):

        close_series, volume_series, error = (
            _download_single(
                ticker,
                period,
                max_retries=3,
            )
        )

        if close_series is None:

            final_failed.append(
                (
                    ticker,
                    error,
                )
            )

            continue

        # 如果 batch 階段曾產生同名欄位，
        # retry 成功後以 retry 結果覆蓋。
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
    # Final cleanup
    # =========================================================

    if close_all.empty:

        raise RuntimeError(
            "No market price data could be downloaded."
        )

    # 移除完全沒有價格資料的欄位
    close_all = close_all.dropna(
        axis=1,
        how="all",
    )

    # Volume 對齊 Close，但缺量保持 NaN
    volume_all = volume_all.reindex(
        index=close_all.index,
        columns=close_all.columns,
    )

    # 排序，讓結果每次比較穩定
    close_all = close_all.sort_index(
        axis=1
    )

    volume_all = volume_all.reindex(
        columns=close_all.columns
    )

    # =========================================================
    # Final failure report
    # =========================================================

    if final_failed:

        print(
            "\nWARNING: "
            f"{len(final_failed)} ticker(s) "
            "still failed after retry:"
        )

        for ticker, error in final_failed:

            print(
                f"  - {ticker}: {error}"
            )

    else:

        if failed_candidates:

            print(
                "\n[recovery] "
                "all failed tickers recovered."
            )

    return (
        close_all,
        volume_all,
    )
