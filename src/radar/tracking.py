from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================
# Tracking Schema
#
# First Ignition = Day 0
#
# 目前只追蹤第一次進入 Ignition Top N 的訊號。
# 同一股票連續或再次進榜，不重新建立 Day 0。
# =============================================================

TRACKING_COLUMNS = [
    "signal_date",
    "ticker",
    "code",
    "name",
    "market",
    "day0_price",
    "lot_cost",
    "ignition_score",
    "ignition_signal_count",
    "day0_ret_5",
    "day0_ret_20",
    "day0_ret_60",
    "day0_volume_ratio_5",
    "day0_distance_to_high_60",
    "d1_return",
    "d3_return",
    "d5_return",
    "d10_return",
    "d20_return",
    "latest_return",
    "latest_price",
    "trading_days_since_signal",
]


def _empty_tracking():

    return pd.DataFrame(
        columns=TRACKING_COLUMNS
    )


def _safe_value(
    row,
    column,
    default=np.nan,
):

    if column not in row.index:
        return default

    return row[column]


# =============================================================
# Signal Date Alignment
#
# yfinance / pandas 的 index 可能帶 timezone，
# signal_date 則可能只是 YYYY-MM-DD。
#
# 統一移除 timezone 並 normalize，
# 避免 Day 0 找不到，造成 D+1 永遠無法更新。
# =============================================================

def _find_signal_position(
    close_series,
    signal_date,
):

    valid = close_series.dropna()

    if valid.empty:
        return None, valid

    index_dates = (
        pd.DatetimeIndex(valid.index)
        .tz_localize(None)
        .normalize()
    )

    signal_date = (
        pd.Timestamp(signal_date)
        .tz_localize(None)
        .normalize()
    )

    matches = np.where(
        index_dates == signal_date
    )[0]

    if len(matches) == 0:
        return None, valid

    return int(
        matches[-1]
    ), valid


# =============================================================
# Forward Return
#
# 定義：
#
# Day 0 收盤價
# → 第 N 個後續交易日收盤價
#
# 注意：
# 這不是下一交易日開盤可實際成交報酬，
# 而是訊號後收盤績效追蹤。
# =============================================================

def _forward_return(
    valid_close,
    signal_position,
    horizon,
):

    target_position = (
        signal_position
        + horizon
    )

    if target_position >= len(
        valid_close
    ):
        return np.nan

    day0_price = float(
        valid_close.iloc[
            signal_position
        ]
    )

    future_price = float(
        valid_close.iloc[
            target_position
        ]
    )

    if day0_price <= 0:
        return np.nan

    return (
        future_price
        / day0_price
        - 1
    )


# =============================================================
# Update One Historical Signal
# =============================================================

def _update_one_signal(
    row,
    close,
):

    ticker = str(
        row["ticker"]
    )

    if ticker not in close.columns:
        return row

    signal_date = row[
        "signal_date"
    ]

    close_series = close[
        ticker
    ]

    (
        signal_position,
        valid_close,
    ) = _find_signal_position(
        close_series,
        signal_date,
    )

    if signal_position is None:
        return row

    day0_price = float(
        valid_close.iloc[
            signal_position
        ]
    )

    row[
        "day0_price"
    ] = day0_price

    # ---------------------------------------------------------
    # 約略整張資金
    #
    # 普通股以 1,000 股估算。
    # 僅供資金量級參考，不作為交易規則判定。
    # 特殊市場／交易制度股票應另行確認。
    # ---------------------------------------------------------

    row[
        "lot_cost"
    ] = (
        day0_price
        * 1000
    )

    available_forward_days = (
        len(valid_close)
        - signal_position
        - 1
    )

    row[
        "trading_days_since_signal"
    ] = max(
        available_forward_days,
        0,
    )

    # ---------------------------------------------------------
    # D+1 / 3 / 5 / 10 / 20
    # ---------------------------------------------------------

    for horizon in [
        1,
        3,
        5,
        10,
        20,
    ]:

        row[
            f"d{horizon}_return"
        ] = _forward_return(
            valid_close,
            signal_position,
            horizon,
        )

    # ---------------------------------------------------------
    # Latest Return
    # ---------------------------------------------------------

    latest_price = float(
        valid_close.iloc[-1]
    )

    row[
        "latest_price"
    ] = latest_price

    if day0_price > 0:

        row[
            "latest_return"
        ] = (
            latest_price
            / day0_price
            - 1
        )

    return row


# =============================================================
# Add Today's New Ignition Signals
# =============================================================

def _append_new_signals(
    tracking,
    ignition,
    data_date,
):

    if ignition.empty:
        return tracking

    for _, signal in ignition.iterrows():

        ticker = str(
            signal["ticker"]
        )

        # -----------------------------------------------------
        # First Ignition Only
        #
        # 同一 ticker 曾經建立過 Day 0，
        # 就不重新建立。
        #
        # 第一階段目的是驗證：
        # Agent 第一次發出 Ignition 時，
        # 後續到底還有沒有肉。
        # -----------------------------------------------------

        if not tracking.empty:

            already_tracked = (
                tracking[
                    "ticker"
                ]
                .astype(str)
                .eq(ticker)
                .any()
            )

            if already_tracked:
                continue

        price = float(
            signal["price"]
        )

        new_row = {
            "signal_date": str(
                data_date
            ),
            "ticker": ticker,
            "code": _safe_value(
                signal,
                "code",
                "",
            ),
            "name": _safe_value(
                signal,
                "name",
                "",
            ),
            "market": _safe_value(
                signal,
                "market",
                "",
            ),
            "day0_price": price,
            "lot_cost": (
                price * 1000
            ),
            "ignition_score": _safe_value(
                signal,
                "ignition_score",
            ),
            "ignition_signal_count": _safe_value(
                signal,
                "ignition_signal_count",
            ),
            "day0_ret_5": _safe_value(
                signal,
                "ret_5",
            ),
            "day0_ret_20": _safe_value(
                signal,
                "ret_20",
            ),
            "day0_ret_60": _safe_value(
                signal,
                "ret_60",
            ),
            "day0_volume_ratio_5": _safe_value(
                signal,
                "volume_ratio_5",
            ),
            "day0_distance_to_high_60": _safe_value(
                signal,
                "distance_to_high_60",
            ),
            "d1_return": np.nan,
            "d3_return": np.nan,
            "d5_return": np.nan,
            "d10_return": np.nan,
            "d20_return": np.nan,
            "latest_return": 0.0,
            "latest_price": price,
            "trading_days_since_signal": 0,
        }

        new_df = pd.DataFrame(
            [new_row]
        )

        if tracking.empty:

            tracking = new_df[
                TRACKING_COLUMNS
            ].copy()

        else:

            tracking = pd.concat(
                [
                    tracking,
                    new_df,
                ],
                ignore_index=True,
            )

    return tracking[
        TRACKING_COLUMNS
    ].copy()


# =============================================================
# Main Tracking Function
# =============================================================

def update_ignition_tracking(
    ignition,
    close,
    data_date,
    reports_dir="reports",
):

    reports = Path(
        reports_dir
    )

    reports.mkdir(
        exist_ok=True
    )

    tracking_path = (
        reports
        / "ignition_tracking.csv"
    )

    # =========================================================
    # Load Existing Tracking
    # =========================================================

    if tracking_path.exists():

        try:

            tracking = pd.read_csv(
                tracking_path,
                encoding="utf-8-sig",
            )

        except Exception as exc:

            print(
                "[tracking] "
                "failed to read existing "
                "ignition_tracking.csv: "
                f"{exc}"
            )

            tracking = (
                _empty_tracking()
            )

    else:

        tracking = (
            _empty_tracking()
        )

    # =========================================================
    # Schema Migration
    #
    # 如果未來增加欄位，
    # 舊 tracking.csv 仍能繼續使用。
    # =========================================================

    for column in TRACKING_COLUMNS:

        if column not in tracking.columns:

            tracking[
                column
            ] = np.nan

    tracking = tracking[
        TRACKING_COLUMNS
    ].copy()

    # =========================================================
    # Add Today's First Ignition Signals
    # =========================================================

    tracking = _append_new_signals(
        tracking,
        ignition,
        data_date,
    )

    # =========================================================
    # Update All Historical Signals
    # =========================================================

    if not tracking.empty:

        updated_rows = []

        for _, row in tracking.iterrows():

            updated_row = (
                _update_one_signal(
                    row.copy(),
                    close,
                )
            )

            updated_rows.append(
                updated_row
            )

        tracking = pd.DataFrame(
            updated_rows
        )

    # =========================================================
    # Normalize Schema
    # =========================================================

    for column in TRACKING_COLUMNS:

        if column not in tracking.columns:

            tracking[
                column
            ] = np.nan

    tracking = tracking[
        TRACKING_COLUMNS
    ].copy()

    # =========================================================
    # Sort
    # =========================================================

    if not tracking.empty:

        tracking[
            "signal_date"
        ] = (
            tracking[
                "signal_date"
            ]
            .astype(str)
        )

        tracking = (
            tracking
            .sort_values(
                [
                    "signal_date",
                    "ignition_score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================
    # Save
    # =========================================================

    tracking.to_csv(
        tracking_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[tracking] "
        f"{len(tracking)} "
        "Ignition Day 0 signals tracked."
    )

    return tracking
