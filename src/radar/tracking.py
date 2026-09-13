from pathlib import Path

import numpy as np
import pandas as pd


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


def _safe_value(row, column, default=np.nan):

    if column not in row.index:
        return default

    return row[column]


def _normalize_date(value):

    return pd.Timestamp(value).normalize()


def _find_signal_position(
    close_series,
    signal_date,
):

    valid = close_series.dropna()

    if valid.empty:
        return None, valid

    index = pd.DatetimeIndex(valid.index)

    signal_date = _normalize_date(
        signal_date
    )

    matches = np.where(
        index.normalize()
        == signal_date
    )[0]

    if len(matches) == 0:
        return None, valid

    return int(matches[-1]), valid


def _forward_return(
    valid_close,
    signal_position,
    horizon,
):

    target_position = (
        signal_position + horizon
    )

    if target_position >= len(valid_close):
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


def _update_one_signal(
    row,
    close,
):

    ticker = row["ticker"]

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

    row["day0_price"] = day0_price
    row["lot_cost"] = (
        day0_price * 1000
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

    if available_forward_days >= 0:

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


def _append_new_signals(
    tracking,
    ignition,
    data_date,
):

    if ignition.empty:
        return tracking

    existing_keys = set()

    if not tracking.empty:

        existing_keys = set(
            zip(
                tracking[
                    "signal_date"
                ].astype(str),
                tracking[
                    "ticker"
                ].astype(str),
            )
        )

    new_rows = []

    for _, signal in ignition.iterrows():

        ticker = str(
            signal["ticker"]
        )

        signal_date = str(
            data_date
        )

        key = (
            signal_date,
            ticker,
        )

        # -----------------------------------------------------
        # 同一股票若已經有任何歷史 Day 0，
        # 不因連續入榜而重新建立 Day 0。
        #
        # 也就是：
        # First Ignition = Day 0
        # -----------------------------------------------------

        already_tracked = False

        if not tracking.empty:

            already_tracked = (
                tracking[
                    "ticker"
                ].astype(str)
                == ticker
            ).any()

        if already_tracked:
            continue

        if key in existing_keys:
            continue

        price = float(
            signal["price"]
        )

        new_rows.append(
            {
                "signal_date": signal_date,
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
                "lot_cost": price * 1000,
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
        )

    if not new_rows:
        return tracking

    new_df = pd.DataFrame(
        new_rows
    )

    if tracking.empty:
        return new_df[
            TRACKING_COLUMNS
        ]

    return pd.concat(
        [
            tracking,
            new_df,
        ],
        ignore_index=True,
    )[
        TRACKING_COLUMNS
    ]


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

    if tracking_path.exists():

        try:

            tracking = pd.read_csv(
                tracking_path,
                encoding="utf-8-sig",
            )

        except Exception:

            tracking = (
                _empty_tracking()
            )

    else:

        tracking = (
            _empty_tracking()
        )

    # ---------------------------------------------------------
    # Schema migration
    #
    # 未來增加欄位時，
    # 舊 tracking.csv 仍可繼續使用。
    # ---------------------------------------------------------

    for column in TRACKING_COLUMNS:

        if column not in tracking.columns:
            tracking[column] = np.nan

    tracking = tracking[
        TRACKING_COLUMNS
    ].copy()

    # ---------------------------------------------------------
    # 先加入今天首次出現的 Ignition
    # ---------------------------------------------------------

    tracking = _append_new_signals(
        tracking,
        ignition,
        data_date,
    )

    # ---------------------------------------------------------
    # 再用目前已有行情更新所有歷史 Signal
    # ---------------------------------------------------------

    if not tracking.empty:

        updated_rows = []

        for _, row in tracking.iterrows():

            updated_rows.append(
                _update_one_signal(
                    row.copy(),
                    close,
                )
            )

        tracking = pd.DataFrame(
            updated_rows
        )

    tracking = tracking[
        TRACKING_COLUMNS
    ]

    if not tracking.empty:

        tracking = tracking.sort_values(
            [
                "signal_date",
                "ignition_score",
            ],
            ascending=[
                False,
                False,
            ],
        )

    tracking.to_csv(
        tracking_path,
        index=False,
        encoding="utf-8-sig",
    )

    return tracking
