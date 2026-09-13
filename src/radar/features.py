import numpy as np
import pandas as pd


def _safe_ratio(numerator, denominator):
    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator <= 0
    ):
        return np.nan

    return numerator / denominator


def latest_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for ticker in close.columns:

        s = close[ticker].dropna()

        if len(s) < 20:
            continue

        # =====================================================
        # Volume
        #
        # 缺資料維持 NaN，不得 fillna(0)
        # =====================================================

        if ticker in volume.columns:
            v = volume[ticker].reindex(s.index)
        else:
            v = pd.Series(
                np.nan,
                index=s.index,
                dtype=float,
            )

        r = s.pct_change()

        row = {
            "ticker": ticker,
            "price": s.iloc[-1],
            "history_days": len(s),
        }

        # =====================================================
        # Returns
        # =====================================================

        for h in (5, 10, 20, 60, 120, 252):

            row[f"ret_{h}"] = (
                s.iloc[-1] / s.iloc[-h - 1] - 1
                if len(s) > h
                else np.nan
            )

        # =====================================================
        # Moving averages
        # =====================================================

        for h in (20, 60, 120):

            if len(s) >= h:

                ma = s.tail(h).mean()

                row[f"above_ma_{h}"] = (
                    s.iloc[-1] / ma - 1
                    if pd.notna(ma) and ma > 0
                    else np.nan
                )

            else:
                row[f"above_ma_{h}"] = np.nan

        # =====================================================
        # Liquidity
        # =====================================================

        turnover = s * v

        valid_turnover_20 = (
            turnover.tail(20).dropna()
        )

        # 至少 15 個有效交易日
        row["avg_turnover_20"] = (
            valid_turnover_20.mean()
            if len(valid_turnover_20) >= 15
            else np.nan
        )

        # =====================================================
        # Volume ratios
        # =====================================================

        recent_vol_5 = (
            v.tail(5).dropna()
        )

        prior_vol_20 = (
            v.iloc[-25:-5].dropna()
            if len(v) >= 25
            else pd.Series(dtype=float)
        )

        recent_vol_20 = (
            v.tail(20).dropna()
        )

        previous_vol_20 = (
            v.iloc[-40:-20].dropna()
            if len(v) >= 40
            else pd.Series(dtype=float)
        )

        # 最近 5 日平均量 / 前 20 日平均量
        if (
            len(recent_vol_5) >= 4
            and len(prior_vol_20) >= 15
        ):

            row["volume_ratio_5"] = _safe_ratio(
                recent_vol_5.mean(),
                prior_vol_20.mean(),
            )

        else:
            row["volume_ratio_5"] = np.nan

        # 最近 20 日平均量 / 前 20 日平均量
        if (
            len(recent_vol_20) >= 15
            and len(previous_vol_20) >= 15
        ):

            row["volume_ratio_20"] = _safe_ratio(
                recent_vol_20.mean(),
                previous_vol_20.mean(),
            )

        else:
            row["volume_ratio_20"] = np.nan

        # =====================================================
        # Previous highs
        #
        # 高點基準不包含今天
        # =====================================================

        if len(s) >= 61:

            prior_high_60 = (
                s.iloc[-61:-1].max()
            )

            price_vs_high_60 = (
                s.iloc[-1] / prior_high_60 - 1
                if pd.notna(prior_high_60)
                and prior_high_60 > 0
                else np.nan
            )

            # 同一個原始數值，不同語意名稱
            row["breakout_60"] = price_vs_high_60
            row["distance_to_high_60"] = price_vs_high_60

        else:

            row["breakout_60"] = np.nan
            row["distance_to_high_60"] = np.nan

        if len(s) >= 121:

            prior_high_120 = (
                s.iloc[-121:-1].max()
            )

            row["breakout_120"] = (
                s.iloc[-1] / prior_high_120 - 1
                if pd.notna(prior_high_120)
                and prior_high_120 > 0
                else np.nan
            )

        else:
            row["breakout_120"] = np.nan

        # =====================================================
        # Momentum shift proxy
        #
        # 僅供橫斷面排序，
        # 不解讀為真正的「加速度百分比」
        # =====================================================

        row["momentum_shift_5_20"] = (
            row["ret_5"]
            - row["ret_20"] / 4
            if pd.notna(row["ret_5"])
            and pd.notna(row["ret_20"])
            else np.nan
        )

        row["momentum_shift_20_60"] = (
            row["ret_20"]
            - row["ret_60"] / 3
            if pd.notna(row["ret_20"])
            and pd.notna(row["ret_60"])
            else np.nan
        )

        # =====================================================
        # Risk
        # =====================================================

        row["volatility_20"] = (
            r.tail(20).std()
            * np.sqrt(252)
        )

        if len(s) >= 60:

            last60 = s.tail(60)

            peak = last60.cummax()

            row["max_drawdown_60"] = (
                last60 / peak - 1
            ).min()

        else:
            row["max_drawdown_60"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def features_on_date(
    close,
    volume,
    date,
):

    c = close.loc[:date]
    v = volume.loc[:date]

    return latest_features(
        c,
        v,
    )
