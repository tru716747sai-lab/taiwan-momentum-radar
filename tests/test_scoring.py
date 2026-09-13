import pandas as pd
from radar.scoring import score

def test_score_order():
    df = pd.DataFrame({
        "ticker":["A","B"], "ret_20":[.2,.1], "ret_60":[.3,.1],
        "ret_120":[.4,.1], "ret_252":[.5,.1],
        "above_ma_20":[.1,.01], "above_ma_60":[.1,.01], "above_ma_120":[.1,.01],
        "avg_turnover_20":[2e8,1e8], "volatility_20":[.2,.3],
        "max_drawdown_60":[-.05,-.2]
    })
    cfg={"score":{"momentum_horizons":{"20":.1,"60":.25,"120":.3,"252":.35},
                  "momentum_weight":.45,"trend_weight":.25,
                  "liquidity_weight":.15,"risk_weight":.15}}
    assert score(df,cfg).iloc[0]["ticker"] == "A"
