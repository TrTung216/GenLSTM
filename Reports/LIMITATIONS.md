# Limitations and Threats to Validity

## Dataset and temporal scope

The final Phase-1 study is centered on **AAPL** and one historical period. Results do not establish generalization to other stocks, sectors, indices, asset classes, or future market regimes.

## Forecasting horizon

The target is the next-session return. Multi-step forecasting, recursive prediction, and longer investment horizons are outside Phase 1.

## Feature scope

The 17-feature input is primarily OHLCV-derived technical information plus VIX, TNX, and a CMF-based proxy stored as `Sentiment_Score`. It is **not text-derived sentiment**. Phase 1 therefore makes no claim about the value of news, social-media, or language-model sentiment.

## Optimizer uncertainty

GA-WOA is stochastic. The final benchmark repeats model training over five seeds, but the complete GA-WOA hyperparameter search itself is not repeated over multiple independent search seeds. Optimizer-level uncertainty is therefore not quantified.

The search is restricted to predefined ranges. Better configurations may exist outside the tested space.

## Directional performance

All final benchmark configurations have mean directional accuracy below 50%. The project therefore does not establish reliable directional prediction. Small changes around zero can also change prediction sign without strongly affecting RMSE or MAE.

## Statistical inference

The reported mean and sample standard deviation describe five training seeds evaluated on the same 587 historical test targets. They are not five independent market datasets. Phase 1 does not include formal inference over independent assets or market periods.

## Fitness ablation

The objective-profile ablation is single-run and predates the final complete rerun of the corrected Phase-1 protocol. It is retained as exploratory analysis and should not be interpreted as a definitive comparison of fitness functions.

## Trading interpretation

RMSE, MAE, and directional accuracy are predictive metrics, not evidence of profitability. Transaction costs, slippage, bid-ask spread, turnover, position sizing, portfolio construction, and risk-adjusted returns are not modeled.

## Data source

The pipeline depends on Yahoo Finance through `yfinance`. Upstream corrections, adjustments, missing observations, API changes, or different download dates can affect exact reproducibility.

## Model comparison scope

Phase 1 compares LSTM-family models and a GA-WOA hyperparameter search. It is not a comprehensive benchmark against Transformer-based forecasting, statistical time-series models, gradient boosting, or other modern methods. Those comparisons are outside the Phase-1 scope.

## Reproducibility boundary

Fixed raw-timeline target boundaries, fold-local preprocessing, walk-forward validation, validation-only early stopping, saved outputs, and multi-seed evaluation reduce important validity threats. They do not remove the broader limitations associated with single-asset historical evaluation and a single final optimizer search.
