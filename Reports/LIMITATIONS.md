# Limitations and Threats to Validity

## Dataset scope

The current final benchmark is centered on **AAPL**. Performance on one equity does not establish that the same architecture or GA-WOA configuration will generalize to other stocks, sectors, indices, asset classes, or market regimes.

The experiment also covers one historical period. Structural market changes may alter relationships between technical indicators and future returns.

## Forecasting horizon

The project predicts the next-step return. Multi-step forecasting introduces additional uncertainty and error accumulation and is not evaluated here.

## Feature scope

The feature set is primarily OHLCV-derived technical indicators plus VIX, TNX, and a CMF-based sentiment proxy. The current `Sentiment_Score` is not text-derived investor sentiment. Consequently, the experiments do not establish the benefit of news, social-media, or language-model sentiment information.

## Hyperparameter-search uncertainty

GA-WOA is stochastic. Although final model training is evaluated over five seeds, the fitness-profile ablation currently uses one search run per profile. Repeating the entire optimization process under multiple independent search seeds would be required to quantify optimizer-level uncertainty.

The search is also limited to the predefined hyperparameter ranges. A configuration outside those ranges could perform differently.

## Statistical inference

The benchmark reports mean and sample standard deviation across five training seeds, but it does not currently include formal paired significance tests or confidence intervals over independent market samples. The five runs share the same historical test observations, so they should not be interpreted as five independent financial datasets.

## Trading interpretation

RMSE, MAE, and directional accuracy are predictive metrics rather than a complete trading evaluation. The project does not currently establish profitability after transaction costs, slippage, bid-ask spread, position sizing, turnover, or risk constraints.

Directional accuracy slightly above 50% should therefore not be interpreted directly as evidence of a profitable trading strategy.

## Data source

The pipeline relies on Yahoo Finance through `yfinance`. Data availability, adjustments, missing observations, or upstream changes may affect reproducibility.

## Model comparison scope

The benchmark compares LSTM, CNN-LSTM, CNN-LSTM-Attention, and a GA-WOA-optimized CNN-LSTM-Attention configuration. It is not a comprehensive comparison with all modern time-series forecasting methods. The current study deliberately focuses on isolating the effect of the chosen architecture and optimization procedure rather than maximizing the number of model families.

## Reproducibility considerations

Fixed chronological target boundaries, fold-local preprocessing, walk-forward validation, explicit validation-based early stopping, saved experiment CSVs, and multi-seed final evaluation reduce several threats to validity. They do not remove the broader limitations associated with single-asset historical evaluation and stochastic hyperparameter optimization.
