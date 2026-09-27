# Discussion

## Effect of architecture

The benchmark does not show an automatic benefit from adding the attention layer. CNN-LSTM and CNN-LSTM-Attention have almost identical mean RMSE (0.018136 versus 0.018139) and similar MAE, while the fixed attention model has lower mean directional accuracy (46.76% versus 47.58%).

This is useful negative evidence: additional architectural complexity alone is not sufficient to improve forecasting quality in this setup. The result also suggests that the contribution observed for the optimized model should not be attributed to attention alone.

## Effect of GA-WOA optimization

The GA-WOA-selected CNN-LSTM-Attention configuration reaches lower mean RMSE and MAE than all three fixed baselines in the five-seed benchmark. It also reaches a higher mean directional accuracy.

A reasonable interpretation is that architecture and hyperparameters interact strongly. GA-WOA searches parameters including hidden size, dropout, learning rate, batch size, lookback window, CNN filters, and LSTM layers instead of relying on one manually fixed configuration. In these experiments, that search produced a configuration that generalizes better to the held-out AAPL test period than the fixed configurations.

This result should not be interpreted as evidence that GA-WOA or CNN-LSTM-Attention is universally superior. The evidence is specific to the current dataset, feature pipeline, search space, validation protocol, and forecasting horizon.

## Stability across seeds

The optimized model has very small variation in RMSE and MAE across five training seeds. This indicates that its prediction-error magnitude is relatively insensitive to the tested initialization randomness.

Directional accuracy behaves differently. Its mean is 52.56%, but its standard deviation is 2.91 percentage points and individual runs vary substantially more than RMSE or MAE. Therefore, directional performance should be reported using the full mean ± standard deviation rather than selecting the strongest individual seed.

The difference between error stability and directional variability is plausible because a prediction can remain numerically close to zero while a small change is sufficient to switch its predicted sign.

## Fitness-objective observations

The fitness ablation provides preliminary evidence that objective design changes the configuration found by GA-WOA. Profile B produced the lowest RMSE/MAE in its single run, whereas the additional drawdown term in profile C did not translate into stronger held-out metrics.

Raw search-fitness values from different profiles should not be compared directly because the objectives use different component weights and therefore different scales. Stronger conclusions about the fitness objective would require repeated independent GA-WOA searches for each profile.

## Evaluation design

Two evaluation details materially strengthen the final benchmark.

First, train, validation, and test target boundaries are fixed on the cleaned raw timeline before lookback sequences are generated. This prevents different window sizes from shifting the observations on which models are evaluated.

Second, GA-WOA walk-forward evaluation refits preprocessing scalers independently within every expanding training fold. Validation observations and the final test period therefore do not contribute statistics to the scaler fitted for a fold.

Together with early stopping on validation data and a held-out final test partition, these choices reduce common forms of temporal leakage and make the comparison more reproducible.
