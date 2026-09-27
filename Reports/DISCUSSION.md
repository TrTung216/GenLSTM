# Discussion

## Final Phase-1 finding

The corrected Phase-1 experiment does not support a claim that GA-WOA CNN-LSTM-Attention is uniformly superior to the simpler baselines. After rerunning GA-WOA under fold-local walk-forward preprocessing and evaluating all models on fixed raw-timeline target boundaries, CNN-LSTM and the optimized attention model are very close on error metrics.

CNN-LSTM obtains the lowest mean RMSE (0.018083), while GA-WOA CNN-LSTM-Attention obtains the lowest mean MAE (0.012220) and the highest mean directional accuracy (47.92%). The differences in RMSE and MAE between those two configurations are small.

This is a more conservative result than the earlier pre-final-protocol experiment and is the result used for the final Phase-1 interpretation.

## Architecture contribution

The controlled architecture ablation is especially informative because all variants use the same final GA-WOA hyperparameters.

Moving from LSTM to CNN-LSTM reduces mean RMSE from 0.018143 to 0.017944 and mean MAE from 0.012312 to 0.012084. Adding attention produces RMSE 0.018026 and MAE 0.012144, which does not improve on CNN-LSTM. CNN-LSTM and CNN-LSTM-Attention both obtain mean directional accuracy of approximately 48.09%.

Therefore, the Phase-1 evidence suggests that the convolutional component is useful under the tested configuration, while attention does not provide a consistent additional benefit. Architectural complexity should not be assumed to improve forecasting performance automatically.

## Effect of GA-WOA

GA-WOA remains useful as a systematic hyperparameter-search procedure. The final search selected `[32, 0.05, 0.001, 16, 20, 64, 1]` with walk-forward fitness 0.6044.

However, the held-out benchmark shows that the searched CNN-LSTM-Attention configuration is competitive rather than clearly dominant. This distinction matters: optimization can identify a viable configuration without proving that the optimizer or the optimized architecture is universally better than simpler alternatives.

The final result also illustrates why the held-out test set must remain separate from search. A high validation/search fitness does not guarantee superiority on unseen market observations.

## Directional prediction

Mean directional accuracy is below 50% for every configuration in the final benchmark. The optimized model reaches 47.92% ± 1.22%.

Consequently, Phase 1 should not claim reliable next-session direction classification or trading advantage. RMSE and MAE indicate that predictions can remain numerically close to realized returns even when the sign is incorrect.

## Fitness-objective observations

The fitness-profile ablation remains exploratory. Profile B produced the lowest RMSE and MAE in its single historical run, while the drawdown component in profile C did not improve held-out metrics.

Because each profile was searched once and the experiment predates the final full-protocol rerun, it is evidence about objective sensitivity rather than evidence that one objective is generally optimal. Raw fitness values across profiles are also not directly comparable because the component weights differ.

## Evaluation design

Phase 1 uses fixed train/validation/test target boundaries on the cleaned raw timeline before lookback generation. This ensures that different lookback windows are evaluated on identical validation and test dates.

During GA-WOA search, preprocessing is refitted separately inside every expanding walk-forward fold and uses only that fold's training history. The final test period is excluded from hyperparameter selection and early stopping.

These choices reduce temporal leakage and make the negative as well as positive findings more credible. They do not eliminate the limitations of a single-asset historical study.
