# Experimental Results

## Evaluation protocol

The final benchmark compares four forecasting configurations on AAPL:

- LSTM
- CNN-LSTM
- CNN-LSTM-Attention
- GA-WOA CNN-LSTM-Attention

Five random seeds are used: `42, 123, 2026, 7, 99`. The raw chronological timeline is split into fixed train, validation, and test target regions **before** sliding-window sequences are constructed. Consequently, models with different lookback windows are evaluated on the same target dates. The final held-out test partition contains **587 observations**.

Preprocessing scalers are fitted using training history only. During GA-WOA search, walk-forward validation uses expanding folds and independently refits the scalers inside each fold. Early stopping uses validation data; the held-out test partition is reserved for final evaluation.

Reported values are mean ± sample standard deviation across the five benchmark seeds.

## Final benchmark

| Model | RMSE ↓ | MAE ↓ | Directional Accuracy ↑ |
| --- | ---: | ---: | ---: |
| LSTM | 0.018400 ± 0.000167 | 0.012629 ± 0.000220 | 47.82% ± 1.44% |
| CNN-LSTM | 0.018136 ± 0.000152 | 0.012300 ± 0.000180 | 47.58% ± 1.15% |
| CNN-LSTM-Attention | 0.018139 ± 0.000136 | 0.012321 ± 0.000183 | 46.76% ± 1.22% |
| **GA-WOA CNN-LSTM-Attention** | **0.017881 ± 0.000021** | **0.011940 ± 0.000044** | **52.56% ± 2.91%** |

Within this experiment, the GA-WOA-selected configuration has the lowest mean RMSE and MAE and the highest mean directional accuracy of the four evaluated configurations.

Relative to the fixed CNN-LSTM-Attention baseline, its mean RMSE is approximately **1.42% lower**, mean MAE approximately **3.10% lower**, and mean directional accuracy approximately **5.80 percentage points higher**. Relative to CNN-LSTM, the corresponding differences are approximately **1.41% lower RMSE**, **2.93% lower MAE**, and **4.98 percentage points higher directional accuracy**.

The optimized configuration is particularly stable for magnitude-based errors: RMSE and MAE have standard deviations of only 0.000021 and 0.000044. Directional accuracy is less stable, with a standard deviation of 2.91 percentage points.

## Fitness-function ablation

Three GA-WOA fitness profiles were evaluated in the existing fixed-seed ablation experiment:

| Profile | Objective | RMSE ↓ | MAE ↓ | Directional Accuracy ↑ |
| --- | --- | ---: | ---: | ---: |
| A | 0.50 RMSE + 0.50 MAE | 0.018249 | 0.012347 | 48.96% |
| B | 0.50 RMSE + 0.50 Directional Accuracy | 0.017943 | 0.012007 | 48.88% |
| C | 0.40 RMSE + 0.40 Directional Accuracy + 0.20 Drawdown | 0.018201 | 0.012374 | 45.09% |

This ablation is a **single-run experiment**. It is therefore descriptive rather than sufficient evidence that one fitness objective is generally superior. Profile B produced the lowest RMSE and MAE in this run, while profile A had only a marginal directional-accuracy advantage over B. Adding the drawdown component in profile C did not improve held-out performance in this experiment.

## Reproducibility artifacts

Detailed per-seed outputs are stored in `experiments/benchmark_results.csv`, while aggregated statistics are stored in `experiments/benchmark_summary.csv`. Comparison and GA-WOA diagnostic figures are available under `experiments/plots/`.
