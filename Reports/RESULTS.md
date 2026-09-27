# Experimental Results

## Evaluation protocol

The final Phase-1 benchmark compares LSTM, CNN-LSTM, CNN-LSTM-Attention, and the GA-WOA-selected CNN-LSTM-Attention configuration on AAPL. Five training seeds are used: `42, 123, 2026, 7, 99`.

Train, validation, and test target boundaries are fixed on the cleaned raw timeline before sliding-window generation. All models are therefore evaluated on the same **587 held-out target observations**. Final-benchmark scalers are fitted on training history only. GA-WOA search uses three expanding walk-forward folds with independently refitted fold-local scalers. Early stopping uses validation data; the test partition is reserved for final evaluation.

After the leakage-control and fixed-boundary pipeline was finalized, GA-WOA was rerun from scratch. The final selected chromosome was:

```text
units=32, dropout=0.05, learning_rate=0.001, batch_size=16,
window_size=20, cnn_filters=64, num_layers=1
```

Its walk-forward search fitness was `0.6044`.

## Final benchmark

Values are mean ± sample standard deviation across five training seeds.

| Model | RMSE ↓ | MAE ↓ | Directional Accuracy ↑ |
| --- | ---: | ---: | ---: |
| LSTM | 0.018416 ± 0.000196 | 0.012646 ± 0.000250 | 47.78% ± 1.39% |
| CNN-LSTM | **0.018083 ± 0.000081** | 0.012222 ± 0.000092 | 47.34% ± 0.95% |
| CNN-LSTM-Attention | 0.018139 ± 0.000136 | 0.012321 ± 0.000183 | 46.76% ± 1.22% |
| GA-WOA CNN-LSTM-Attention | 0.018088 ± 0.000142 | **0.012220 ± 0.000183** | **47.92% ± 1.22%** |

The final corrected experiment does **not** show one model dominating every metric. CNN-LSTM has the lowest mean RMSE, while the GA-WOA-selected configuration has the lowest mean MAE and highest mean directional accuracy, but the margins over CNN-LSTM are very small for error magnitude.

Relative to fixed CNN-LSTM-Attention, the GA-WOA configuration reduces mean RMSE by about **0.28%**, reduces mean MAE by about **0.82%**, and increases mean directional accuracy by about **1.16 percentage points**. Relative to CNN-LSTM, GA-WOA has approximately **0.03% higher RMSE**, essentially tied MAE, and about **0.58 percentage points higher** directional accuracy.

Directional accuracy remains below 50% on average for every final benchmark configuration, so Phase 1 does not establish reliable directional prediction.

## Controlled architecture ablation

To separate architecture effects from hyperparameter differences, LSTM, CNN-LSTM, and CNN-LSTM-Attention were retrained using the same final GA-WOA configuration, same five seeds, same training procedure, and same 587 test targets.

| Architecture | RMSE ↓ | MAE ↓ | Directional Accuracy ↑ |
| --- | ---: | ---: | ---: |
| LSTM | 0.018143 ± 0.000048 | 0.012312 ± 0.000112 | 47.65% ± 2.46% |
| CNN-LSTM | **0.017944 ± 0.000070** | **0.012084 ± 0.000092** | **48.09% ± 2.07%** |
| CNN-LSTM-Attention | 0.018026 ± 0.000078 | 0.012144 ± 0.000108 | **48.09% ± 1.13%** |

Under controlled hyperparameters, adding the CNN improves mean RMSE and MAE relative to plain LSTM. Adding attention on top of CNN-LSTM does not improve mean RMSE or MAE; mean directional accuracy is effectively identical to CNN-LSTM.

## Fitness-function ablation

The earlier single-run objective ablation produced:

| Profile | Objective | RMSE ↓ | MAE ↓ | Directional Accuracy ↑ |
| --- | --- | ---: | ---: | ---: |
| A | 0.50 RMSE + 0.50 MAE | 0.018249 | 0.012347 | 48.96% |
| B | 0.50 RMSE + 0.50 Directional Accuracy | 0.017943 | 0.012007 | 48.88% |
| C | 0.40 RMSE + 0.40 Directional Accuracy + 0.20 Drawdown | 0.018201 | 0.012374 | 45.09% |

These results are retained as exploratory evidence only. They came from one search per profile and predate the final complete Phase-1 rerun; they must not be used to claim a generally superior objective.

## Reproducibility artifacts

Final per-seed benchmark outputs are written to `experiments/benchmark_results.csv` and `experiments/benchmark_summary.csv`. Controlled architecture outputs are written to `experiments/architecture_ablation.csv` and `experiments/architecture_ablation_summary.csv`. GA diagnostics and comparison figures are stored under `experiments/plots/`.
