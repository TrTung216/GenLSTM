# Methodology

## 1. Research objective

Phase 1 studies next-session stock-return forecasting with a hybrid **CNN-LSTM-Attention** network whose hyperparameters are optimized by a **Genetic Algorithm–Whale Optimization Algorithm (GA-WOA)** procedure. The main experimental question is whether hyperparameter search improves held-out forecasting performance relative to fixed LSTM-based baselines under a leakage-aware temporal evaluation protocol.

The current study uses AAPL as the primary asset. Results are evaluated on returns rather than dollar prices.

## 2. Input data and features

Daily OHLCV data are downloaded with `yfinance`. Two macro-market series are aligned to the equity timeline: VIX and TNX. Missing macro observations are forward/backward filled after alignment.

The feature vector contains 17 variables:

`Open, High, Low, Close, log(1 + Volume), SMA_10, SMA_20, EMA_20, RSI_14, MACD, Signal_Line, BB_Middle, BB_Upper, BB_Lower, VIX, TNX, Sentiment_Score`.

`Sentiment_Score` is currently a **Chaikin Money Flow (CMF) proxy derived from OHLCV**, not text sentiment. Rows made incomplete by rolling indicators are removed.

The prediction target is the one-step simple return

[
r_t = \frac{C_t-C_{t-1}}{C_{t-1}},
]

where (C_t) is the closing price at time (t).

## 3. Temporal partitioning and leakage control

The final benchmark fixes train, validation, and test boundaries on the cleaned raw timeline **before sliding-window generation**. With validation fraction 0.15 and test fraction 0.20, the chronological partitions are approximately 65% / 15% / 20%.

For a lookback (w), each sample uses observations (t-w,\ldots,t-1) to predict the return at target time (t). Because target boundaries are fixed before sequences are generated, different lookback values predict the same validation and test dates.

Feature scaling uses `RobustScaler`; target scaling uses `StandardScaler`. Both are fitted only on training history. Validation and test observations never contribute statistics to final benchmark preprocessing.

## 4. Walk-forward validation during GA-WOA search

Hyperparameters are selected only from the development region; the held-out test region is excluded from search.

The development region is evaluated using three expanding walk-forward folds. For every fold:

1. determine the chronological training and validation ranges;
2. instantiate fresh feature and target scalers;
3. fit both scalers only through that fold's final training target;
4. transform the fold data;
5. train the candidate network;
6. compute validation fitness.

The chromosome fitness is the mean validation fitness across the three folds. This fold-local scaling prevents later validation statistics from leaking into earlier folds.

## 5. CNN-LSTM-Attention architecture

For input (X\in\mathbb{R}^{w\times d}):

1. a one-dimensional convolution extracts local temporal patterns;
2. ReLU provides nonlinear activation;
3. an LSTM models sequential dependencies;
4. a learned linear attention score is applied to every LSTM output;
5. softmax-normalized attention weights produce a weighted context vector;
6. dropout is applied to the context vector;
7. a linear layer outputs the predicted scaled return.

For hidden states (h_t), the attention mechanism is

[
e_t=W_a h_t+b_a,
qquad
\alpha_t=\frac{\exp(e_t)}{\sum_j\exp(e_j)},
qquad
c=\sum_t\alpha_t h_t.
]

The prediction is obtained from a linear projection of the dropout-regularized context vector (c).

## 6. GA-WOA hyperparameter optimization

Each chromosome contains seven genes:

| Gene | Search space |
| --- | --- |
| LSTM hidden units | 32, 64, 96, 128 |
| Dropout | initial 0.05, 0.10, 0.15; WOA may refine continuously within [0, 0.5] |
| Learning rate | 0.0001, 0.0005, 0.001, 0.005 |
| Batch size | 16, 32, 64 |
| Lookback window | 20, 30, 45, 60 |
| CNN filters | 16, 32, 64 |
| LSTM layers | 1, 2 |

The search uses population size 20 for 15 generations, crossover probability 0.8, tournament size 3, elitism, dynamic mutation, and WOA refinement of part of the population.

The Phase-1 default objective is

[
F=0.40S_{RMSE}+0.40S_{DA}+0.20S_{DD},
]

where the components are normalized scores for RMSE, directional accuracy, and drawdown behavior. Higher fitness is preferred.

## 7. Training procedure

Models are optimized with Adam and Huber loss. Mini-batches preserve chronological order (`shuffle=False`). Training is capped at 120 epochs and uses validation-based early stopping with patience 15 in the final benchmark. The checkpoint with the lowest validation loss is restored before test evaluation.

Randomness is controlled for Python, NumPy, and PyTorch. CUDA deterministic mode is enabled where supported.

## 8. Baselines and final benchmark

The benchmark contains:

- LSTM;
- CNN-LSTM;
- CNN-LSTM-Attention;
- GA-WOA CNN-LSTM-Attention.

Fixed baselines use a shared manually specified configuration. The optimized model uses the configuration saved by the GA-WOA search. Final training is repeated for seeds `42, 123, 2026, 7, 99`.

The held-out test set is not used for hyperparameter selection or early stopping.

A separate controlled architecture ablation uses the same hyperparameters and target boundaries across the LSTM-family variants so that the contribution of CNN and attention can be examined without conflating architecture with different training settings.

## 9. Evaluation metrics

For (n) observations with true returns (y_i) and predictions (hat y_i):

[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2},
]

[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|.
]

Directional accuracy is the fraction of non-flat targets whose predicted return has the same sign:

[
DA=\frac{1}{N}\sum I(\operatorname{sign}(y_i)=\operatorname{sign}(\hat y_i)).
]

Final benchmark results are reported as **mean ± sample standard deviation** across five training seeds. These repetitions measure training-seed variability; they are not independent market samples and are not treated as such.

## 10. Reproducibility

Primary commands are:

```bash
python -m pytest -q
python -m src.ga_lstm
python -m src.benchmark
python -m src.architecture_ablation
```

Raw per-seed benchmark results, summaries, GA generation statistics, ablation outputs, and plots are stored under `experiments/`. Model weights, preprocessing scalers, and the selected configuration are stored under `artifacts/`.

The Phase-1 interpretation is deliberately limited to the tested AAPL dataset, historical period, feature pipeline, search space, and evaluation protocol. Predictive metrics are not interpreted as evidence of trading profitability.
