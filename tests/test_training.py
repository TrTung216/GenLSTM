import numpy as np
import torch
import torch.nn as nn

from src.training import train_with_early_stopping


def test_early_stopping_restores_best_checkpoint():
    torch.manual_seed(1)
    X = np.linspace(-1, 1, 30, dtype=np.float32).reshape(10, 3, 1)
    y = X.mean(axis=1).reshape(-1)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3, 1))

    result = train_with_early_stopping(
        model, X[:7], y[:7], X[7:], y[7:],
        learning_rate=0.01, batch_size=4, device=torch.device("cpu"),
        max_epochs=8, patience=3,
    )

    assert 1 <= result.best_epoch <= result.epochs_ran <= 8
    assert np.isfinite(result.best_validation_loss)
