"""Shared deterministic training helpers."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    epochs_ran: int


def train_with_early_stopping(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    learning_rate,
    batch_size,
    device,
    max_epochs=120,
    patience=10,
    min_delta=0.0,
):
    """Train on train rows and restore the best validation checkpoint."""
    if len(X_train) == 0 or len(X_validation) == 0:
        raise ValueError("Training and validation data must be non-empty")
    model = model.to(device)
    loss_fn = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        ),
        batch_size=int(batch_size),
        shuffle=False,
    )
    X_val = torch.as_tensor(X_validation, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_validation, dtype=torch.float32, device=device).reshape(-1, 1)

    best_loss = np.inf
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    epochs_ran = 0
    for epoch in range(int(max_epochs)):
        model.train()
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(sequences), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(X_val), y_val).item())
        epochs_ran = epoch + 1
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:  # defensive; max_epochs is normally positive
        raise ValueError("max_epochs must be at least 1")
    model.load_state_dict(best_state)
    return TrainingResult(model, best_epoch, best_loss, epochs_ran)
