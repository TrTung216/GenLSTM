import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """CNN + LSTM + Attention model used by both training and inference."""

    def __init__(
        self,
        input_size,
        hidden_layer_size=50,
        dropout_rate=0.2,
        cnn_filters=16,
        num_layers=1,
    ):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            cnn_filters,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_layer_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        x = input_seq.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        return self.linear(self.dropout(context_vector))
