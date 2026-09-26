import torch.nn as nn


class LSTMModel(nn.Module):
    """Plain LSTM baseline."""

    def __init__(
        self,
        input_size,
        hidden_layer_size=64,
        dropout_rate=0.1,
        num_layers=1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        last_hidden = lstm_out[:, -1, :]
        return self.linear(self.dropout(last_hidden))


class CNNLSTMModel(nn.Module):
    """CNN + LSTM baseline without attention."""

    def __init__(
        self,
        input_size,
        hidden_layer_size=64,
        dropout_rate=0.1,
        cnn_filters=32,
        num_layers=1,
    ):
        super().__init__()
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
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        x = input_seq.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.linear(self.dropout(last_hidden))
