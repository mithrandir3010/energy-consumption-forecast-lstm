import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    """
    LSTM tabanlı 1-adım ileri zaman serisi tahmin modeli.
    Girdi: (batch, seq_len, 1)
    Çıktı: (batch, 1)
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)   # (B, seq, hidden)
        out = out[:, -1, :]     # last step
        out = self.fc(out)      # (B, 1)
        return out
