import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_baseline(nn.Module):
    def __init__(
        self,
        num_freqs: int,
        feature_dim_per_freq: int,
        num_sensor_pairs: int,
        lstm_hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        attention_dim: int = 256,
        decoder_hidden_dim: int = 256,
        output_dim: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_sensor_pairs = num_sensor_pairs
        self.num_freqs = num_freqs
        self.feature_dim_per_freq = feature_dim_per_freq

        self.lstm_encoder = nn.LSTM(
            input_size=feature_dim_per_freq,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
        )

        lstm_output_dim = lstm_hidden_dim * 2
        self.attention_w = nn.Linear(lstm_output_dim, attention_dim)
        self.attention_v = nn.Linear(attention_dim, 1, bias=False)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(decoder_hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(decoder_hidden_dim // 2, output_dim),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        batch_size = pair_features.shape[0]
        lstm_input = pair_features.view(-1, self.num_freqs, self.feature_dim_per_freq)
        _, (h_n, _) = self.lstm_encoder(lstm_input)
        lstm_outputs = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        pair_embeddings = lstm_outputs.view(batch_size, self.num_sensor_pairs, -1)

        attention_scores = self.attention_v(
            torch.tanh(self.attention_w(pair_embeddings))
        )
        attention_weights = F.softmax(attention_scores, dim=1)
        graph_embedding = (attention_weights * pair_embeddings).sum(dim=1)

        prediction = self.decoder_mlp(graph_embedding)
        return prediction
