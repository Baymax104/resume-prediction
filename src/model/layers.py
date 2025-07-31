# -*- coding: UTF-8 -*-
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

from setting import SettingManager


class TimeEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, *, input_dim)

        Returns:
            output: (batch_size, *, output_dim)
        """
        return self.mlp(x)


class ResumeEncoder(nn.Module):

    def __init__(self, output_dim: int):
        super().__init__()
        settings = SettingManager.get_settings()
        bert_path = settings.model.pretrained_model_dir / "bert-base-chinese"
        self.bert = BertModel.from_pretrained(bert_path.resolve())
        for parameter in self.bert.parameters():
            parameter.requires_grad = False
        self.projection = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch_size, window_size, seq_len)
            attention_mask: (batch_size, window_size, seq_len)

        Returns:
            output: (batch_size, window_size, output_dim)
        """
        batch_size, window_size, seq_len = input_ids.size()
        input_ids = input_ids.view(batch_size * window_size, seq_len)
        attention_mask = attention_mask.view(batch_size * window_size, seq_len)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.projection(outputs.pooler_output)
        outputs = self.dropout(outputs)
        return outputs.view(batch_size, window_size, -1)


class Decoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, *, input_dim)

        Returns:
            output: (batch_size, *, output_dim)
        """
        return self.mlp(x)


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # (1, window_size, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, d_model)

        Returns:
            output: (batch_size, window_size, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ParameterPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, window_size: int, dropout: float = 0.1):
        super(ParameterPositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, window_size, d_model))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, d_model)

        Returns:
            output: (batch_size, window_size, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerModel, self).__init__()
        self.position_encoding = SinusoidalPositionalEncoding(d_model, dropout)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, d_model)

        Returns:
            output: (batch_size, window_size, d_model)
        """
        x = self.position_encoding(x)
        x = self.transformer(x)
        return x


class AttentionPooling(nn.Module):

    def __init__(self, embedding_dim: int):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, embedding_dim)

        Returns:
            output: (batch_size, embedding_dim)
        """
        # (batch_size, window_size, 1)
        attention_scores = self.attention(x)
        attention_scores = F.softmax(attention_scores, dim=1)

        # (batch_size, embedding_dim)
        output = torch.sum(x * attention_scores, dim=1)
        return output


class LSTMModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, window_size, input_dim)

        Returns:
            output: (batch_size, window_size, hidden_dim)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        return output


class ResumePredictor(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super(ResumePredictor, self).__init__()
        self.time_encoder = TimeEncoder(input_dim=1, hidden_dim=256, output_dim=d_model)
        self.resume_encoder = ResumeEncoder(output_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.transformer = TransformerModel(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.lstm = LSTMModel(
            input_dim=d_model,
            hidden_dim=d_model,
            num_layers=num_layers,
        )

        self.pool = AttentionPooling(embedding_dim=embedding_dim)
        self.decoder = Decoder(input_dim=d_model, hidden_dim=2048, output_dim=embedding_dim)

    def forward(self, input_ids, attention_mask, resume_time):
        """
        Args:
            input_ids: (batch_size, window_size, seq_len)
            attention_mask: (batch_size, window_size, seq_len)
            resume_time: (batch_size, window_size, 1)

        Returns:
            output: (batch_size, embedding_dim)
        """

        # (batch_size, window_size, d_model)
        time_features = self.time_encoder(resume_time)
        # (batch_size, window_size, d_model)
        resume_features = self.resume_encoder(input_ids, attention_mask)

        # (batch_size, window_size, d_model * 2)
        x = torch.cat([time_features, resume_features], dim=-1)

        # (batch_size, window_size, d_model)
        x = self.projection(x)
        y = self.transformer(x)
        # y = self.lstm(x)
        x = x + y
        x = self.layer_norm(x)

        # (batch_size, window_size, embedding_dim)
        x = self.decoder(x)

        # pool for window_size dimension
        x = self.pool(x)  # (batch_size, embedding_dim)
        x = self.dropout(x)
        return x
