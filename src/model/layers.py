# -*- coding: UTF-8 -*-
import math
from pathlib import Path

import torch
from torch import nn
from transformers import BertModel


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
        bert_path = Path(__file__).parent.parent.parent / "model" / "bert-base-chinese"
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


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model: int,
        window_size: int,
        dropout: float = 0.1,
    ):
        super(PositionalEncoding, self).__init__()
        self.window_size = window_size
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(window_size, d_model)
        position = torch.arange(0, window_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # (1, window_size, d_model)
        self.register_buffer('pe', pe)
        # (1, window_size, 1)
        self.register_buffer('position_weights', self.__position_weights(window_size))

    def __position_weights(self, window_size: int):
        positions = torch.arange(window_size, dtype=torch.float)
        weights = positions + 1
        weights = weights / weights.sum()
        weights = weights.view(1, window_size, 1)
        return weights

    def __get_time_weights(self, resume_time):
        sums = torch.sum(resume_time, dim=1, keepdim=True)
        sums = torch.where(sums == 0, torch.ones_like(sums) * 1e-8, sums)
        time_weights = resume_time / sums
        return time_weights

    def forward(self, x, resume_time):
        """
        Args:
            x: (batch_size, window_size, d_model)
            resume_time: (batch_size, window_size, 1)

        Returns:
            output: (batch_size, window_size, d_model)
        """
        time_weights = self.__get_time_weights(resume_time)
        weights = self.position_weights * time_weights
        weighted_pe = self.pe * weights
        x = x + weighted_pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        window_size: int = 2,
        dropout: float = 0.1,
    ):
        super(TransformerModel, self).__init__()
        self.position_encoding = PositionalEncoding(d_model, window_size, dropout)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, x, resume_time):
        """
        Args:
            x: (batch_size, window_size, d_model)
            resume_time: (batch_size, window_size, 1)

        Returns:
            output: (batch_size, window_size, d_model)
        """
        x = self.position_encoding(x, resume_time)
        x = self.transformer(x)
        return x
