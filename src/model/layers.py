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
        Shape:
            input_ids: (batch_size, window_size, seq_len)
            attention_mask: (batch_size, window_size, seq_len)
        """
        batch_size, window_size, seq_len = input_ids.size()
        input_ids = input_ids.view(batch_size * window_size, seq_len)
        attention_mask = attention_mask.view(batch_size * window_size, seq_len)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.projection(outputs.pooler_output)
        outputs = self.dropout(outputs)
        return outputs.view(batch_size, window_size, -1)


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        window_size,
        dropout=0.1,
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

    def forward(self, x, time_features):
        r"""
        Shape:
            x: (batch_size, window_size, emb_dim)
            time_features: (batch_size, window_size, 1)
            output: (batch_size, window_size, emb_dim)
        """
        time_weights = torch.clamp(time_features, min=0, max=120) / 120
        time_weights = time_weights / time_weights.sum()
        weights = self.position_weights * time_weights
        weighted_pe = self.pe * weights
        x = x + weighted_pe[:, :x.size(1)]
        return self.dropout(x)
