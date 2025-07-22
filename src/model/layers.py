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
        self.projection = nn.Linear(768, output_dim)

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
        return outputs.view(batch_size, window_size, -1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
