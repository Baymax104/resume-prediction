# -*- coding: UTF-8 -*-
import math

import torch
from torch import nn
from transformers import BertModel


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
        self.bert = BertModel.from_pretrained("../model/bert-base-chinese")
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


class ResumePredictor(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        window_size: int = 10,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super(ResumePredictor, self).__init__()
        self.time_encoder = TimeEncoder(input_dim=1, hidden_dim=64, output_dim=d_model)
        self.resume_encoder = ResumeEncoder(output_dim=d_model)
        self.position_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=window_size)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.decoder = nn.Linear(d_model, embedding_dim)

    def forward(self, sample):
        """
        Shape:
            input_ids: (batch_size, window_size, seq_len)
            attention_mask: (batch_size, window_size, seq_len)
            time_features: (batch_size, window_size, 1)
        """
        input_ids = sample["window_resume_input_ids"]
        attention_mask = sample["window_resume_attention_mask"]
        time_features = sample["window_time_features"]

        # (batch_size, window_size, d_model)
        time_features = self.time_encoder(time_features)
        # (batch_size, window_size, d_model)
        resume_features = self.resume_encoder(input_ids, attention_mask)

        fuse_features = time_features + resume_features
        fuse_features = self.position_encoding(fuse_features)
        fuse_features = self.transformer(fuse_features)

        # mean pool for window_size dimension
        fuse_features = fuse_features.transpose(1, 2)  # (batch_size, d_model, window_size)
        fuse_features = self.pool(fuse_features).squeeze(-1)  # (batch_size, d_model)
        fuse_features = self.decoder(fuse_features)  # (batch_size, output_dim)
        return fuse_features
