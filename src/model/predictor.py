# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

from model.layers import PositionalEncoding, ResumeEncoder, TimeEncoder


class ResumePredictor(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        window_size: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super(ResumePredictor, self).__init__()
        self.time_encoder = TimeEncoder(input_dim=1, hidden_dim=256, output_dim=d_model)
        self.resume_encoder = ResumeEncoder(output_dim=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(d_model * 2, d_model)
        self.position_encoding = PositionalEncoding(d_model, window_size=window_size, dropout=dropout)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.decoder = nn.Linear(d_model, embedding_dim)

    def forward(self, input_ids, attention_mask, resume_time):
        """
        Shape:
            input_ids: (batch_size, window_size, seq_len)
            attention_mask: (batch_size, window_size, seq_len)
            time_features: (batch_size, window_size, 1)
            output: (batch_size, output_dim)
        """

        # (batch_size, window_size, d_model)
        time_features = self.time_encoder(resume_time)
        # (batch_size, window_size, d_model)
        resume_features = self.resume_encoder(input_ids, attention_mask)

        # (batch_size, window_size, d_model * 2)
        fuse_features = torch.cat([time_features, resume_features], dim=-1)

        # (batch_size, window_size, d_model)
        fuse_features = self.projection(fuse_features)
        fuse_features = F.relu(fuse_features)
        fuse_features = self.dropout(fuse_features)
        fuse_features = self.position_encoding(fuse_features, time_features)
        fuse_features = self.transformer(fuse_features)

        # mean pool for window_size dimension
        fuse_features = fuse_features.transpose(1, 2)  # (batch_size, d_model, window_size)
        fuse_features = self.pool(fuse_features).squeeze(-1)  # (batch_size, d_model)
        fuse_features = self.decoder(fuse_features)  # (batch_size, output_dim)
        return fuse_features
