# -*- coding: UTF-8 -*-
import torch
from torch import nn

from model.layers import AttentionPooling, Decoder, LSTMModel, ResumeEncoder, TimeEncoder


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
        self.projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
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
        # x = self.transformer(x)
        x = self.lstm(x)

        # (batch_size, window_size, embedding_dim)
        x = self.decoder(x)

        # pool for window_size dimension
        x = self.pool(x)  # (batch_size, embedding_dim)
        x = self.dropout(x)
        return x
