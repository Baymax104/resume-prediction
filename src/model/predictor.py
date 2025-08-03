# -*- coding: UTF-8 -*-
from pathlib import Path

from torch import nn
from transformers import AutoModel


class TextEmbedding(nn.Module):

    def __init__(
        self,
        output_dim: int,
        pretrained_model_dir: Path,
    ):
        super(TextEmbedding, self).__init__()
        bert_path = pretrained_model_dir / "bge-large-zh-v1.5"
        self.bert = AutoModel.from_pretrained(bert_path.resolve())
        self.projection = nn.Linear(1024, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = outputs.pooler_output
        x = self.projection(x)
        return x


class ResumePredictor(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        pretrained_model_dir: Path,
        dropout: float = 0.1
    ):
        super(ResumePredictor, self).__init__()
        self.embedding = TextEmbedding(hidden_dim, pretrained_model_dir)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
            # nn.Dropout(p=dropout),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.embedding(input_ids, attention_mask, token_type_ids)
        x = self.mlp(x)
        return x
