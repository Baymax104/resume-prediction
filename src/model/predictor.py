# -*- coding: UTF-8 -*-
from torch import nn
from transformers import AutoModel

from setting import SettingManager


class TextEmbedding(nn.Module):

    def __init__(self, output_dim: int, dropout: float = 0.1):
        super(TextEmbedding, self).__init__()
        settings = SettingManager.get_settings()
        bert_path = settings.model.pretrained_model_dir / "bert-base-chinese"
        self.bert = AutoModel.from_pretrained(bert_path.resolve())
        self.projection = nn.Linear(768, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = outputs.pooler_output
        x = self.projection(x)
        x = self.dropout(x)
        return x


class ResumePredictor(nn.Module):

    def __init__(self, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super(ResumePredictor, self).__init__()
        self.embedding = TextEmbedding(hidden_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.embedding(input_ids, attention_mask, token_type_ids)
        x = self.mlp(x)
        return x
