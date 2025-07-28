# -*- coding: UTF-8 -*-
from pathlib import Path

from torch import nn
from transformers import AutoModel


class ResumeEmbedding(nn.Module):

    def __init__(self):
        super(ResumeEmbedding, self).__init__()
        bge_path = Path(__file__).parent.parent.parent / "model" / "bge-large-zh-v1.5"
        self.bge = AutoModel.from_pretrained(bge_path.resolve())
        for parameter in self.bge.parameters():
            parameter.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bge(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output
