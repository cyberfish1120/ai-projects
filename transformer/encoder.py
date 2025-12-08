import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = Embedding()
        self.position = Position()
        self.attention = Attention()
        self.ffn = [mlp1, mlp2]

    def forward(self, x):
        emb = self.embedding(x)
        feature = self.position(emb)
        after_attention = self.attention(feature)
        after_add_mha = concate(feature+after_attention)
        output = self.ffn(Norm(after_add_mha))
        return output
