import torch
import torch.nn as nn

# Transformer架构总共分为三部分：编码层，解码层和输出层
class Transformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()  # 编码层
        self.decoder = Decoder()  # 解码层
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)  # 输出层：一个mlp网络，将模型的隐向量维度映射成词表维度，来生成token

    def forward(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        token_logit = self.projection(decoder_output)
        return token_logit