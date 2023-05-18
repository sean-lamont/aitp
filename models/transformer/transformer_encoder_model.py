import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import einops
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# todo wrapper for transformer embedding taking in batch
# class TransformerFromBatch(nn.Module):
#         def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
#                      nlayers: int, dropout: float = 0.5, enc=True, in_embed=True, global_pool=True):
#             super().__init__(())
#             self.transformer_embedding = TransformerEmbedding(ntoken, d_model, nhead, d_hid,
#                                                               nlayers, dropout, enc, in_embed, global_pool)


'''
Transformer Embedding 
'''
class TransformerEmbedding(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, enc=True, in_embed=True, global_pool=True):
        super().__init__()
        self.in_embed = in_embed
        self.enc = enc
        self.global_pool = global_pool
        self.model_type = 'Transformer'

        if self.enc:
            self.pos_encoder = PositionalEncoding(d_model, dropout=0)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)#, activation='gelu')

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        if self.in_embed:
            self.encoder = nn.Embedding(ntoken, d_model)
            self.init_weights()

        self.d_model = d_model

        # self.cls_token = nn.Parameter(torch.randn(1, d_model))

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, data):
        """
        Args:
            data: x with token indices, ptr identifying which sequence tokens belong to
        Returns:
            output Tensor of shape [seq_len, batch_size, d_model], if global pool [batch_size, d_model]
        """

        # todo change this, maybe make a separate model wrapping this with processing by batch and splitting/padding
        # src = data.x

        src = data
        # split by batch ptr and pad
        # src = torch.tensor_split(src, data.ptr[1:-1])
        # src = torch.nn.utils.rnn.pad_sequence(src)

        if self.in_embed:
            src = self.encoder(src) * math.sqrt(self.d_model)

        # if self.global_pool:
        #     cls_tokens = einops.repeat(self.cls_token, '() d -> 1 b d', b=len(data.ptr) - 1)
        #     src = torch.cat([src, cls_tokens], dim=0)

        if self.enc:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        output = torch.transpose(output, 0, 1)

        if self.global_pool:
            #CLS token value
            output = output[:, 0]
            return output

        return output
