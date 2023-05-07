import torch
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

class TransformerEmbedding(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, enc=True, in_embed=True, global_pool=True):

        super().__init__()

        print (f"dropout: {dropout}")

        self.in_embed = in_embed
        self.enc = enc
        self.global_pool = global_pool

        self.model_type = 'Transformer'

        if self.enc:
            self.pos_encoder = PositionalEncoding(d_model, dropout=0)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        if self.in_embed:
            self.encoder = nn.Embedding(ntoken, d_model)
            self.init_weights()

        self.d_model = d_model


    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, d_model]
        """
        seq_len = src.size(0)
        # print (src.shape)
        if self.in_embed:
            src = self.encoder(src) * math.sqrt(self.d_model)

        if self.enc:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src)

        output = torch.transpose(output, 0, 1)
        # print (output.shape)

        if self.global_pool:
            #CLS token value
            output = output[:, 0]
            return output

        # print (output.shape)
        # gmp = nn.MaxPool1d(seq_len, stride=1)
        # return gmp(output).squeeze(-1)  # orch.cat([gmp(out).squeeze(-1), torch.sum(out,dim=2)], dim = 1)

        return output
