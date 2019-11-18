import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from model.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):

    def __init__(self, ninp, ntoken, ntoken_dec, nhid=2048, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder_emb = nn.Embedding(ntoken_dec, ninp)
        self.decoder_out = nn.Linear(ninp, ntoken_dec)
        self.model = Transformer(d_model=ninp, dim_feedforward=nhid)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        tgt = self.decoder_emb(tgt)
        src_mask = src_mask != 1
        tgt_mask = tgt_mask != 1
        output = self.model(src.transpose(0, 1), tgt.transpose(0, 1), src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        output = self.decoder_out(output)
        return output
