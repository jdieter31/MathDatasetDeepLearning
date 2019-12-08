import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from model.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):

    def __init__(self, ninp, ntoken, ntoken_dec, nhid=2048, dropout=0):
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
        tgt = self.decoder_emb(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        src_mask = src_mask != 1
        tgt_mask = tgt_mask != 1
        subseq_mask = self.model.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.model(src.transpose(0, 1), tgt.transpose(0, 1), tgt_mask=subseq_mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        output = self.decoder_out(output)
        return output

    def greedy_decode(self, src, src_mask, sos_token, max_length=20):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        src_mask = src_mask != 1
        encoded = self.model.encoder(src.transpose(0, 1), src_key_padding_mask=src_mask)
        generated = encoded.new_full((encoded.size(1), 1), sos_token, dtype=torch.long)
        for i in range(max_length - 1):
            subseq_mask = self.model.generate_square_subsequent_mask(generated.size(1)).to(src.device)
            decoder_in = self.decoder_emb(generated) * math.sqrt(self.ninp)
            decoder_in = self.pos_encoder(decoder_in)
            logits = self.decoder_out(self.model.decoder(decoder_in.transpose(0, 1), encoded, tgt_mask=subseq_mask, memory_key_padding_mask=src_mask)[-1,:,:])
            new_generated = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, new_generated], dim=-1)
        return generated
    
    def save(self, file_dir):
        torch.save(self.state_dict(), file_dir)



            



