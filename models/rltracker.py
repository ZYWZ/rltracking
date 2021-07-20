import math
import torch
from torch import nn
from .extractor import Extractor, build_extractor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

torch.set_grad_enabled(True)


"""
    RLTracker, output four actions : update, add, remove, block; representing action to input queries
"""
class RLTracker(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=3,
                 num_decoder_layers=3):
        super().__init__()
        self.position_embedding = PositionalEncoding()
        self.pos_embed = nn.Embedding(1200, 32)
        self.reduce_dim = nn.Linear(512, 128)

        self.empty_template = nn.Parameter(torch.rand((1, 1, hidden_dim)))

        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        self.linear_action = nn.Linear(hidden_dim, 4)


    def forward(self, obs):
        det, det_feat, obj, obj_feat = obs

        det = self.pos_embed(det.long()).flatten(1)
        det_feat = self.reduce_dim(det_feat)
        input = torch.cat((det, det_feat), dim=1).unsqueeze(0)
        encoder_pos = self.position_embedding(input).permute(1, 0, 2)

        obj = self.pos_embed(obj.long()).flatten(1)
        obj_feat = self.reduce_dim(obj_feat)
        input = torch.cat((obj, obj_feat), dim=1).unsqueeze(0)
        input = torch.cat((input, self.empty_template), dim=1)
        decoder_pos = self.position_embedding(input).permute(1, 0, 2)

        out = self.transformer(encoder_pos, decoder_pos).permute(1, 0, 2)
        action = self.linear_action(out).softmax(-1)
        return action

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=256, max_batch_size=32, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_batch_size, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        # pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :x.size(1), :]
        return self.dropout(x)

def build_agent(args):
    # backbone = build_backbone(args)
    extractor = build_extractor()
    model = RLTracker()
    return extractor.cuda(), model.cuda()