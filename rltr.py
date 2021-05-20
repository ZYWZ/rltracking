import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.distributions.categorical import Categorical
from torchvision.models import resnet50
import torchvision.transforms as T
import time

torch.set_grad_enabled(True)


class RLTR(nn.Module):
    def __init__(self, obs_space=None, action_space=None, hidden_dim=256, nheads=8, num_encoder_layers=3, num_decoder_layers=3):
        super().__init__()

        self.detection_embedder = Embedder(1000, 64)
        self.entity_embedder = Embedder(1000, 64)
        self.operation_embedder = Embedder(4, 64)

        # the positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layers = TransformerEncoderLayer(hidden_dim, nheads)
        self.detection_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.entity_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(hidden_dim, nheads)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # the sequence to the decoder, target sequence
        self.query_pos = nn.Parameter(torch.rand(16, hidden_dim))

        # prediction heads, output the predicted offsets for each object (x, y, w, h)
        # self.linear_offset = nn.Linear(hidden_dim, 4)
        self.linear_offset = self.mlp([hidden_dim, 32, 4])

        # prediction head, output the operation for each object
        # [0,1,2,3] stands for [add, keep, remove, ignore]
        self.operation = self.mlp([hidden_dim, 64, 4])

    def mlp(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        return nn.Sequential(*layers)

    def forward(self, obs):
        detections = obs['next_frame']
        entities = obs['locations']
        detections = detections[:, :4].long()
        entities = entities.long()

        cur_emb = self.detection_embedder(detections).flatten(start_dim=1).unsqueeze(1)
        entitity_emb = self.entity_embedder(entities).flatten(start_dim=1).unsqueeze(1)

        src1 = self.pos_encoder(cur_emb)
        if entitity_emb.shape[2] == 64:
            print(entities)
        src2 = self.pos_encoder(entitity_emb)

        det_encod = self.detection_encoder(src1)
        ent_encod = self.entity_encoder(src2)

        tgt = self.query_pos.unsqueeze(1)
        memory = torch.cat((det_encod, ent_encod), dim=0)

        output = self.transformer_decoder(tgt, memory).transpose(0, 1)

        # return transformer output
        return {'pred_boxes': self.linear_offset(output).sigmoid(),
            'operations': self.operation(output)}


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)
