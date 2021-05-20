# import requests
# import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import math
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

torch.set_grad_enabled(True)


class RLTRdemo(nn.Module):
    def __init__(self, obs_space=None, action_space=None, hidden_dim=256, nheads=8, num_encoder_layers=3, num_decoder_layers=3):
        super().__init__()

        # create conversion layers
        # self.fc1 = nn.Linear(528, hidden_dim)
        self.conver = self.mlp([528, 400, hidden_dim])

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # torch.nn.init.xavier_uniform_(self.transformer.weight)

        # transformer output head, output embedded obj_list
        # in: output.flatten(), out: 256x16 = 4096
        self.embed_list = nn.Linear(4096, 512)

        # prediction heads, output the predicted offsets for each object (x, y, w, h)
        # self.linear_offset = nn.Linear(hidden_dim, 4)
        self.linear_offset = self.mlp([hidden_dim, 32, 4])

        # prediction head, output the current object number
        self.obj_number = nn.Linear(512, 1)

        # prediction head, output the operation for each object
        # [0,1,2,3] stands for [add, keep, remove, ignore]
        self.operation = self.mlp([hidden_dim, 64, 4])

        self.embedder = Embedder(1000, 4)

        # the sequence to the decoder, target sequence
        self.query_pos = nn.Parameter(torch.rand(16, hidden_dim))

        # the positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

    def mlp(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        return nn.Sequential(*layers)

    def forward(self, obs):
        detections = obs['next_frame']
        cur_locations = obs['locations']
        det_memory = obs['cur_frame']
        number = obs['number']
        cur_location = detections[:, :4].long()
        prev_location = det_memory[:, :4].long()

        cur_emb = self.embedder(cur_location).flatten(start_dim=1)
        prev_emb = self.embedder(prev_location).flatten(start_dim=1)

        # x = torch.cat((cur_locations, detections), 1)
        detections = torch.cat((cur_emb, detections[:, 4:]), 1)
        det_memory = torch.cat((prev_emb, det_memory[:, 4:]), 1)

        input = torch.cat((detections,det_memory), dim=0)

        # construct positional encodings
        x = self.conver(detections).unsqueeze(1)
        y = self.conver(det_memory).unsqueeze(1)

        input = self.conver(input).unsqueeze(1)

        src = self.pos_encoder(input)

        tgt = self.query_pos.unsqueeze(1)

        # propagate through the transformer
        output = self.transformer(src, tgt).transpose(0, 1)

        # embedList = self.embed_list(output.flatten())

        # return transformer output
        return {#'pred_number': self.obj_number(embedList).sigmoid(),
                # 'list_embeddings': output,
                'pred_boxes': self.linear_offset(output).sigmoid(),
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


