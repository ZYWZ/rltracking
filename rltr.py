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

        self.detection_embedder = Embedder(1200, 64)
        self.entity_embedder = Embedder(1200, 64)
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

    def forward(self, obss):
        """
        expect list of dicts. Each dict contains a observation from the envrionment.

        :param - obss: batched observations of list [batch_size x dict]
        :return:
             - pred_boxes: list of tensors. batched predicted normalized boxes coordinates [batch_size x num_queries x 4]
             - operations: list of tensors. batched predicted operation logits [batch_size x num_queries x 4]
        """
        batch_size = len(obss)
        detections = []
        entities = []
        masks = []

        for i, obs in enumerate(obss):
            detection = obs['next_frame']
            entity = obs['locations']
            mask = obs['mask']
            detections.append(detection)
            entities.append(entity)
            masks.append(mask)

        detections = torch.Tensor(detections).long()
        entities = torch.Tensor(entities).long()
        masks = torch.Tensor(masks)
        cur_emb = self.detection_embedder(detections).flatten(start_dim=2)
        entitity_emb = self.entity_embedder(entities).flatten(start_dim=2)

        src1 = self.pos_encoder(cur_emb).permute(1, 0, 2)
        src2 = self.pos_encoder(entitity_emb).permute(1, 0, 2)

        det_encod = self.detection_encoder(src1, src_key_padding_mask=masks)
        ent_encod = self.entity_encoder(src2)

        memory = torch.cat((det_encod, ent_encod), dim=0)

        tgt = self.query_pos.unsqueeze(1).repeat(1, batch_size, 1)
        output = self.transformer_decoder(tgt, memory).permute(1, 0, 2)

        pred_boxes = self.linear_offset(output).sigmoid()
        operations = self.operation(output)

        # return transformer output
        return {'pred_boxes': pred_boxes,
            'operations': operations}


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, batch_size=8, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(batch_size, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        # pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)
