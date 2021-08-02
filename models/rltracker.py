import math
import torch
from torch import nn
from .extractor import Extractor, build_extractor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.distributions.categorical import Categorical

from .layers import StableTransformerXL, PositionalEmbedding

torch.set_grad_enabled(True)


"""
    RLTracker, output four actions : update, add, remove, block; representing action to input queries
"""
class RLTracker(nn.Module):
    def __init__(self, hidden_dim=1024, n_heads=4, n_layers=6,
                 d_head_inner=64, d_ff_inner=128, action_space=3):
        super().__init__()
        # self.position_embedding = PositionalEncoding()
        self.memory = None
        self.pos_embed = nn.Embedding(1200, 128)
        # self.reduce_dim = self.mlp([512, 256, 128])

        self.transformer = StableTransformerXL(d_input=hidden_dim, n_layers=n_layers,
            n_heads=n_heads, d_head_inner=d_head_inner, d_ff_inner=d_ff_inner)

        # self.linear_action = nn.Linear(hidden_dim, 3)

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_space)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def _distribution(self, trans_state):
        logits = self.actor(trans_state).softmax(-1).permute(1, 0, 2)
        return Categorical(logits)

    def _log_prob_from_distribution(self, policy, action):
        return policy.log_prob(action)

    def mlp(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        return nn.Sequential(*layers)

    def forward(self, obs, action=None):
        det = obs['det']
        det_feat = obs['det_feat']

        det = self.pos_embed(det.long()).flatten(1)
        # det_feat = self.reduce_dim(det_feat)
        input = torch.cat((det, det_feat), dim=1).unsqueeze(0).permute(1, 0, 2)
        # input = det.unsqueeze(0).permute(1, 0, 2)
        # encoder_pos = self.position_embedding(input).permute(1, 0, 2)

        # obj = self.pos_embed(obj.long()).flatten(1)
        # obj_feat = self.reduce_dim(obj_feat)
        # input = torch.cat((obj, obj_feat), dim=1).unsqueeze(0)
        # input = torch.cat((input, self.empty_template), dim=1)
        # decoder_pos = self.position_embedding(input).permute(1, 0, 2)
        # self.memory = None
        trans_state = self.transformer(input, self.memory)
        trans_state, self.memory = trans_state['logits'], trans_state['memory']
        policy = self._distribution(trans_state)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)
        action = policy.sample()

        return action, logp_a

# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model=256, max_batch_size=32, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_batch_size, max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, :, 0::2] = torch.sin(position * div_term)
#         pe[:, :, 1::2] = torch.cos(position * div_term)
#         # pe = pe.transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :x.size(1), :]
#         return self.dropout(x)

def build_agent(args):
    # backbone = build_backbone(args)
    extractor = build_extractor()
    model = RLTracker()
    return extractor.cuda(), model.cuda()