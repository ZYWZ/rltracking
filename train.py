import torch
import argparse
import numpy as np
from PIL import Image
from torch.optim import Adam

from models.rltracker import build_agent
import gym
import torchvision.transforms as T

from utils.vpg import VpgAlgo
from torch.distributions.categorical import Categorical

# INIT_MODEL_PATH = "models/state_dict_rltr_init.pt"
MODEL_PATH = "models/state_dict_rltr_RL.pt"
test_img = "datasets/2DMOT2015/train/PETS09-S2L1/img1/000001.jpg"

def get_args_parser():
    parser = argparse.ArgumentParser('RLTracker args', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # PyTorch checkpointing for saving memory (torch.utils.checkpoint.checkpoint)
    parser.add_argument('--checkpoint_enc_ffn', default=False, action='store_true')
    parser.add_argument('--checkpoint_dec_ffn', default=False, action='store_true')

    return parser

def train(args, env_name='gym_rltracking:rltracking-v1', lr=1e-5,
          epochs=1, batch_size=10, render=False, total_frames=500):
    # make environment, check spaces, get obs / act dims

    env = gym.make(env_name)
    env.init_source("PETS09-S2L1")

    # assert isinstance(env.action_space, Tuple), \
    #     "This example only works for envs with Tuple action spaces."
    # assert isinstance(env.observation_space, Dict), \
    #     "This example only works for envs with Dict state spaces."

    extractor, agent = build_agent(args)
    extractor.eval()
    agent.eval()
    env.set_extractor(extractor)
    obs = env.initiate_env()
    # logits = agent(obs)
    # action = Categorical(logits).sample()
    for i in range(10):
        action = torch.Tensor([[0, 0, 0, 1]]).long()
        obs, reward = env.step(action)
        env.printTrack()

    # algo = VpgAlgo(env=env, model=agent, device=device, num_frames=10, lr=1e-4)

    num_frames = 0

    epoch = 0

    # while num_frames < total_frames:
    #     frame = random.randint(0, 10)
    #     exps = algo.collect_experiences(frame)
    #     logs = algo.update_parameters(exps)
    #     print("epoch ", epoch, logs)
    #
    #     epoch += 1
    #     num_frames += 10
    #
    # print("Saving model...")
    # torch.save(agent.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RLTracker args', parents=[get_args_parser()])
    args = parser.parse_args()

    train(args)
