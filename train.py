import torch
import argparse
import numpy as np
import random
from PIL import Image
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
from models.rltracker import build_agent
import gym
import torchvision.transforms as T
import os
from utils.vpg import VpgAlgo
from torch.distributions.categorical import Categorical

INIT_MODEL_PATH = "models/state_dict_rltr_RL_init.pt"
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


def write_to_log(log):
    with open('gym_rltracking/logs.txt', 'w') as f:
        for item in log:
            f.write(','.join(map(repr, item)))
            f.write('\n')


def plot_validate_reward(rewards):
    epochs = range(0, int(len(rewards)))

    plt.plot(epochs, rewards, '#FFA500', label='Validate reward')
    plt.title('Validate reward of rltracker')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    plt.savefig('rewards_validate.png')


def plot_reward(rewards, lr):
    epochs = range(0, int(len(rewards)))

    plt.plot(epochs, rewards, 'b', label='Training reward')
    plt.title('Training reward of rltracker, lr=' + str(lr))
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    plt.savefig('rewards.png')

def calculate_discount_reward(index, reward_list):
    discount = 0.9
    reward = 0
    for i in range(len(reward_list[index:])):
        factor = pow(discount, i)
        reward += factor * reward_list[i]
    return reward


def train_one_epoch(env, agent, optimizer, source, start_frame, train_length):
    # torch.autograd.set_detect_anomaly(True)
    obs = env.initiate_env(start_frame)
    ep_obs = []
    ep_actions = []
    ep_rewards = []
    ep_logp_old = []
    memory = None
    ep_memory = [memory]
    MseLoss = nn.MSELoss()
    with torch.no_grad():
        for i in range(train_length):
            ep_obs.append(obs)
            action, _, _, logp_a, memory = agent(obs, memory)
            ep_memory.append(memory)
            ep_logp_old.append(logp_a)
            # action = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            ep_actions.append(action)
            obs, reward, end, _ = env.step(action)
            # ep_rewards.append(reward * (i+1))
            ep_rewards.append(reward)
            if end is True:
                env.reset()
                break
    final_rewards = []
    for i in range(len(ep_obs)):
        optimizer.zero_grad()
        _, value, dist_entropy, logp_a, memory = agent(ep_obs[i], ep_memory[i], ep_actions[i])
        weight = calculate_discount_reward(i, ep_rewards)
        final_rewards.append(weight)
        ratio = torch.exp(logp_a - ep_logp_old[i])
        weight = torch.as_tensor(weight, dtype=torch.float32)

        value_weight = torch.ones(1, 60).cuda() * weight
        value_loss = MseLoss(value[0], value_weight).sum()

        surr1 = ratio * weight
        clip_param = 0.2
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * weight

        # loss = -(logp_a * weight).sum()
        loss = -torch.min(surr1, surr2).sum() + 0.5 * value_loss - 0.01 * dist_entropy.sum()
        loss.backward(retain_graph=True)
        optimizer.step()

    print(ep_actions)
    print("discouted reward: ", np.mean(final_rewards))
    return np.mean(ep_rewards)


def validate_model(env, agent, start_frame, train_length):
    obs = env.initiate_env(start_frame)
    ep_rewards = []
    memory = None
    with torch.no_grad():
        for i in range(train_length):
            action, logp_a, memory = agent(obs, memory)
            obs, reward, end, _ = env.step(action)
            ep_rewards.append(reward)
            if end is True:
                env.reset()
                break
    return np.mean(ep_rewards)


def train(args, env_name='gym_rltracking:rltracking-v1', lr=1e-5,
          epochs=1, batch_size=10, render=False, total_frames=500):
    # make environment, check spaces, get obs / act dims
    # source = 'ADL-Rundle-6'
    source = 'MOT17-04-SDP'
    source_validate = 'MOT17-04-FRCNN'
    env = gym.make(env_name)
    env_validate = gym.make(env_name)
    # env.init_source("PETS09-S2L1")
    env.init_source(source, "train")
    env_validate.init_source(source_validate, "train")

    # assert isinstance(env.action_space, Tuple), \
    #     "This example only works for envs with Tuple action spaces."
    # assert isinstance(env.observation_space, Dict), \
    #     "This example only works for envs with Dict state spaces."
    lr = 0.0001
    extractor, agent = build_agent(args)
    agent.load_state_dict(torch.load(MODEL_PATH))
    extractor.eval()
    agent.train()
    optimizer = Adam(agent.parameters(), lr=lr)
    env.set_extractor(extractor)
    env_validate.set_extractor(extractor)
    logs = []
    rewards = []
    rewards_validate = []
    last_reward_validate = 0

    mota_log_file = 'motaLog.txt'
    if os.path.isfile(mota_log_file):
        os.remove(mota_log_file)

    for i in range(1000):
        # check for stop criterion
        # average_latest_reward = -99
        # if i > 100:
        #     average_latest_reward = np.mean(rewards[-100:])
        # if average_latest_reward > 1.3:
        #     break
        print("ep ", i)
        # start_frame = random.randint(1, 740)
        start_frame = 1
        train_length = 100
        reward = train_one_epoch(env, agent, optimizer, source, start_frame, train_length)
        # if i % 100 == 0:
        #     reward_validate = validate_model(env_validate, agent, start_frame, train_length)
        #     rewards_validate.append(reward_validate)
        #     last_reward_validate = reward_validate
        # logs.append(reward)
        rewards.append(np.mean(reward))
        # rewards_validate.append(last_reward_validate)


    # write_to_log(logs)
    print("Saving model...")
    torch.save(agent.state_dict(), MODEL_PATH)
    plot_reward(rewards, lr)
    plot_validate_reward(rewards_validate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RLTracker args', parents=[get_args_parser()])
    args = parser.parse_args()

    train(args)
