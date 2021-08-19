import torch
import argparse
import numpy as np
import random
from PIL import Image
from torch.optim import Adam
import matplotlib.pyplot as plt
from models.rltracker import build_agent
import gym
import torchvision.transforms as T
import os
from utils.vpg import VpgAlgo
from torch.distributions.categorical import Categorical
from ppo import *
from datetime import datetime

INIT_MODEL_PATH = "models/state_dict_rltr_RL_init.pt"
MODEL_PATH = "models/state_dict_rltr_RL.pt"

def get_args_parser():
    parser = argparse.ArgumentParser('RLTracker args', add_help=False)
    return parser

def train_ppo():
    max_training_timesteps = int(3e4)
    lr_actor = 0.00003  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network
    gamma = 0.99
    K_epochs = 8
    eps_clip = 0.2
    max_ep_len = 100
    update_timestep = max_ep_len * 4

    source = 'MOT17-02-FRCNN'
    env_name = 'gym_rltracking:rltracking-v1'
    env = gym.make(env_name)
    env.init_source(source, "train")

    extractor, _ = build_agent(args)
    ppo_agent = PPO(lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    extractor.eval()

    env.set_extractor(extractor)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    time_step = 0
    start_frame = 1
    # training loop
    while time_step <= max_training_timesteps:
        state = env.initiate_env(start_frame)
        current_ep_reward = 0
        memory = None
        for t in range(1, max_ep_len + 1):
            # select action with policy
            action, logp_a, memory = ppo_agent.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RLTracker args', parents=[get_args_parser()])
    args = parser.parse_args()

    train_ppo()