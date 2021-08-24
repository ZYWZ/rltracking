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

def read_action_list():
    file = 'action_list.txt'
    result = []
    with open(file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    for line in content:
        line = line.split(',')
        line = [int(x) for x in line]
        result.append(line)
    return result

actions = read_action_list()
source = 'MOT17-04-FRCNN'
env = gym.make('gym_rltracking:rltracking-v1')
env.init_source(source, "train")
args = ""
extractor, agent = build_agent(args)
env.set_extractor(extractor)
obs = env.initiate_env(1)

rewards = []
for action in actions:
    action = torch.Tensor(action).unsqueeze(0)
    obs, reward, end, _ = env.step(action)
    print(reward)
    rewards.append(reward)

env.output_result()
env.output_gt()
