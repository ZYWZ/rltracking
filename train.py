import torch
from torch.optim import Adam

from utils.base import BaseAlgo
from model import RLTRdemo
from rltr import RLTR
import gym
from gym.spaces import Dict, Tuple
import random
from torch.distributions.normal import Normal

from utils.vpg import VpgAlgo

# INIT_MODEL_PATH = "models/state_dict_rltr_init.pt"
MODEL_PATH = "models/state_dict_rltr_RL.pt"

def train(env_name='gym_rltracking:rltracking-v0', lr=1e-5,
          epochs=1, batch_size=10, render=False, total_frames=500):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    env.init_device(True)
    # obs_dim = env.observation_space.shape[0]
    # n_acts = env.action_space.n
    assert isinstance(env.action_space, Tuple), \
        "This example only works for envs with Tuple action spaces."
    assert isinstance(env.observation_space, Dict), \
        "This example only works for envs with Dict state spaces."

    agent = RLTR()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = ("cpu")

    algo = VpgAlgo(env=env, model=agent, device=device, num_frames=10, lr=1e-4)

    num_frames = 0

    epoch = 0

    while num_frames < total_frames:
        frame = random.randint(0, 10)
        exps = algo.collect_experiences(frame)
        logs = algo.update_parameters(exps)
        print("epoch ", epoch, logs)

        epoch += 1
        num_frames += 10

    print("Saving model...")
    torch.save(agent.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    train()
