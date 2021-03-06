import gym
from model import RLTRdemo
from rltr import RLTR
import csv
import torch
from torch.distributions.categorical import Categorical
import numpy as np

BACKUP_PATH = "models/state_dict_rltr_30.pt"
INIT_MODEL_PATH = "models/state_dict_rltr_init.pt"
MODEL_PATH = "models/state_dict_rltr.pt"
RL_PATH = "models/state_dict_rltr_RL.pt"

PATH = MODEL_PATH

def get_action(obs, model):
    actions = model(obs)
    return actions


# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_action(obs).log_prob(act)
    return -(logp * weights).mean()


if __name__ == "__main__":
    # create env
    env = gym.make("gym_rltracking:rltracking-v0")
    env.init_source("ADL-Rundle-6")
    # env.init_view("001")
    # observation = env.reset()
    action = env.action_space.sample()

    # load model
    model = RLTR()
    # model.load_state_dict(torch.load(INIT_MODEL_PATH))
    model.load_state_dict(torch.load(PATH))
    # model.load_state_dict(torch.load(BACKUP_PATH))


    # initiate objects according to first frame's detection
    obs = env.initiate_obj(10)
    # env.render_img()

    outputs = model([obs])
    # print(outputs['pred_boxes'].detach().numpy()[0])

    # obs, reward, done, _ = env.step(outputs['pred _boxes'].numpy()[0])
    # obs, reward, done, _ = env.step([np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])])

    for i in range(500):
        outputs = model([obs])
        op_action = Categorical(logits=outputs['operations']).sample()
        bbox_action = outputs['pred_boxes']
        action = {'pred_boxes': bbox_action[0],
            'operations': op_action[0]}
        obs, reward, done, _ = env.step(action)
        print(op_action)
        env.render_img(action)

    env.reset()

    # load algorithm
