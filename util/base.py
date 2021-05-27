from abc import ABC, abstractmethod
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch_ac.utils import DictList
import torch

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, env, model, device, num_frames, discount=1.0, lr=1e-5):
        self.env = env
        self.model = model
        self.device = device
        self.num_frames = num_frames
        self.discount = discount
        self.lr = lr

        # Configure acmodel
        self.model.to(self.device)
        self.model.train()

        # Initialize experience values

        shape = self.num_frames

        self.obs = self.env.reset()
        self.obss = [None] * shape
        self.actions = torch.zeros((shape, 16, 4), device=self.device)
        self.op_actions = torch.zeros((shape, 16), device=self.device)
        self.rewards = torch.zeros((shape, 1), device=self.device)
        self.log_probs = torch.zeros((shape, 16), device=self.device)

    def get_operation_policy(self, obs):
        logits = self.model(obs)
        return Categorical(logits=logits['operations'])

    def get_operation_action(self, obs):
        return self.get_operation_policy(obs).sample()

    def get_policy(self, obs, model):
        outputs = model(obs)
        return Normal(outputs['pred_boxes'], torch.Tensor([0.005]))

    def get_action(self, obs, model):
        return self.get_policy(obs, model).sample()

    def collect_experiences(self, start_frame):
        """Collects rollouts and computes advantages.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        # initiate objects according to first frame's detection
        self.obs = self.env.initiate_obj(start_frame)
        for i in range(self.num_frames):
            policy = self.get_policy(self.obs, self.model)
            op_policy = self.get_operation_policy(self.obs)
            act = self.get_action(self.obs, self.model)
            op_act = self.get_operation_action(self.obs)
            obs, rew, done, _ = self.env.step(op_act)

            self.obss[i] = self.obs
            self.obs = obs
            # self.actions[i] = act
            self.op_actions[i] = op_act
            self.rewards[i] = torch.tensor(rew, device=self.device)
            self.log_probs[i] = op_policy.log_prob(op_act).sum(axis=-1)

        # print(self.actions.shape)

        exps = DictList()
        exps.obs = [self.obss[i]
                    for i in range(self.num_frames)]
        # exps.action = self.actions
        exps.op_action = self.op_actions
        exps.reward = self.rewards
        exps.log_prob = self.log_probs

        return exps

