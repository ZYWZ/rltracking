from utils.base import BaseAlgo
from torch.optim import Adam
import numpy as np


class VpgAlgo(BaseAlgo):

    def __init__(self, env, model, device, num_frames, discount=1.0, lr=1e-4):
        super().__init__(env, model, device, num_frames, discount, lr)

        self.optimizer = Adam(model.parameters(), lr=lr)

    def update_parameters(self, exps):
        update_loss = 0
        reward = 0
        inds = self._get_starting_indexes()

        for i in range(self.num_frames):
            sb = exps[i]
            op_policy = self.get_operation_policy(sb.obs)
            loss = -(op_policy.log_prob(sb.op_action).sum(axis=-1) * sb.reward).mean()

            update_loss += loss
            reward += sb.reward

        update_loss /= self.num_frames
        reward /= self.num_frames
        self.optimizer.zero_grad()
        update_loss.backward()
        self.optimizer.step()

        logs = {
            "loss": update_loss,
            "reward": reward
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = np.arange(0, self.num_frames)
        return starting_indexes
