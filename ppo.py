import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device('cuda:0')

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.memories = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.memories[:]

class Agent(nn.Module):
    def __init__(self, hidden_dim=256, n_heads=4, n_layers=6,
                 d_head_inner=64, d_ff_inner=128, action_space=3):
        super().__init__()
        # self.position_embedding = PositionalEncoding()
        self.memory = None
        self.pos_embed = nn.Embedding(1200, 32)
        self.reduce_dim = self.mlp([512, 256, 128])

        #         self.transformer = StableTransformerXL(d_input=hidden_dim, n_layers=n_layers,
        #                                                n_heads=n_heads, d_head_inner=d_head_inner, d_ff_inner=d_ff_inner)

        # self.linear_action = nn.Linear(hidden_dim, 3)
        self.memory_rnn = nn.LSTMCell(hidden_dim, hidden_dim)
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

    def forward(self, obs, hidden_state=None, action=None):
        det = obs['det']
        det_feat = obs['det_feat']

        det = self.pos_embed(det.long()).flatten(1)

        det_feat = self.reduce_dim(det_feat)
        tr_input = torch.cat((det, det_feat), dim=1)

        if hidden_state is None:
            hidden = self.memory_rnn(tr_input)
            tr_input = hidden[0]
        else:
            hidden = (hidden_state[:, :256], hidden_state[:, 256:])
            hidden = self.memory_rnn(tr_input, hidden)
            tr_input = hidden[0]
        hidden_state = torch.cat(hidden, dim=1)
        # # tr_input = torch.stack([tr_input]).permute(1, 0, 2)
        tr_input = tr_input.unsqueeze(0).permute(1, 0, 2)

        #         trans_state = self.transformer(tr_input, self.memory)
        #         trans_state, self.memory = trans_state['logits'], trans_state['memory']

        policy = self._distribution(tr_input)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)
        action = policy.sample()

        return action, logp_a, hidden_state


class PPO:
    def __init__(self, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = Agent()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = Agent()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):

        with torch.no_grad():
            action, logp_a, hidden_state = self.policy_old.forward(state, memory)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logp_a)
        self.buffer.memories.append(hidden_state)

        return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))