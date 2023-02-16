import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class AC_On_RG_Agent:
    def __init__(self, n_actions, obs_size, discrete,
        discount=0.99, gae_lambda=0.95,
        log_std_init=0.0, log_std_annealing_rate=-0.001, log_std_lr=0.0001,
        batch_size=64, train_steps=2048, n_epochs=10,
        # total_steps=None,
        learning_rate=0.0003, max_grad_norm=0.5,
        # vf_loss_beta=0.5, entropy_loss_beta=0.0,
        tensorboard_log_dir=None,
        device=torch.device('cpu'),
        actor=None, critic=None):

        self.n_actions = n_actions                              # num of actions (scalar)
        self.obs_size = obs_size                                # dimension of observation (scalar) used for default actor/critic architectures
        self.discrete = discrete                                # whether actions are discrete (categorical) or continuous (Gaussian)
        self.discount = discount                                # discount factor i.e. gamma
        self.gae_lambda = gae_lambda                            # gae lambda factor
        self.log_std_annealing_rate = log_std_annealing_rate    # added to self.log_std after every train cycle i.e. every n_epochs
        self.log_std_lr = log_std_lr                            # different optimizer/lr for log_std
        self.batch_size = batch_size                            # num of steps to include in each minibatch
        self.train_steps = train_steps                          # num of steps per training cycle
        self.n_epochs = n_epochs                                # num of epochs to train per train_steps
        # self.total_steps = total_steps                          # total steps to train for scheduling purpose / no scheduling if None
        self.lr = learning_rate                                 # learning rate for total loss from actor, critic, entropy
        self.max_grad_norm = max_grad_norm                      # max norm for gradient clipping
        # self.vf_loss_beta = vf_loss_beta                        # coefficient for critic loss
        # self.entropy_loss_beta = entropy_loss_beta              # coefficient for entropy loss
        self.device = device
        self.rng = np.random.default_rng()

        if tensorboard_log_dir is None:
            self.tensorboard = False
        else:
            dir = tensorboard_log_dir
            self.writer = SummaryWriter() if dir == 'default' else SummaryWriter(dir)
            self.tensorboard = True

        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(self.n_actions, dtype=torch.float32) + log_std_init)
            self.log_std_optim = torch.optim.Adam([self.log_std], lr=self.log_std_lr)

        if actor is None:
            self.actor = nn.Sequential(
                nn.Linear(self.obs_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.n_actions),
            )
        else: self.actor = actor
        self.actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        if critic is None:
            self.critic = nn.Sequential(
                nn.Linear(self.obs_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        else: self.critic = critic
        self.critic.to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.training_mode = True   # for train mode actions
        self.steps = 0              # to keep track for training
        self.episode = 1            # for logging of episode lengths and rewards
        self.buffer = []
        self.n_batchs = math.ceil(self.train_steps / self.batch_size)

    def agent_start(self, observation):
        # for tensorboard logging
        self.cum_reward = 0

        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0) # only obs, out of all transition elements in buffer, has additional batch dim
        action = self.get_action(obs)

        # for transferring to buffer
        self.update_internal_states(obs, action)

        return action.numpy().squeeze()

    def agent_step(self, reward, observation):
        # for tensorboard logging
        self.cum_reward += np.squeeze(reward)

        # gae computation requires last state of agent (not in the buffer)
        self.next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        reward = torch.tensor([reward], dtype=torch.float32) if np.shape(reward)==() else torch.tensor(reward, dtype=torch.float32) # some env gives reward as a (1,) array
        # must be before new action/obs replaces self.prev_action/self.prev_state which are not passed into function but are moved into buffer
        if self.training_mode: self.train_mode_actions(reward, True)

        action = self.get_action(self.next_state)

        # for transferring to buffer
        self.update_internal_states(self.next_state, action)

        return action.numpy().squeeze()

    def agent_end(self, reward, observation):
        # for tensorboard logging
        self.cum_reward += np.squeeze(reward)

        # gae computation requires last state of agent and whether it is terminal (not in the buffer)
        self.next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        reward = torch.tensor([reward], dtype=torch.float32) if np.shape(reward)==() else torch.tensor(reward, dtype=torch.float32) # some env gives reward as a (1,) array
        if self.training_mode: self.train_mode_actions(reward, False)
        if self.tensorboard: self.episode_log()

    def update_internal_states(self, observation, action):
        self.prev_state = observation
        self.prev_action = action
        with torch.no_grad(): self.critic_value = self.critic(observation.to(self.device)).cpu().squeeze(0) # remove batch dim for single instance

    def get_action(self, observation, return_dist=False):
        observation = observation.to(self.device)
        if self.discrete:
            logits = self.actor(observation)
            action_dist = torch.distributions.Categorical(logits=logits)
        else:
            batch_mean = self.actor(observation)
            batch_cov = torch.diag((2*self.log_std).exp()).repeat(batch_mean.shape[0], 1, 1).to(self.device)
            # print(batch_mean.shape, batch_cov.shape) #(1,3) and (1,3,3)
            action_dist = torch.distributions.MultivariateNormal(batch_mean, covariance_matrix=batch_cov)

        if return_dist:
            return action_dist
        else:
            action = action_dist.sample().detach()
            return action.cpu()

    def train_mode_actions(self, reward, next_state_not_terminal):
        # completion of a full transition i.e. after first agent.step => self.steps = 1
        self.steps += 1

        next_state_not_terminal = torch.tensor([next_state_not_terminal], dtype=torch.bool)
        transition = (self.prev_state, self.critic_value, self.prev_action, reward, next_state_not_terminal)
        self.buffer.append(transition)

        # training after last transition is added into buffer and steps incremented
        if (self.steps % self.train_steps == 0): self.train()

    def train(self):
        # process transitions from each step = tuple to vertical stack of types of transition and calculate gae/returns
        unpacked_tuples = zip(*self.buffer)
        current_states, current_critic_values, actions, rewards, next_state_not_terminals = (torch.cat(tuple) for tuple in unpacked_tuples)
        advantages, returns = self.compute_gae_and_returns(current_critic_values, rewards, next_state_not_terminals)

        for epoch in range(self.n_epochs):
            indices = np.arange(self.train_steps)
            self.rng.shuffle(indices) # randomised indices to shuffle transitions and ensure each data point is used (if train_steps % batch_size = 0)

            for i in range(self.n_batchs):
                start = i * self.batch_size
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch = (tensors[batch_indices] for tensors in (current_states, actions, advantages, returns))
                self.train_batch(*batch)

        self.buffer = [] # clear buffer after full batch has been used for training
        if not self.discrete:
            with torch.no_grad(): self.log_std += self.log_std_annealing_rate # apply log_std annealing

    def train_batch(self, current_states, actions, advantages, returns):
        current_states, actions, advantages, returns = \
            self.to_device([current_states, actions, advantages, returns])

        # standardise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_dist = self.get_action(current_states, return_dist=True)
        actor_loss  = -torch.sum(action_dist.log_prob(actions) * (advantages + 1.))

        if not self.discrete: self.log_std_optim.zero_grad()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()), self.max_grad_norm)
        self.actor_optimizer.step()
        if not self.discrete: self.log_std_optim.step()

        critic_values = self.critic(current_states).squeeze()
        assert critic_values.shape == returns.shape, 'critic_values shape: ' + str(critic_values.shape) + '\nreturns shape: ' + str(returns.shape)
        critic_loss = F.mse_loss(returns, critic_values)

        # entropy = action_dist.entropy().mean() if self.entropy_loss_beta > 0 else 0.
        # assert self.vf_loss_beta > 0, 'Value function loss beta is not positive'

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic.parameters()), self.max_grad_norm)
        self.critic_optimizer.step()

    def compute_gae_and_returns(self, critic_values, rewards, next_state_not_terminals):
        with torch.no_grad():
            gae = 0
            advantages = torch.zeros_like(rewards)
            for t in reversed(range(self.train_steps)):
                # for last state in buffer, must calculate next state critic value based on stored final obs
                if t == self.train_steps - 1: next_state_value = self.critic(self.next_state.to(self.device)).cpu().squeeze(0) # remove batch dim for single instance
                else: next_state_value = critic_values[t+1]

                # gae starts with last transition in buffer so if it is the last transition, it will just be final reward - value(final state)
                # not_terminal will break the gae continuing product when transitioning to different episode
                delta = rewards[t] + self.discount * next_state_value * next_state_not_terminals[t] - critic_values[t]
                gae = delta + self.discount * self.gae_lambda * next_state_not_terminals[t] * gae
                advantages[t] = gae
            # assert advantages.shape == critic_values.shape, 'advantage shape: ' + str(advantages.shape) + '\ncritic_values shape: ' + str(critic_values.shape)
            returns = advantages + critic_values
            return advantages, returns

    def to_device(self, list):
        device_var = []
        for var in list:
            var = var.to(self.device)
            device_var.append(var)
        return device_var