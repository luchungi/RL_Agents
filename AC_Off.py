import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple
import random
import numpy as np
import copy
import time

class AC_Off_Agent:


    def __init__(self, n_actions, obs_size, actor_lr=0.001, critic_lr=0.001, discount=0.99, buffer_max_length=int(1e6),
        batch_size=128, train_steps=8, n_batches=1, update_target_steps=8,
        baseline=True, standardise=True, log_std_init=-0.5, log_std_lr=0.001,
        soft_param=0.005, clip_gradients=False, clip_value=None,
        noise_corr=0, actor=None, critic=None, baseline_model=None, device='cpu', discrete=False
    ):

        self.n_actions = n_actions
        self.obs_size = obs_size
        self.discount = discount
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.train_steps = train_steps
        self.update_target_steps = update_target_steps
        self.soft_param = soft_param
        self.clip_gradients = clip_gradients
        self.clip_value = clip_value
        self.discrete = discrete
        self.device = device

        self.log_std_lr = log_std_lr                            # different optimizer/lr for log_std
        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(self.n_actions, dtype=torch.float32) + log_std_init)
            self.log_std_optim = torch.optim.Adam([self.log_std], lr=self.log_std_lr)

        # replay buffer
        self.buffer_max_length = buffer_max_length
        self.buffer = deque(maxlen=self.buffer_max_length)
        self.buffer_filled = False                               # flag to indicate buffer is full
        self.batch_in_buffer = False                            # flag to indicate if n_batches is in buffer

        # actor and critic
        if actor:
            self.actor = actor
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.obs_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.n_actions)
            )
        self.target_actor = copy.deepcopy(self.actor)
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        if critic:
            self.critic = critic
        else:
            self.critic = nn.Sequential(
                nn.Linear(self.obs_size + self.n_actions, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) #, weight_decay=0.01) # WEIGHT DECAY CAUSES LEARNING ISSUES WITH MOUNTAIN CAR
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        self.critic_loss_fn = nn.MSELoss()

        # baseline and advantage
        self.baseline = baseline
        self.standardise = standardise
        if self.baseline:
            if baseline_model:
                self.baseline_model = baseline_model
            else:
                self.baseline_model = nn.Sequential(
                    nn.Linear(self.obs_size, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
            self.baseline_optimizer = torch.optim.Adam(self.baseline_model.parameters(), lr=critic_lr)
            self.baseline_loss_fn = nn.MSELoss()
            self.baseline_model.to(self.device)

        # defaults
        self.debug_mode = False         # print debug info
        self.training_mode = True       # flag to determine if training or testing
        self.steps = 0                  # steps since initialisation
        self.rng = np.random.default_rng()

    def agent_start(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        if self.debug_mode: print('Action:', action)
        return action.cpu().detach().numpy()

    def agent_step(self, reward, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        if self.training_mode: self.train_mode_actions(reward, observation, False) # must be before new action/obs replaces self.prev_action/self.prev_state

        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        if self.debug_mode: print('Action:', action)
        return action.cpu().detach().numpy()

    def agent_end(self, reward, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        if self.training_mode: self.train_mode_actions(reward, observation, True)

    def get_action(self, observation, return_dist=False):
        observation = observation.to(self.device)
        if self.discrete:
            logits = self.actor(observation)
            action_dist = torch.distributions.Categorical(logits=logits)
        else:
            batch_mean = self.actor(observation)
            batch_cov = torch.diag((2*self.log_std).exp()).to(self.device)
            # print(batch_mean.shape, batch_cov.shape) # (3) and (3,3)
            action_dist = torch.distributions.MultivariateNormal(batch_mean, covariance_matrix=batch_cov)

        if return_dist:
            return action_dist
        else:
            action = action_dist.sample().detach()
            return action.cpu()

    def train_mode_actions(self, reward, observation, terminal):
        self.steps += 1
        self.add_to_replay_buffer(reward, observation, terminal)
        if self.training_condition(): self.train()
        if self.update_target_net_condition(): self.update_target_networks()

    def update_target_networks(self):
        t = self.soft_param
        with torch.no_grad():
            for target_param, param, in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data = (1-t) * target_param.data + t * param.data
            for target_param, param, in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data = (1-t) * target_param.data + t * param.data

    def update_target_net_condition(self):
        bool_step_multiple = (self.steps % self.update_target_steps == 0)
        return bool_step_multiple and self.batch_in_buffer

    def training_condition(self):
        bool_step_multiple = (self.steps % self.train_steps == 0)
        return bool_step_multiple and self.batch_in_buffer

    def add_to_replay_buffer(self, reward, observation, terminal):
        terminal_state = torch.tensor([terminal], dtype=torch.bool)
        transition = (self.prev_state, self.prev_action, reward, observation, terminal_state)
        if self.buffer_filled:
            self.buffer.popleft()
            self.buffer.append(transition)
        else:
            self.buffer.append(transition)
            if len(self.buffer) == self.buffer_max_length:
                self.buffer_filled = True
        if not self.batch_in_buffer:
            if len(self.buffer) >= self.batch_size: self.batch_in_buffer = True

    def sample_batch(self):
        if self.n_batches * self.batch_size > len(self.buffer):
            return [torch.stack(i, dim=0) for i in [*zip(*self.buffer)]]
        else:
            return [torch.stack(i, dim=0) for i in [*zip(*random.sample(self.buffer, self.n_batches * self.batch_size))]]

    def train(self):
        current_states, actions, rewards, next_states, terminal_state = self.sample_batch()
        not_terminal = torch.logical_not(terminal_state)

        if self.n_batches * self.batch_size >= len(self.buffer):
            dataset = torch.utils.data.TensorDataset(current_states, actions, rewards, next_states, not_terminal)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            n_batches_trained = 0
            while n_batches_trained < self.n_batches:
                for current_states, actions, rewards, next_states, not_terminal in dataloader:
                    self.train_batch(current_states, actions, rewards, next_states, not_terminal)
                    n_batches_trained += 1
                    if n_batches_trained == self.n_batches: break
        else:
            for i in range(self.n_batches):
                start = i * self.batch_size
                end = (i+1) * self.batch_size
                self.train_batch(
                    current_states[start:end],
                    actions[start:end],
                    rewards[start:end],
                    next_states[start:end],
                    not_terminal[start:end]
                )

    def train_batch(self, current_states, actions, rewards, next_states, not_terminal):
        current_states, actions, rewards, next_states, not_terminal = self.to_device([current_states, actions, rewards, next_states, not_terminal])
        self.train_critic(current_states, actions, rewards, next_states, not_terminal)
        self.train_actor(current_states)

    def train_critic(self, current_states, actions, rewards, next_states, not_terminal):
        # compute targets = reward + gamma * target_q(next_state, action) where action = max(q(next_state)) i.e. double Q-learning
        with torch.no_grad():
            next_actions = self.get_action(next_states)
            next_state_q = self.target_critic(torch.cat([next_states, next_actions], dim=-1))
        targets = (rewards + self.discount * next_state_q * not_terminal)

        # compute current state q value = Q(current_state, action)
        current_state_q = self.critic(torch.cat([current_states, actions], dim=-1))
        # assert current_state_q.shape == targets.shape,  str(current_state_q.shape) + ' ' + str(targets.shape)

        loss = self.critic_loss_fn(targets, current_state_q)
        self.critic_optimizer.zero_grad()
        loss.backward() # retain graph required if log_std trainable and outside of model
        if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.clip_value)
        self.critic_optimizer.step()
        if self.debug_mode: print('Critic loss:', loss.mean().item())

        # train baseline
        if self.baseline:
            with torch.no_grad():
                next_state_v = self.baseline_model(next_states)
            baseline_targets = (rewards + self.discount * next_state_v * not_terminal)
            baseline = self.baseline_model(current_states)

            baseline_loss = self.baseline_loss_fn(baseline_targets, baseline)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.baseline_model.parameters(), self.clip_value)
            self.baseline_optimizer.step()
            if self.debug_mode: print('Baseline loss:', baseline_loss.mean().item())

        return loss

    # Refer to DPG paper for proof of the deterministic policy gradient
    def train_actor(self, current_states):
        actions = self.get_action(current_states) # no noise for actor training
        q_values = self.critic(torch.cat([current_states, actions], dim=-1))
        advantages = self.calculate_advantages(q_values, current_states).to(self.device)
        if self.debug_mode:
            print('Adv values:', advantages)
            for params in self.actor.parameters():
                print('Gradients:', params.grad)
        action_dist = self.get_action(current_states, return_dist=True)
        loss  = -torch.sum(action_dist.log_prob(actions) * advantages)
        self.actor_optimizer.zero_grad()
        loss.backward()

        if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.clip_value)
        self.actor_optimizer.step()

    def calculate_advantages(self, q_values, obs=None):
        # Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values

        if self.baseline:
            with torch.no_grad(): values_unnormalized = self.baseline_model(obs).cpu()
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.shape == q_values.shape, (values_unnormalized.shape, q_values.shape)
            ## values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values
            values = values_unnormalized * torch.std(q_values) + q_values.mean()
            advantages = q_values - values
        else:
            advantages = q_values

        if self.standardise: advantages = (advantages - advantages.mean()) / (torch.std(advantages) + 0.000001)

        return advantages

    def to_device(self, list):
        device_var = []
        for var in list:
            var = var.to(self.device)
            device_var.append(var)
        return device_var

    def analyse_train_actor(self):
        current_states, actions, _, _, _ = [torch.stack(i, dim=0) for i in [*zip(*random.sample(self.buffer, self.batch_size))]]
        # not_terminal = torch.logical_not(terminal_state)
        actions = self.get_action(current_states, require_grad=True)
        q_values = self.critic(torch.cat([current_states, actions], dim=-1))
        advantages = self.calculate_advantages(q_values, current_states).detach().cpu().numpy()
        return advantages, actions.detach().cpu().numpy()