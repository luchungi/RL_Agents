import random
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time

class TD3_Agent:
    '''
    Implementation of TD3
    Clipped double Q learning where min of Q1 and Q2 is used for target
    Gaussian noise is added to actions for robustness
    Added baseline model to calculate advantage and reduce variance
    - train_critic_steps: number of environment steps between each critic update
    - train_actor_steps: number of environment steps between each actor update
    Delayed policy updates means train_actor_steps should be larger than train_critic_steps (e.g. x2 as in paper)
    - update_target_steps: number of environment steps between each target network soft update
    - soft_param: soft update parameter
    Refer to DDPG for other param descriptions
    '''

    def __init__(self, num_actions, obs_size, actor_lr=0.001, critic_lr=0.001, discount=0.99,
        buffer_max_length=int(1e6), batch_size=128,
        train_critic_steps=8, train_actor_steps=16, actor_n_batches=1, critic_n_batches=1,
        update_target_steps=8, learning_starts=128,
        baseline=True, standardise=True, soft_param=0.005,
        softmax=False, action_limit=False, action_limit_value=None,
        sigma=0.1, end_sigma=0.1, sigma_steps=None, noise_corr=0,
        target_noise_sigma=0.001, target_clip_value=0.01,
        clip_gradients=False, clip_value=None,
        actor=None, critic=None, baseline_model=None,
        device='cpu'
    ):

        self.num_actions = num_actions
        self.obs_size = obs_size
        self.discount = discount
        self.batch_size = batch_size
        self.actor_n_batches = actor_n_batches
        self.critic_n_batches = critic_n_batches
        self.train_critic_steps = train_critic_steps
        self.train_actor_steps = train_actor_steps
        self.update_target_steps = update_target_steps
        self.soft_param = soft_param
        self.clip_gradients = clip_gradients
        self.clip_value = clip_value
        self.device = device

        # leverage and shorting constraints
        self.softmax = softmax
        self.action_limit = action_limit
        self.action_limit_value = action_limit_value

        # noise for exploration
        self.sigma = sigma
        self.rho = noise_corr
        if end_sigma != self.sigma:
            self.sigma_decay = (end_sigma - self.sigma) / sigma_steps
            self.sigma_steps = sigma_steps
        else: self.sigma_decay = None

        # noise for target policy smoothing
        self.target_noise_sigma = target_noise_sigma
        self.target_clip_value = target_clip_value

        # replay buffer
        self.buffer_max_length = buffer_max_length
        self.buffer = deque(maxlen=self.buffer_max_length)
        self.buffer_filled = False                               # flag to indicate buffer is full

        # actor and critic
        self.learning_starts = learning_starts                  # number of steps before learning starts
        self.start_learning = False
        if actor:
            self.actor = actor
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.obs_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.num_actions)
            )
        self.target_actor = copy.deepcopy(self.actor)
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        if critic:
            self.critic1 = critic
            self.critic2 = copy.deepcopy(self.critic1)
            for layer in self.critic2.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        else:
            self.critic1 = nn.Sequential(
                nn.Linear(self.obs_size + self.num_actions, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            self.critic2 = nn.Sequential(
                nn.Linear(self.obs_size + self.num_actions, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr) #, weight_decay=0.01) # WEIGHT DECAY CAUSES LEARNING ISSUES WITH MOUNTAIN CAR
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr) #, weight_decay=0.01)

        self.critic1.to(self.device)
        self.target_critic1.to(self.device)
        self.critic2.to(self.device)
        self.target_critic2.to(self.device)
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

    def get_action(self, observation, require_grad=False, target=False):
        observation = observation.to(self.device)

        if target:
            # add noise for target smoothing policy regularization (see TD3 paper)
            with torch.no_grad(): action = self.target_actor(observation)
            noise = torch.normal(torch.zeros(self.num_actions), self.target_noise_sigma * torch.ones(self.num_actions)).to(self.device)
            noise = torch.clamp(noise, -self.target_clip_value, self.target_clip_value).to(self.device)
        else:
            if require_grad:
                action = self.actor(observation)
            else:
                with torch.no_grad(): action = self.actor(observation)

            # add noise only when interacting with environment in training mode
            # no noise for actor training
            if (not require_grad) and (self.training_mode):
                cov = np.eye(self.num_actions, dtype=np.float32) * self.sigma**2
                cov[~np.eye(self.num_actions, dtype=bool)] = np.float32(self.sigma**2 * self.rho)
                noise = self.rng.multivariate_normal(np.zeros(self.num_actions), cov)
                noise = torch.tensor(noise, dtype=torch.float32).to(self.device)
                assert action.shape == noise.shape, f'action shape {action.shape} / noise shape {noise.shape} / require_grad {require_grad} / target {target}'
            else:
                noise = torch.zeros(self.num_actions).to(self.device)

        # when actions includes risk free asset and adding softmax to remove leverage / shortselling
        # limit action to min/max position based on action_limit_value (from env)
        # use in combination to remove shortselling but allow leverage
        if self.softmax and self.action_limit: return torch.clamp(nn.Sigmoid()(action) * self.action_limit_value + noise, 0, self.action_limit_value)
        elif self.softmax: return nn.Softmax()(nn.Softmax()(action, dim=-1) + noise)
        elif self.action_limit: return torch.clamp(nn.Tanh()(action) * self.action_limit_value + noise, -self.action_limit_value, self.action_limit_value)
        else: return action

    def train_mode_actions(self, reward, observation, terminal):
        self.steps += 1

        # reward, next state, current state terminal status i.e. next_state is irrelevant if terminal
        self.add_to_replay_buffer(reward, observation, terminal)
        if self.training_condition(): self.train()
        if self.update_target_net_condition(): self.update_target_networks()
        if self.sigma_decay and self.steps < self.sigma_steps: self.sigma += self.sigma_decay

    def update_target_networks(self):
        t = self.soft_param
        with torch.no_grad():
            for target_param, param, in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data = (1-t) * target_param.data + t * param.data
            for target_param, param, in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data = (1-t) * target_param.data + t * param.data
            for target_param, param, in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data = (1-t) * target_param.data + t * param.data

    def update_target_net_condition(self):
        bool_step_multiple = (self.steps % self.update_target_steps == 0)
        return bool_step_multiple and self.start_learning

    def training_condition(self):
        bool_step_multiple = (self.steps % self.train_critic_steps == 0) or (self.steps % self.train_actor_steps == 0)
        return bool_step_multiple and self.start_learning

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
        if not self.start_learning and len(self.buffer) == self.learning_starts:
            self.start_learning = True

    def sample_buffer(self, n_batches):
        # return entire buffer if buffer is smaller than batch size else return random sample of batch size
        if n_batches * self.batch_size > len(self.buffer):
            return [torch.stack(i, dim=0) for i in [*zip(*self.buffer)]]
        else:
            return [torch.stack(i, dim=0) for i in [*zip(*random.sample(self.buffer, n_batches * self.batch_size))]]

    def train(self):
        if (self.steps % self.train_critic_steps == 0):
            current_states, actions, rewards, next_states, terminal_state = self.sample_buffer(self.critic_n_batches)
            not_terminal = torch.logical_not(terminal_state)
            dataset = torch.utils.data.TensorDataset(current_states, actions, rewards, next_states, not_terminal)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            critic_n_batches_trained = 0
            while (critic_n_batches_trained < self.critic_n_batches):
                for current_states, actions, rewards, next_states, not_terminal in dataloader:
                    current_states, actions, rewards, next_states, not_terminal = self.to_device([current_states, actions, rewards, next_states, not_terminal])
                    self.train_critic(current_states, actions, rewards, next_states, not_terminal)
                    critic_n_batches_trained += 1
                if (critic_n_batches_trained == self.critic_n_batches): break

        if (self.steps % self.train_actor_steps == 0):
            current_states, actions, rewards, next_states, terminal_state = self.sample_buffer(self.actor_n_batches)
            not_terminal = torch.logical_not(terminal_state)
            dataset = torch.utils.data.TensorDataset(current_states, actions, rewards, next_states, not_terminal)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            actor_n_batches_trained = 0
            while (actor_n_batches_trained < self.actor_n_batches):
                current_states, actions, rewards, next_states, terminal_state = self.sample_buffer(self.actor_n_batches)
                for current_states, actions, rewards, next_states, not_terminal in dataloader:
                    current_states, actions, rewards, next_states, not_terminal = self.to_device([current_states, actions, rewards, next_states, not_terminal])
                    self.train_actor(current_states)
                    actor_n_batches_trained += 1
                if (actor_n_batches_trained == self.actor_n_batches): break

        # print(f"critic_n_batches_trained: {critic_n_batches_trained}, actor_n_batches_trained: {actor_n_batches_trained}")

    def train_critic(self, current_states, actions, rewards, next_states, not_terminal):
        # clipped double Q-learning
        with torch.no_grad():
            next_actions = self.get_action(next_states, target=True)
            next_state_q = torch.minimum(
                self.target_critic1(torch.cat([next_states, next_actions], dim=-1)),
                self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
            )
        targets = (rewards + self.discount * next_state_q * not_terminal)

        # compute current state q value = Q(current_state, action)
        current_state_q1 = self.critic1(torch.cat([current_states, actions], dim=-1))
        current_state_q2 = self.critic2(torch.cat([current_states, actions], dim=-1))
        assert current_state_q1.shape == targets.shape,  str(current_state_q1.shape) + ' ' + str(targets.shape)

        loss1 = self.critic_loss_fn(targets, current_state_q1)
        self.critic1_optimizer.zero_grad()
        loss1.backward()

        loss2 = self.critic_loss_fn(targets, current_state_q2)
        self.critic2_optimizer.zero_grad()
        loss2.backward()

        if self.clip_gradients:
            torch.nn.utils.clip_grad_value_(self.critic1.parameters(), self.clip_value)
            torch.nn.utils.clip_grad_value_(self.critic2.parameters(), self.clip_value)

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        if self.debug_mode: print('Critic 1/2 loss:', loss1.mean().item() + '/', loss2.mean().item())

        # train baseline
        if self.baseline:
            with torch.no_grad(): next_state_v = self.baseline_model(next_states)
            baseline_targets = (rewards + self.discount * next_state_v * not_terminal)
            baseline = self.baseline_model(current_states)

            baseline_loss = self.baseline_loss_fn(baseline_targets, baseline)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.baseline_model.parameters(), self.clip_value)
            self.baseline_optimizer.step()
            if self.debug_mode: print('Baseline loss:', baseline_loss.mean().item())

    def train_actor(self, current_states):
        actions = self.get_action(current_states, require_grad=True) # no noise for actor training
        q_values = self.critic1(torch.cat([current_states, actions], dim=-1)).cpu()
        advantages = self.calculate_advantages(q_values, current_states).to(self.device)

        loss = -torch.mean(advantages)
        self.actor_optimizer.zero_grad()
        loss.backward()

        if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.clip_value)
        self.actor_optimizer.step()

    def calculate_advantages(self, q_values, obs=None):
        if self.baseline:
            with torch.no_grad(): values_unnormalized = self.baseline_model(obs).cpu()
            assert values_unnormalized.shape == q_values.shape, (values_unnormalized.shape, q_values.shape)
            ## values were trained with standardized q_values -> ensure predictions have same mean & std as current batch of q_values
            values = values_unnormalized * torch.std(q_values) + q_values.mean()
            advantages = q_values - values
        else:
            advantages = q_values.copy()

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
        q_values = self.critic1(torch.cat([current_states, actions], dim=-1))
        advantages = self.calculate_advantages(q_values, current_states).detach().cpu().numpy()
        return advantages, actions.detach().cpu().numpy()