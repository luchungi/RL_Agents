import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import time

from replay_buffer.prioritized_buffer import Buffer

class AC_PER_Agent:
    '''
    Implementation of DDPG with actor-critic
    - Determininistic policy with Gaussian noise added
    - max buffer length controls the maximum length of the replay buffer in terms of steps
    - train_steps controls # of steps between training iterations
    - n_batches controls # of batches to train in each training iteration
    - n_update_target_steps controls # of steps between soft updating of target networks
    - soft_param controls speed of soft updating of target network
    - sigma controls standard deviation of the noise added to action
    - end_sigma controls the final value of sigma after sigma_steps
    - noise_corr controls the correlation between noise added to actions (default is zero corr)
    - the noise is only added during training mode
    - clip_gradients controls whether to clip the gradient to [-clip_value, clip_value]
    - softmax controls whether to use softmax on the action (post noise addition)
    - baseline will train a value function based on state to calculate advantage and reduce variance
    - difficult to implement GAE as it requires calculating the advantage for a full episode rollout
      which is in conflict with the replay buffer and train during episode style
    - standardise will normalise the advantage
    - action limit will add a tanh layer to the output of the actor multiplied by the
      action limit value i.e. action in range [-value, value]
    - actor/critic/baseline models default to the PPO NN architecture for better comparison
    '''

    def __init__(self, num_actions, obs_size, actor_lr=0.001, critic_lr=0.001, discount=0.99, buffer_max_length=int(1e6),
        batch_size=128, train_critic_steps=8, train_actor_steps=16, n_batches=1, update_target_steps=8,
        baseline=True, standardise=True, soft_param=0.005,
        discrete=False, log_std_init=0.0, log_std_lr=0.001,
        clip_gradients=False, clip_value=None,
        actor=None, critic=None, baseline_model=None, device='cpu'
    ):

        self.time1 = 0
        self.time2 = 0
        self.time3 = 0

        self.num_actions = num_actions
        self.obs_size = obs_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discount = discount
        self.buffer_max_length = buffer_max_length
        self.batch_size = batch_size
        self.train_critic_steps = train_critic_steps
        self.train_actor_steps = train_actor_steps
        self.n_batches = n_batches
        self.update_target_steps = update_target_steps
        self.soft_param = soft_param
        self.clip_gradients = clip_gradients
        self.clip_value = clip_value
        self.device = device
        self.rng = np.random.default_rng()
        self.baseline = baseline
        self.standardise = standardise
        self.discrete = discrete
        self.rng = np.random.default_rng()

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
        # self.target_actor = copy.deepcopy(self.actor)
        self.actor.to(self.device)
        # self.target_actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(self.num_actions, dtype=torch.float32) + log_std_init)
            self.log_std_optim = torch.optim.Adam([self.log_std], lr=log_std_lr)

        if critic:
            self.critic = critic
            # self.critic2 = copy.deepcopy(self.critic1)
            # for layer in self.critic2.children():
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
        else:
            self.critic = nn.Sequential(
                nn.Linear(self.obs_size + self.num_actions, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            # self.critic2 = nn.Sequential(
            #     nn.Linear(self.obs_size + self.num_actions, 64),
            #     nn.Tanh(),
            #     nn.Linear(64, 64),
            #     nn.Tanh(),
            #     nn.Linear(64, 1),
            # )

        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr) #, weight_decay=0.01) # WEIGHT DECAY CAUSES LEARNING ISSUES WITH MOUNTAIN CAR
        # self.target_critic2 = copy.deepcopy(self.critic2)
        # self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.critic_lr) #, weight_decay=0.01)

        self.critic.to(self.device)
        self.target_critic.to(self.device)
        # self.critic2.to(self.device)
        # self.target_critic2.to(self.device)

        self.critic_loss_fn = nn.MSELoss()

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
            self.baseline_optimizer = torch.optim.Adam(self.baseline_model.parameters(), lr=self.critic_lr)
            self.baseline_loss_fn = nn.MSELoss()
            self.baseline_model.to(self.device)

        self.debug_mode = False
        self.training_mode = True
        self.steps = 0
        self.buffer = Buffer(self.buffer_max_length)
        self.buffer_filled = False
        self.batch_in_buffer = False

    def agent_start(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        if self.debug_mode: print('Action:', action)
        return action.numpy()

    def agent_step(self, reward, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        if self.training_mode: self.train_mode_actions(reward, observation, False) # must be before new action/obs replaces self.prev_action/self.prev_state

        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        if self.debug_mode: print('Action:', action)
        return action.numpy()

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
            # print(action.shape)
            return action.cpu()

    def train_mode_actions(self, reward, observation, terminal):
        self.steps += 1

        # reward, next state, current state terminal status i.e. next_state is irrelevant if terminal
        self.add_to_replay_buffer(reward, observation, terminal)
        if self.training_condition(): self.train()
        if self.update_target_net_condition(): self.update_target_networks()

    def update_target_networks(self):
        t = self.soft_param
        with torch.no_grad():
            # for target_param, param, in zip(self.target_actor.parameters(), self.actor.parameters()):
            #     target_param.data = (1-t) * target_param.data + t * param.data
            for target_param, param, in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data = (1-t) * target_param.data + t * param.data
            # for target_param, param, in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            #     target_param.data = (1-t) * target_param.data + t * param.data
            # for target_param, param, in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            #     target_param.data = (1-t) * target_param.data + t * param.data

    def update_target_net_condition(self):
        bool_step_multiple = (self.steps % self.update_target_steps == 0)
        return bool_step_multiple and self.batch_in_buffer

    def training_condition(self):
        bool_step_multiple = (self.steps % self.train_critic_steps == 0) or (self.steps % self.train_actor_steps == 0)
        return bool_step_multiple and self.batch_in_buffer

    def add_to_replay_buffer(self, reward, observation, terminal):
        with torch.no_grad():
            old_val = self.critic(torch.cat([self.prev_state, self.prev_action]))

            next_action = self.actor(observation)
            if terminal:
                new_val = reward
            else:
                new_val = reward + self.discount * self.target_critic(torch.cat([observation, next_action]))

            error = abs(old_val - new_val)

        terminal_state = torch.tensor([terminal], dtype=torch.bool)
        transition = (self.prev_state, self.prev_action, reward, observation, terminal_state)
        self.buffer.add(error, transition)
        if not self.buffer_filled:
            if len(self.buffer) == self.buffer_max_length: self.buffer_filled = True
        if not self.batch_in_buffer:
            if len(self.buffer) >= self.batch_size * self.n_batches: self.batch_in_buffer = True

    def sample_batch(self):
        batch, idxs, is_weights = self.buffer.sample(self.batch_size)
        batch = [torch.stack(i) for i in [*zip(*batch)]]
        is_weights = torch.tensor(is_weights, dtype=torch.float32)
        current_states, actions, rewards, next_states, terminal_state = batch
        return current_states, actions, rewards, next_states, terminal_state, idxs, is_weights

    def train(self):
        current_states, actions, rewards, next_states, terminal_state, idxs, is_weights = self.sample_batch()
        not_terminal = torch.logical_not(terminal_state)
        self.train_batch(current_states, actions, rewards, next_states, not_terminal, idxs, is_weights)

    def train_batch(self, current_states, actions, rewards, next_states, not_terminal, idxs, is_weights):
        current_states, actions, rewards, next_states, not_terminal, is_weights = self.to_device([current_states, actions, rewards, next_states, not_terminal, is_weights])
        if (self.steps % self.train_critic_steps == 0):
            self.train_critic(current_states, actions, rewards, next_states, not_terminal, idxs, is_weights)
        if (self.steps % self.train_actor_steps == 0):
            self.train_actor(current_states, actions)

    def train_critic(self, current_states, actions, rewards, next_states, not_terminal, idxs, is_weights):
        # clipped double Q-learning

        is_weights = torch.FloatTensor(is_weights).to(self.device)

        # with torch.no_grad():
        #     next_actions = self.target_actor(next_states)
        #     next_state_q = torch.minimum(
        #         self.target_critic1(torch.cat([next_states, next_actions], dim=-1)),
        #         self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
        #     )
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_state_q = self.target_critic(torch.cat([next_states, next_actions], dim=-1))
        targets = (rewards + self.discount * next_state_q * not_terminal)

        # compute current state q value = Q(current_state, action)
        current_state_q = self.critic(torch.cat([current_states, actions], dim=-1))
        assert current_state_q.shape == targets.shape,  str(current_state_q.shape) + ' ' + str(targets.shape)
        # current_state_q1 = self.critic1(torch.cat([current_states, actions], dim=-1))
        # current_state_q2 = self.critic2(torch.cat([current_states, actions], dim=-1))
        # assert current_state_q1.shape == targets.shape,  str(current_state_q1.shape) + ' ' + str(targets.shape)

        loss = (is_weights * F.mse_loss(current_state_q, targets)).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()

        # loss1 = self.critic_loss_fn(targets, current_state_q1)
        # self.critic1_optimizer.zero_grad()
        # loss1.backward()

        # loss2 = self.critic_loss_fn(targets, current_state_q2)
        # self.critic2_optimizer.zero_grad()
        # loss2.backward()

        # if self.clip_gradients:
        #     torch.nn.utils.clip_grad_value_(self.critic1.parameters(), self.clip_value)
        #     torch.nn.utils.clip_grad_value_(self.critic2.parameters(), self.clip_value)

        # self.critic1_optimizer.step()
        # self.critic2_optimizer.step()

        # if self.debug_mode: print('Critic 1/2 loss:', loss1.mean().item() + '/', loss2.mean().item())

        # train baseline
        if self.baseline:
            with torch.no_grad():
                next_state_v = self.baseline_model(next_states)
            baseline_targets = (rewards + self.discount * next_state_v * not_terminal)
            baseline = self.baseline_model(current_states)

            baseline_loss = (is_weights * F.mse_loss(baseline_targets, baseline)).mean()
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            # if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.baseline_model.parameters(), self.clip_value)
            self.baseline_optimizer.step()
            if self.debug_mode: print('Baseline loss:', baseline_loss.mean().item())

        # update priority
        with torch.no_grad():
            errors = torch.abs(targets - current_state_q).cpu().numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.buffer.update(idx, errors[i])

    # Refer to DPG paper for proof of the deterministic policy gradient
    def train_actor(self, current_states, actions):
        q_values = self.critic(torch.cat([current_states, self.actor(current_states)], dim=-1))
        advantages = self.calculate_advantages(q_values, current_states).to(self.device)

        action_dist = self.get_action(current_states, return_dist=True)
        log_prob = action_dist.log_prob(actions).unsqueeze(-1)
        assert log_prob.shape == advantages.shape, (log_prob.shape, advantages.shape)

        loss = -torch.sum(advantages * log_prob) # policy gradient for stochastic policy

        self.actor_optimizer.zero_grad()
        if not self.discrete: self.log_std_optim.zero_grad()

        loss.backward()
        self.actor_optimizer.step()

        if not self.discrete:
            self.log_std_optim.step()

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
        current_states, actions, _, _, _ = [torch.stack(i, dim=0) for i in [*zip(*random.sample(self.buffer, 32))]]
        # not_terminal = torch.logical_not(terminal_state)
        actions = self.actor(current_states) # No noise for calculate critic value for training actor
        q_values = self.critic1(torch.cat([current_states, actions], dim=-1))
        advantages = self.calculate_advantages(q_values, current_states).to(self.device)
        return advantages, actions