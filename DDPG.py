import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import random
import numpy as np
import copy

class DDPG_Agent:
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

    def __init__(self, num_actions, obs_size,
        discount=0.99,
        actor_lr=0.001, critic_lr=0.001, train_steps=8, update_target_steps=8, soft_param=0.005,
        buffer_max_length=int(1e6), batch_size=128, n_batches=1,
        baseline=False, standardise=False,
        no_leverage=False, no_shorting=False, squash_action=False, action_limit=False, action_limit_value=None,
        sigma=0.1, end_sigma=0.1, sigma_steps=None, noise_corr=0,
        clip_gradients=False, clip_value=None,
        actor=None, critic=None, baseline_model=None,
        next_state_action=False,
        device='cpu'
    ):

        self.num_actions = num_actions
        self.obs_size = obs_size
        self.discount = discount
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.train_steps = train_steps
        self.update_target_steps = update_target_steps
        self.soft_param = soft_param
        self.clip_gradients = clip_gradients
        self.clip_value = clip_value
        self.device = device

        # for GBM project
        self.next_state_action = next_state_action

        # leverage and shorting constraints
        self.no_leverage = no_leverage
        self.no_shorting = no_shorting
        self.squash_action = squash_action
        self.action_limit = action_limit
        self.action_limit_value = action_limit_value

        # noise for exploration
        self.sigma = sigma
        self.rho = noise_corr
        if end_sigma != self.sigma:
            self.sigma_decay = (end_sigma - self.sigma) / sigma_steps
            self.sigma_steps = sigma_steps
        else: self.sigma_decay = None

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
                nn.Linear(64, self.num_actions)
            )
        self.target_actor = copy.deepcopy(self.actor)
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        if critic:
            self.critic = critic
        else:
            self.critic = nn.Sequential(
                nn.Linear(self.obs_size + self.num_actions, 64),
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

    def get_action(self, observation, require_grad=False, target=False):
        observation = observation.to(self.device)
        if require_grad:
            action = self.actor(observation)
        else:
            with torch.no_grad():
                if target: action = self.target_actor(observation)
                else: action = self.actor(observation)

        # add noise only when interacting with environment in training mode
        # no noise for actor training
        if (not require_grad) and (not target) and (self.training_mode):
            cov = np.eye(self.num_actions, dtype=np.float32) * self.sigma**2
            cov[~np.eye(self.num_actions, dtype=bool)] = np.float32(self.sigma**2 * self.rho)
            noise = self.rng.multivariate_normal(np.zeros(self.num_actions), cov)
            noise = torch.tensor(noise, dtype=torch.float32).to(self.device)
            assert action.shape == noise.shape, f'action shape {action.shape} / noise shape {noise.shape} / require_grad {require_grad} / target {target}'
        else:
            noise = torch.zeros(self.num_actions).to(self.device)

        # squash action uses tanh or softmax depending on whether leverage is allowed
        # else clamp action to action_limit_value
        if self.squash_action:
            if self.no_shorting: action = torch.clamp(nn.Softmax()(action) * self.action_limit_value + noise, 0, self.action_limit_value)
            else: action = torch.clamp(nn.Tanh()(action) * self.action_limit_value + noise, -self.action_limit_value, self.action_limit_value)
        elif self.action_limit:
            if self.no_shorting: action = torch.clamp(action + noise, 0, self.action_limit_value)
            else: action = torch.clamp(action + noise, -self.action_limit_value, self.action_limit_value)
        if self.no_leverage: action = nn.softmax()(action)
        return action

    def train_mode_actions(self, reward, observation, terminal):
        self.steps += 1
        self.add_to_replay_buffer(reward, observation, terminal)
        if self.training_condition(): self.train()
        if self.update_target_net_condition(): self.update_target_networks()
        if self.sigma_decay and self.steps < self.sigma_steps: self.sigma += self.sigma_decay

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
            if self.next_state_action:
                next_state_q = self.target_critic(torch.cat([next_states, actions], dim=-1))
            else:
                next_actions = self.get_action(next_states, target=True) # no noise added in DDPG (only in TD3)
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
        actions = self.get_action(current_states, require_grad=True) # no noise for actor training
        q_values = self.critic(torch.cat([current_states, actions], dim=-1))
        advantages = self.calculate_advantages(q_values, current_states).to(self.device)
        if self.debug_mode:
            print('Adv values:', advantages)
            for params in self.actor.parameters():
                print('Gradients:', params.grad)
        loss = -torch.mean(advantages)
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