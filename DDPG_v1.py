import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple
import random
import numpy as np
import copy

class DDPG_Agent:
    '''
    Implementation of DDPG with actor-critic
    - Determininistic policy with Gaussian noise added
    - train_steps controls # of steps between training iterations
    - n_batches controls # of batches to train in each training iteration
    - n_update_target_steps controls # of steps between soft updating of target networks
    - soft_param controls speed of soft updating of target network
    - sigma controls standard deviation of the noise added to action
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
    def agent_init(self, agent_init_info):
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info['num_actions']
        self.obs_size = agent_init_info['obs_size']
        self.actor_lr = agent_init_info['actor_lr']
        self.critic_lr = agent_init_info['critic_lr']
        self.discount = agent_init_info['discount']
        self.buffer_max_length = agent_init_info['buffer_max_length'] # num of steps
        self.batch_size = agent_init_info['batch_size']     # num of steps
        self.train_steps = agent_init_info['train_steps']   # num of steps between updates
        self.n_batches = agent_init_info['n_batches']       # num of batches per update step
        self.update_target_steps = agent_init_info['update_target_steps']
        self.soft_param = agent_init_info['soft_param']
        self.sigma = torch.tensor(agent_init_info['sigma'])
        self.clip_gradients = agent_init_info['clip_gradients']
        if self.clip_gradients: self.clip_value = agent_init_info['clip_value']
        self.device = agent_init_info['device']
        self.rng = np.random.default_rng()

        self.debug_mode = False

        try: self.standardise = agent_init_info['standardise']
        except: self.standardise = True

        try: self.baseline = agent_init_info['baseline']
        except: self.baseline = False

        try: self.softmax = agent_init_info['softmax']
        except: self.softmax = False

        try:
            self.action_limit_value = agent_init_info['action_limit_value']
            self.action_limit = True
        except:
            self.action_limit = False

        if self.softmax == True and self.action_limit == True:
            raise Exception('Cannot use softmax with action limits')

        try: self.actor = agent_init_info['actor_model']
        except:
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
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        try: self.critic = agent_init_info['critic_model']
        except:
            self.critic = nn.Sequential(
                nn.Linear(self.obs_size + self.num_actions, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr) #, weight_decay=0.01) # WEIGHT DECAY CAUSES LEARNING ISSUES WITH MOUNTAIN CAR
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        self.critic_loss_fn = nn.MSELoss()

        if self.baseline:
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

        self.training_mode = True
        self.steps = 0
        self.buffer_max_length = agent_init_info['buffer_max_length'] # in terms of steps
        self.buffer = deque(maxlen=self.buffer_max_length)
        self.buffer_filled = False
        self.batch_in_buffer = False

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

    def get_action(self, observation):
        with torch.no_grad(): action = self.actor(observation.to(self.device))
        if self.training_mode: noise = torch.normal(torch.zeros(self.num_actions), self.sigma * torch.ones(self.num_actions))
        else: noise = torch.zeros(self.num_actions) # noise only when interacting with environment in training mode
        noise = noise.to(self.device)
        assert action.shape == noise.shape, str(action.shape) + str(noise.shape)
        if self.softmax: return F.softmax(action + noise, dim=-1) # when actions includes risk free asset and adding softmax to remove leverage / shortselling
        elif self.action_limit: return nn.Tanh()(action + noise) * self.action_limit_value
        else: return (action + noise)

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
            if len(self.buffer) >= self.n_batches * self.batch_size: self.batch_in_buffer = True

    def sample_batch(self): return [torch.stack(i, dim=0) for i in [*zip(*random.sample(self.buffer, self.n_batches * self.batch_size))]]

    def train(self):
        current_states, actions, rewards, next_states, terminal_state = self.sample_batch()
        not_terminal = torch.logical_not(terminal_state)
        for i in range(self.n_batches):
            self.train_batch(
                current_states[i*self.batch_size:(i+1)*self.batch_size], 
                actions[i*self.batch_size:(i+1)*self.batch_size], 
                rewards[i*self.batch_size:(i+1)*self.batch_size], 
                next_states[i*self.batch_size:(i+1)*self.batch_size], 
                not_terminal[i*self.batch_size:(i+1)*self.batch_size]
            )

    def train_batch(self, current_states, actions, rewards, next_states, not_terminal):
        current_states, actions, rewards, next_states, not_terminal = self.to_device([current_states, actions, rewards, next_states, not_terminal])
        self.train_critic(current_states, actions, rewards, next_states, not_terminal)
        self.train_actor(current_states)

    def train_critic(self, current_states, actions, rewards, next_states, not_terminal):
        # compute targets = reward + gamma * target_q(next_state, action) where action = max(q(next_state)) i.e. double Q-learning
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_state_q = self.target_critic(torch.cat([next_states, next_actions], dim=-1))
        targets = (rewards + self.discount * next_state_q * not_terminal)

        # compute current state q value = Q(current_state, action)
        current_state_q = self.critic(torch.cat([current_states, actions], dim=-1))
        assert current_state_q.shape == targets.shape,  str(current_state_q.shape) + ' ' + str(targets.shape)

        loss = self.critic_loss_fn(targets, current_state_q)
        self.critic_optimizer.zero_grad()
        loss.backward()
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
        actions = self.actor(current_states) # No noise for calculate critic value for training actor
        q_values = self.critic(torch.cat([current_states, actions], dim=-1))
        advantages = self.calculate_advantages(q_values, current_states).to(self.device)
        if self.debug_mode: print('Adv values:', advantages.mean().item())
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
            advantages = q_values.copy()

        if self.standardise: advantages = (advantages - advantages.mean()) / (torch.std(advantages) + 0.0001)
        return advantages

    def to_device(self, list):
        device_var = []
        for var in list:
            var = var.to(self.device)
            device_var.append(var)
        return device_var