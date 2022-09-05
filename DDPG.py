import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple
import random
import numpy as np
import copy

class DDPG_Agent:
    def agent_init(self, agent_init_info):
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info['num_actions']
        self.obs_size = agent_init_info['obs_size']
        self.actor_lr = agent_init_info['actor_lr']
        self.critic_lr = agent_init_info['critic_lr']
        self.discount = agent_init_info['discount']
        self.buffer_max_length = agent_init_info['buffer_max_length'] # num of steps
        self.batch_size = agent_init_info['batch_size'] # num of steps
        self.train_steps = agent_init_info['train_steps']
        self.n_batches = agent_init_info['n_batches']
        self.n_critic_steps = agent_init_info['n_critic_steps']
        self.n_actor_steps = agent_init_info['n_critic_steps']
        self.update_target_steps = agent_init_info['update_target_steps']
        self.soft_param = agent_init_info['soft_param']
        self.sigma = torch.tensor(agent_init_info['sigma'])
        self.clip_gradients = agent_init_info['clip_gradients']
        self.clip_value = agent_init_info['clip_value']
        self.device = agent_init_info['device']
        self.softmax = agent_init_info['softmax']
        self.rng = np.random.default_rng()        

        try: self.actor = agent_init_info['actor_model']
        except:
            self.actor = nn.Sequential(
                nn.Linear(self.obs_size, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),                
                nn.Linear(32, self.num_actions)
            )
        self.target_actor = copy.deepcopy(self.actor)
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        try: self.critic = agent_init_info['critic_model']
        except: self.critic = nn.Sequential(
                nn.Linear(self.obs_size + self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr) #, weight_decay=0.01) # WEIGHT DECAY CAUSES LEARNING ISSUES WITH MOUNTAIN CAR
        self.critic.to(self.device)
        self.target_critic.to(self.device)
        self.critic_loss_fn = nn.MSELoss()

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
        return action.cpu().detach().numpy()

    def agent_step(self, reward, observation):
        observation = torch.tensor(observation, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        if self.training_mode: self.train_mode_actions(reward, observation, False) # must be before new action/obs replaces self.prev_action/self.prev_state

        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
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
        if self.softmax: return F.relu(action + noise, dim=-1) # when actions includes risk free asset and adding softmax to remove leverage / shortselling
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
            if len(self.buffer) >= self.batch_size: self.batch_in_buffer = True

    def sample_batch(self): return [torch.stack(i, dim=0) for i in [*zip(*random.sample(self.buffer, self.batch_size))]]        

    def train(self):
        current_states, actions, rewards, next_states, terminal_state = self.sample_batch()
        not_terminal = torch.logical_not(terminal_state)
        self.train_batch(current_states, actions, rewards, next_states, not_terminal)
 
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
        return loss

    # Refer to DPG paper for proof of the deterministic policy gradient
    def train_actor(self, current_states):
        actions = self.actor(current_states) # No noise for calculate critic value for training actor
        q_values = self.critic(torch.cat([current_states, actions], dim=-1))
        loss = -torch.mean(q_values)
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.clip_value)
        self.actor_optimizer.step()

    def to_device(self, list):
        device_var = []
        for var in list:
            var = var.to(self.device)
            device_var.append(var)
        return device_var