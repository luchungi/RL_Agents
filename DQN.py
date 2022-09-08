import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import copy

class BufferDataset(Dataset):
    def __init__(self, ds_list):
        self.dataset = ds_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        current_state, action, rewards, next_state, terminal_state = self.dataset[idx]      
        return current_state, action, rewards, next_state, terminal_state

class DQNAgent:
    '''
    Implementation of Double DQN 
    Policy is implicit using argmax of q-values determined by q-network  
    clone_steps controls # of steps before target network weights are updated to most recent q-network weights
    train_steps controls # of steps between each training iteration
    n_epochs controls # of epochs for each training iteration
    n_batches controls # of batches to train in each epoch
    clip gradients controls whether to clip gradient to [-1,1]
    greedy controls whether to use episilon greedy
    epsilon controls the epsilon greedy param
    training_mode controls whether to switch off update buffer, target network and train q-network    
    '''
    def agent_init(self, agent_init_info):
        ''' Store the parameters provided in agent_init_info. '''
        self.num_actions = agent_init_info['num_actions']
        self.obs_shape = agent_init_info['obs_shape']
        self.epsilon = agent_init_info['epsilon']
        self.step_size = agent_init_info['step_size']
        self.discount = agent_init_info['discount']
        self.batch_size = agent_init_info['batch_size']
        self.buffer_max_length = agent_init_info['buffer_max_length'] # in terms of steps
        self.clone_steps = agent_init_info['clone_steps']
        self.train_steps = agent_init_info['train_steps']
        self.n_batches = agent_init_info['n_batches']
        self.n_epochs = agent_init_info['n_epochs']
        self.clip_gradients = agent_init_info['clip_gradients']
        self.q = agent_init_info['model']
        self.device = agent_init_info['device']
        self.greedy = False
        self.training_mode = True
        self.steps = 0
        self.buffer = []
        self.buffer_filled = False
        self.target_q = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.step_size)
        self.loss_fn = torch.nn.MSELoss()
        self.rng = np.random.default_rng()

    def agent_start(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        return action.cpu().numpy().squeeze()

    def agent_step(self, reward, observation):
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        if self.training_mode: self.train_mode_actions(reward, observation, False) # must be before new action/obs replaces self.prev_action/self.prev_state

        action = self.get_action(observation)
        self.prev_state = observation
        self.prev_action = action
        return action.cpu().numpy().squeeze()
    
    def agent_end(self, reward, observation):
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        if self.training_mode: self.train_mode_actions(reward, observation, True)
    
    def get_action(self, observation, batched=False, double_q_learning=False):
        if not batched: observation = observation[np.newaxis,...]

        q_values = self.q(observation)
        actions = torch.argmax(q_values, dim=-1)
        
        if (not self.greedy) and (not double_q_learning): 
            if self.rng.uniform() < self.epsilon:
                all_actions = np.arange(self.num_actions)
                non_greedy_actions = np.setdiff1d(all_actions, actions.cpu().numpy(), assume_unique=False)
                actions = torch.tensor([self.rng.choice(non_greedy_actions)], dtype=torch.int32)

        return actions

    def train_mode_actions(self, reward, observation, terminal):
        ''' actions to take when agent in training mode i.e. adding to replay buffer, cloning target q network and training q network'''
        self.steps += 1
        self.add_to_replay_buffer(reward, observation, terminal)
        if self.clone_q_net_condition(): self.clone_q()
        if self.training_condition(): self.update_q()

    def add_to_replay_buffer(self, reward, observation, terminal):
        ''' add step sequence to buffer '''
        terminal_state = torch.tensor([terminal], dtype=torch.bool).to(self.device)
        values = (self.prev_state, self.prev_action, reward, observation, terminal_state)
        self.buffer.append(values)
        if self.buffer_filled:
            self.buffer.pop(0)
        else:
            if len(self.buffer) >= self.buffer_max_length: 
                self.buffer_filled = True
        
    def training_condition(self):
        bool_step_multiple = (self.steps % self.train_steps == 0)
        return bool_step_multiple and self.buffer_filled

    def clone_q_net_condition(self):
        bool_step_multiple = (self.steps % self.clone_steps == 0)
        return bool_step_multiple and self.buffer_filled

    def clone_q(self):        
        self.target_q.load_state_dict(self.q.state_dict())

    def update_q(self): 
        ''' train the self.q neural network by drawing from replay buffer '''
        dataset = BufferDataset(self.buffer)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
        for _ in range(self.n_epochs): # n_epochs controls how many epochs to train
            epoch_loss = []
            n_batches_processed = 0       
            for current_states, actions, rewards, next_states, terminal_state in dataloader:
                batch_loss = self.train_batch(current_states, actions, rewards, next_states, terminal_state)
                epoch_loss.append(batch_loss.detach().numpy())
                n_batches_processed += 1
                if n_batches_processed == self.n_batches: break # n_batches controls how many batches to train

    def train_batch(self, current_states, actions, rewards, next_states, terminal_state):
        ''' train self.q neural network given a batch '''

        self.optimizer.zero_grad()
        current_states, actions, rewards, next_states, terminal_state = self.to_device([current_states, actions, rewards, next_states, terminal_state])

        # compute targets = reward + gamma * target_q(next_state, action) where action = max(q(next_state)) i.e. double Q-learning
        next_actions = self.get_action(next_states, batched=True, double_q_learning=True)
        row_indices = np.arange(next_actions.shape[0])
        with torch.no_grad(): next_state_q = self.target_q(next_states)[row_indices, next_actions.to(torch.int64)].unsqueeze(-1)
        not_terminal = torch.logical_not(terminal_state)
        targets = (rewards + self.discount * next_state_q * not_terminal).squeeze()

        # compute current state q value = Q(current_state)[action]
        row_indices = np.arange(actions.shape[0])
        current_state_q = self.q(current_states)[row_indices, actions.squeeze().to(torch.int64)]
        loss = self.loss_fn(current_state_q, targets)
        loss.backward()
        if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.q.parameters(), 1)
        self.optimizer.step()
        
        return loss
    
    def to_device(self, list):
        device_var = []
        for var in list:
            var = var.to(self.device)
            device_var.append(var)
        return device_var