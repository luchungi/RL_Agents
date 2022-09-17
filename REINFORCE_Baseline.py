import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class REINFORCE_Agent:
    '''
    Implementation of REINFORCE with state dependent baseline using neural network
    Action output can be discrete or continuous based on stochastic policy
    Categorial distribution used for discrete actions 
    Multivariate Gaussian for continuous with diagonal cov matrix
    Causality variable switches full episode rewards to rewards to go in policy gradient
    Baseline variable switches on/off inclusion of the baseline in policy gradient
    '''

    def agent_init(self, agent_init_info):
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info['num_actions']
        self.obs_size = agent_init_info['obs_size']        
        self.step_size = agent_init_info['step_size']
        self.discount = agent_init_info['discount']
        self.buffer_max_length = agent_init_info['buffer_max_length'] # number of episodes
        self.batch_size = agent_init_info['batch_size'] # number of steps
        self.causality = agent_init_info['causality']
        self.baseline = agent_init_info['baseline']
        self.standardise = agent_init_info['standardise']
        self.discrete = agent_init_info['discrete']
        self.device = agent_init_info['device']
        self.rng = np.random.default_rng()      
        self.training_mode = True

        try:
            self.p = agent_init_info['policy_model']
        except:            
            self.p = nn.Sequential(
                nn.Linear(self.obs_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.num_actions)
            )
        self.p.to(self.device)

        # log std of distribution used for stochastic continuous action with zero covariance between actions
        if not self.discrete: self.logstd = nn.Parameter(torch.ones(self.num_actions, dtype=torch.float32)) 
        
        self.p_optimizer = torch.optim.Adam(list(self.p.parameters()) +list(self.logstd), lr=self.step_size)

        try:
            self.q = agent_init_info['baseline_model']
        except:
            self.q = nn.Sequential(
                nn.Linear(self.obs_size + self.num_actions, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        self.q.to(self.device)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=self.step_size)
        self.loss_fn = nn.MSELoss()

        self.episode_obs = []
        self.episode_actions = []
        self.episode_rewards = []
        self.buffer = []
        self.steps = 0

    def agent_start(self, observation):
        action = self.get_action(observation).numpy().squeeze()
        if self.training_mode:
            self.episode_obs.append(observation)
            self.episode_actions.append(action)
        return action

    def agent_step(self, reward, observation):
        action = self.get_action(observation).numpy().squeeze()
        if self.training_mode:
            self.episode_obs.append(observation)
            self.episode_actions.append(action)
            self.episode_rewards.append(reward)
        return action

    def agent_end(self, reward):
        if self.training_mode:
            self.episode_rewards.append(reward)
            self.add_to_buffer()
            if self.steps > self.batch_size: # if new accumulated episodes have total steps > batch size then train
                self.train()
                self.steps = 0 # reset counter for next training

    def get_action(self, observation, return_dist=False, batched=False):
        if not batched: observation = torch.tensor(observation[np.newaxis,...], dtype=torch.float32)
        observation = observation.to(self.device)
        if self.discrete:
            logits = self.p(observation)
            action_dist = torch.distributions.Categorical(logits=logits)
        else:
            # multivariate normal distribution with zero covariance used for stochastic continuous action
            batch_mean = self.p(observation)
            scale_tril = torch.diag(torch.exp(self.logstd)).to(self.device)
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_dist = torch.distributions.MultivariateNormal(batch_mean, 
                                                        scale_tril=batch_scale_tril)
        if return_dist:
            return action_dist
        else:
            return action_dist.sample().cpu()

    def add_to_buffer(self):
        self.steps += len(self.episode_obs)       
        self.buffer.append((self.episode_obs, self.episode_actions, self.episode_rewards))
        if len(self.buffer) > self.buffer_max_length:
            self.buffer.pop(0)
        # reset episode gradient and reward lists
        self.episode_obs = []
        self.episode_actions = []
        self.episode_rewards = []

    def sample_buffer(self, batch_size):
        '''
        sampling from the latest episode backwards until total number of sampled steps
        is sufficient for the required batch size
        i.e. no shuffling of buffer
        '''
        steps = 0
        paths = []
        index = -1
        while steps < batch_size: # batch_size = # of steps
            path = self.buffer[index]
            paths.append(path)
            steps += len(path[0])
            index -= 1

        obs = np.concatenate([path[0] for path in paths])
        actions = np.concatenate([path[1] for path in paths])
        rewards = [path[2] for path in paths] # do no concatenate as discount function takes in list of episodes of rewards
        return obs, actions, rewards

    def train(self):
        obs, actions, rewards = self.sample_buffer(self.batch_size)
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        obs, actions = self.to_device([obs, actions])
        
        q_values = torch.tensor(self.calculate_q_values(rewards), dtype=torch.float32)
        advantages = self.calculate_advantages(q_values, obs).to(self.device)

        action_dist = self.get_action(obs, return_dist=True, batched=True)
        log_prob = action_dist.log_prob(actions)
        assert log_prob.shape == advantages.shape
        
        loss = -torch.sum(advantages * log_prob) # policy gradient for stochastic policy
        self.p_optimizer.zero_grad()
        loss.backward()
        self.p_optimizer.step()

        if self.baseline: self.train_baseline(q_values, obs)
        del obs, actions, q_values, advantages, action_dist, log_prob, loss

    def train_baseline(self, targets, obs):  
        pred = self.q(obs).squeeze()
        loss = self.loss_fn(pred, targets.to(self.device))
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        del loss, pred

    def discounted_return(self, rewards):
        """
            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T
            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        n_steps = len(rewards)
        discounts = np.logspace(0, n_steps-1, num=n_steps, base=self.discount)
        discounted_returns = np.array(discounts * rewards)
        list_of_discounted_returns = [discounted_returns.sum()] * n_steps

        return np.array(list_of_discounted_returns)

    def discounted_cumsum(self, rewards):
        """
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        n_steps = len(rewards)
        discounts = np.logspace(0, n_steps-1, num=n_steps, base=self.discount)
        list_of_discounted_cumsums = [np.sum(rewards[i:] * discounts[:n_steps-i]) for i in range(n_steps)]

        return np.array(list_of_discounted_cumsums)

    def calculate_q_values(self, rewards): 
        # Monte Carlo estimation of the Q function
  
        if self.causality:
            # Case 1: reward-to-go PG
            # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
            q_values = np.concatenate([self.discounted_cumsum(episode_rewards) for episode_rewards in rewards])
        else:
            # Case 1: trajectory-based PG
            # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
            q_values = np.concatenate([self.discounted_return(episode_rewards) for episode_rewards in rewards])
        return q_values
    
    def calculate_advantages(self, q_values, obs=None):
        # Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        
        if self.baseline:
            values_unnormalized = self.q(obs).squeeze().cpu()
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