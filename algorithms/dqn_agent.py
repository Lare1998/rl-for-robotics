import gym
import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the neural network for the Q-function
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model. """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    Interacts with and learns from the environment using a Deep Q-Network.
    """
    def __init__(self, state_size, action_size, seed, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4):
        """
        Initializes a DQNAgent.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            seed (int): Random seed.
            buffer_size (int): Replay buffer size.
            batch_size (int): Minibatch size.
            gamma (float): Discount factor.
            tau (float): For soft update of target parameters.
            lr (float): Learning rate.
            update_every (int): How often to update the network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, self.device)
        # Initialize time step (for updating every 'update_every' steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use it to learn.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every 'update_every' time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state (array_like): Current state.
            eps (float): Epsilon, for epsilon-greedy action selection.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Q_targets = r + γ * max(Q_targets(next_state, :))
        where max(Q_targets(next_state, :)) is computed from target network
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """
        Initialize a ReplayBuffer object.

        Args:
            action_size (int): Dimension of each action.
            buffer_size (int): Maximum size of buffer.
            batch_size (int): Size of each training batch.
            seed (int): Random seed.
            device (torch.device): Device to store tensors on (cpu or cuda).
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        e = dict(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e["state"] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e["action"] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e["reward"] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e["next_state"] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e["done"] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)

if __name__ == "__main__":
    # Example usage with CartPole environment
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    seed = 0

    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed)

    # Simulate training loop
    n_episodes = 2
    max_t = 100
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start

    scores = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        eps = max(eps_end, eps_decay * eps) # decrease epsilon

        print(f"Episode {i_episode}\tAverage Score: {np.mean(scores):.2f}")

    env.close()
