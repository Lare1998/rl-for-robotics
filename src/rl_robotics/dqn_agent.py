
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple

class DQNAgent:
    """Deep Q-Network (DQN) agent for reinforcement learning in robotics."""
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) # Replay buffer
        self.learning_rate = learning_rate
        self.gamma = gamma    # Discount factor
        self.epsilon = epsilon  # Exploration-exploitation trade-off
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Builds the neural network for the DQN agent."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Stores experiences in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Chooses an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size: int):
        """Trains the model using experiences sampled from the replay buffer."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name: str):
        self.model.load_weights(name)

    def save(self, name: str):
        self.model.save_weights(name)

class RobotEnvironment:
    """Simulates a simple robot environment for testing the DQN agent."""
    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        self.robot_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.state_space = grid_size * grid_size
        self.action_space = 4 # Up, Down, Left, Right

    def reset(self) -> np.ndarray:
        self.robot_pos = (0, 0)
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.state_space)
        idx = self.robot_pos[0] * self.grid_size + self.robot_pos[1]
        state[idx] = 1
        return state.reshape(1, -1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        x, y = self.robot_pos
        if action == 0: # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3: # Right
            y = min(self.grid_size - 1, y + 1)
        
        self.robot_pos = (x, y)
        new_state = self._get_state()
        reward = -0.1 # Penalty for each step
        done = False
        info = {}

        if self.robot_pos == self.goal_pos:
            reward = 10
            done = True
        
        return new_state, reward, done, info

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ". ", dtype=str)
        grid[self.robot_pos[0], self.robot_pos[1]] = "R "
        grid[self.goal_pos[0], self.goal_pos[1]] = "G "
        for row in grid:
            print("".join(row))
        print("\n")

if __name__ == "__main__":
    env = RobotEnvironment()
    state_size = env.state_space
    action_size = env.action_space
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    episodes = 100
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        for time_step in range(500): # Max steps per episode
            # env.render() # Uncomment to visualize
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    print("Training complete.")

# Update on 2023-01-02 00:00:00
# Update on 2023-01-02 00:00:00
# Update on 2023-01-03 00:00:00
# Update on 2023-01-04 00:00:00
# Update on 2023-01-05 00:00:00
# Update on 2023-01-05 00:00:00
# Update on 2023-01-06 00:00:00
# Update on 2023-01-06 00:00:00
# Update on 2023-01-10 00:00:00
# Update on 2023-01-10 00:00:00
# Update on 2023-01-10 00:00:00
# Update on 2023-01-11 00:00:00
# Update on 2023-01-11 00:00:00
# Update on 2023-01-12 00:00:00
# Update on 2023-01-12 00:00:00
# Update on 2023-01-13 00:00:00
# Update on 2023-01-16 00:00:00
# Update on 2023-01-17 00:00:00
# Update on 2023-01-19 00:00:00
# Update on 2023-01-19 00:00:00
# Update on 2023-01-23 00:00:00
# Update on 2023-01-26 00:00:00
# Update on 2023-01-30 00:00:00
# Update on 2023-02-03 00:00:00
# Update on 2023-02-06 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-08 00:00:00
# Update on 2023-02-13 00:00:00
# Update on 2023-02-14 00:00:00
# Update on 2023-02-14 00:00:00
# Update on 2023-02-14 00:00:00
# Update on 2023-02-15 00:00:00
# Update on 2023-02-15 00:00:00
# Update on 2023-02-16 00:00:00
# Update on 2023-02-22 00:00:00
# Update on 2023-02-27 00:00:00
# Update on 2023-02-28 00:00:00
# Update on 2023-02-28 00:00:00
# Update on 2023-03-01 00:00:00
# Update on 2023-03-01 00:00:00
# Update on 2023-03-03 00:00:00
# Update on 2023-03-08 00:00:00
# Update on 2023-03-08 00:00:00
# Update on 2023-03-09 00:00:00
# Update on 2023-03-09 00:00:00
# Update on 2023-03-09 00:00:00