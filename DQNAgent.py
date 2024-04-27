import numpy as np
from mlp import MultilayerPerceptron

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = MultilayerPerceptron()
        self.model.initialize_mlp(num_features=state_size, num_actions=action_size)
        self.epsilon = .1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = state.reshape(1, -1)  # Reshape for the network
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Returns the action with the highest Q-value

    def train(self, state, action, reward, next_state, done, num_actions):
        target = reward
        if not done:
            next_state = next_state.reshape(1, -1)  # Reshape for the network
            target = reward + 0.99 * np.amax(self.model.predict(next_state)[0])
        
        target_f = self.model.predict(state.reshape(1, -1))
        target_f[0][action] = target
        
        self.model.fit(state.reshape(1, -1), target_f, num_actions)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = self.model.load(name)

    def save(self, name):
        self.model.save(name)
