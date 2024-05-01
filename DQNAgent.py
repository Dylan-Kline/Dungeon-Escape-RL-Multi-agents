from collections import deque
import numpy as np
from mlp import MultilayerPerceptron

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = MultilayerPerceptron()
        self.model.initialize_mlp(num_features=state_size, num_actions=action_size)
        # Deep mind "hack"
        self.target_model = MultilayerPerceptron()
        self.target_model.initialize_mlp(num_features=state_size, num_actions=action_size)
        
        # Agent hyperparameters
        self.discount_factor = 0.95 # Importance of future rewards
        self.learning_rate = 0.001
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory_size = 300
        self.memory_bank = deque(maxlen=self.memory_size)

    def remember(self, state, action, reward, next_state, terminal):
        if len(self.memory_bank) >= self.memory_size:
            self.memory_bank.popleft()
        self.memory_bank.append((state, action, reward, next_state, terminal))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = state.reshape(1, -1)  # Reshape for the network
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Returns the action with the highest Q-value

    # def replay(self, batch_size):
        
    #     if len(self.memory_bank) < batch_size:
    #         return
        
    #     minibatch_indices = np.random.choice(len(self.memory_bank), batch_size, replace=False)
    #     minibatch = [self.memory_bank[i] for i in minibatch_indices]
    #     for state, action, reward, next_state, terminal in minibatch:
    #         state = state.reshape(1, -1)
    #         next_state = next_state.reshape(1, -1)
    #         target = reward
            
    #         if not terminal:
    #             target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
                
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, self.action_size)
        
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
            
    def target_train(self):
        layers = self.model.get_layers()
        target_layers = self.target_model.get_layers()
        for i in range(len(target_layers)):
            target_layers[i] = layers[i]
        self.target_model.set_layers(target_layers)

    def load(self, name):
        self.model = self.model.load(name)

    def save(self, name):
        self.model.save(name)
        
    def replay(self, batch_size):
        if len(self.memory_bank) < batch_size:
            return

        minibatch_indices = np.random.choice(len(self.memory_bank), batch_size, replace=False)
        minibatch = [self.memory_bank[i] for i in minibatch_indices]

        states = np.array([item[0] for item in minibatch]).reshape(batch_size, -1)
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.array([item[3] for item in minibatch]).reshape(batch_size, -1)
        terminals = np.array([item[4] for item in minibatch])

        # Predict current Q-values and next Q-values for all states and next_states
        current_q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        # Compute the target Q-values
        targets = rewards + self.discount_factor * np.amax(next_q_values, axis=1) * (1 - terminals)

        # Set the targets for the actions taken
        target_f = current_q_values.copy()
        target_f[np.arange(batch_size), actions] = targets

        # Fit the model on the states and the target Q-values
        self.model.fit(states, target_f, self.action_size)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
