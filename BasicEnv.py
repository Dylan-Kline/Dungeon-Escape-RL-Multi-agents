import mlagents
import numpy as np
from BasicQLearningAgent import QAgent
from mlagents_envs.environment import UnityEnvironment as UE 
from mlagents_envs.environment import ActionTuple

env = UE(file_name="pyrenv/UnityEnvironment", seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]
print(f"Number of actions: ", spec.action_spec)

num_actions = spec.action_spec.discrete_branches[0]
num_states = sum([obs.shape[0] for obs in spec.observation_specs])

# hyperparameters
learning_rate = 0.01
n_episodes = 10000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes)  # reduce the exploration over time
final_epsilon = 0.03

agent = QAgent(learning_rate = learning_rate,
               initial_epsilon = start_epsilon,
               epsilon_decay = epsilon_decay,
               final_epsilon = final_epsilon,
               action_num = num_actions)

for episode in range(n_episodes):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    done = False
    tracked_agent = -1
    episode_rewards = 0
    print(f"Starting episode {episode + 1}")
    
    
    while not done:
        
        if tracked_agent == -1 and len(decision_steps) >= 1:
            tracked_agent = decision_steps.agent_id[0]
            
        if tracked_agent in terminal_steps:
            
            state = np.concatenate([obs.flatten() for obs in terminal_steps[tracked_agent].obs])
            reward = terminal_steps[tracked_agent].reward
            episode_rewards += reward
            agent.update(state, None, reward, True, None)  # Update for terminal state
            done = True
            print(f"Episode {episode + 1} ended with terminal state")
            
        elif tracked_agent in decision_steps:
            
            state = np.concatenate([obs.flatten() for obs in decision_steps[tracked_agent].obs])
            action = agent.get_action(state)
            action_tuple = ActionTuple(discrete=np.array([[action]]))
            
            # Apply action and get new state
            env.set_action_for_agent(behavior_name, tracked_agent, action_tuple)
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if tracked_agent in terminal_steps:
                
                next_state = np.concatenate([obs.flatten() for obs in terminal_steps[tracked_agent].obs])
                reward = terminal_steps[tracked_agent].reward
                episode_rewards += reward
                done = True
                print(f"Episode {episode + 1} transitioning to terminal state")
                
            elif tracked_agent in decision_steps:
                
                next_state = np.concatenate([obs.flatten() for obs in decision_steps[tracked_agent].obs])
                reward = decision_steps[tracked_agent].reward
                episode_rewards += reward
            
            # Update agent
            agent.update(state, action, reward, done, next_state)
    
    print(f"End of episode {episode + 1}, rewards: {episode_rewards}")
    agent.decay_epsilon()
    
env.close()