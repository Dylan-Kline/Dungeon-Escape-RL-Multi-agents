import mlagents
import numpy as np
from DQNAgent import DQNAgent
from mlagents_envs.environment import UnityEnvironment as UE 
from mlagents_envs.environment import ActionTuple

env = UE(file_name="UnityEnvironment", seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]
print(f"Number of actions: ", spec.action_spec)

num_actions = spec.action_spec.discrete_branches[0]
num_states = sum([obs.shape[0] for obs in spec.observation_specs])
total_episodes = 1000

agents = {}  # Dictionary to store DQNAgent instances by agent_id
model = DQNAgent(state_size=num_states, action_size=num_actions)

for episode in range(total_episodes):
    env.reset()
    episode_rewards = 0
    done = False # Flag for whether an episode is finsihed
    
    while not done:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        actions = np.empty((0, 1), int)

        for agent_id, decision_step in decision_steps.items():
            if agent_id not in agents:
                agents[agent_id] = model  # Initialize new agent if not in dict

            state = np.concatenate([obs.flatten() for obs in decision_step.obs])
            action = agents[agent_id].choose_action(state)
            #print(f"Action chosen by agent {agent_id}: {action}")
            actions = np.vstack((actions, [action]))
            
        # Create an ActionTuple and set actions for all agents at once
        action_tuple = ActionTuple(discrete=actions)
        env.set_actions(behavior_name, action_tuple)
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        print(list(decision_steps))

        # Handle training and updating for both decision and terminal steps
        for agent_id, step in decision_steps.items():
            reward = step.reward
            #print(reward)
            next_state = np.concatenate([obs.flatten() for obs in step.obs])
            done = agent_id in terminal_steps
            agents[agent_id].train(state, action, reward, next_state, done, num_actions)
            episode_rewards += reward
            
        # Additional handling for agents in terminal steps
        for agent_id, step in terminal_steps.items():
            reward = step.reward
            episode_rewards += reward

        if all(agent_id in terminal_steps for agent_id in decision_steps.agent_id):
            done = True  # End episode if all agents are done
            print(episode_rewards)
            break

    print(f'Episode: {episode + 1} completed. Rewards: {episode_rewards}')
    if episode % 10 == 0:
        count = 0
        model.save(f"dqn_model_{count}.json")  # Save model for each agent

env.close()