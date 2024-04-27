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
total_episodes = 100

agents = {}  # Dictionary to store DQNAgent instances by agent_id

for episode in range(total_episodes):
    env.reset()
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        actions = np.empty((0, 1), int)

        for agent_id, decision_step in decision_steps.items():
            if agent_id not in agents:
                agents[agent_id] = DQNAgent(state_size=num_states, action_size=num_actions)  # Initialize new agent if not in dict

            state = np.concatenate([obs.flatten() for obs in decision_step.obs])
            action = agents[agent_id].choose_action(state)
            print(f"Action chosen by agent {agent_id}: {action}")
            print(f"Current Actions for each agent {actions}")
            actions = np.vstack((actions, [action]))
            
        # Create an ActionTuple and set actions for all agents at once
        action_tuple = ActionTuple(discrete=actions)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        # Handle training and updating for both decision and terminal steps
        for agent_id, step in decision_steps.items():
            reward = step.reward
            next_state = np.concatenate(step.obs) if step in decision_steps else None
            done = agent_id in terminal_steps
            agents[agent_id].train(state, action, reward, next_state, done)

        if all(agent_id in terminal_steps for agent_id in decision_steps.agent_id):
            break  # End episode if all agents are done

    print(f'Episode: {episode + 1} completed.')
    if episode % 10 == 0:
        for agent_id, agent in agents.items():
            agent.save(f"dqn_model_{agent_id}.json")  # Save model for each agent

env.close()