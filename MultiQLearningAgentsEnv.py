import mlagents
import numpy as np
from BasicQLearningAgent import QAgent
from mlagents_envs.environment import UnityEnvironment as UE 
from mlagents_envs.environment import ActionTuple

env = UE(file_name="DungeonEscapeEnv", seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

num_actions = spec.action_spec.discrete_branches[0]
num_states = sum([obs.shape[0] for obs in spec.observation_specs])

# Hyperparameters
learning_rate = 0.01
n_episodes = 10000
start_epsilon = 1.0
epsilon_decay = start_epsilon / n_episodes
final_epsilon = 0.03

# Initialize a fixed number of QAgents based on the maximum number expected to be active
max_agents = 36  # Adjust this number based on the maximum number of agents ever observed at once
agents_pool = [QAgent(learning_rate, start_epsilon, epsilon_decay, final_epsilon, num_actions) for _ in range(max_agents)]
agent_id_map = {}

def get_available_agent():
    if agents_pool:
        return agents_pool.pop(0)
    return QAgent(learning_rate, start_epsilon, epsilon_decay, final_epsilon, num_actions)

for episode in range(n_episodes):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # Reset or reassign agent IDs to existing QAgent instances
    current_ids = list(decision_steps.agent_id)
    print(current_ids)
    agent_id_map = {agent_id: get_available_agent() for agent_id in current_ids}

    done = {agent_id: False for agent_id in current_ids}
    episode_rewards = {agent_id: 0 for agent_id in current_ids}

    print(f"Starting episode {episode + 1}")

    while not all(done.values()):

        # Grab new agent ids
        done = {agent_id: False for agent_id in current_ids}

        for agent_id in current_ids:

            # Assign existing QLearning "Brains" to new agents
            if agent_id not in agent_id_map:
                agent_id_map[agent_id] = get_available_agent()
                
            if agent_id in decision_steps and not done[agent_id]:
                state = np.concatenate([obs.flatten() for obs in decision_steps[agent_id].obs])
                agent = agent_id_map[agent_id]
                action = agent.get_action(state)
                action_tuple = ActionTuple(discrete=np.array([[action]]))
                env.set_action_for_agent(behavior_name, agent_id, action_tuple)

        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Update agents based on new decision and terminal steps
        for agent_id in current_ids:
            if agent_id in terminal_steps:
                next_state = np.concatenate([obs.flatten() for obs in terminal_steps[agent_id].obs])
                reward = terminal_steps[agent_id].reward 
                episode_rewards[agent_id] = episode_rewards.get(agent_id, 0) + reward
                agent_id_map[agent_id].update(state, action, reward, True, next_state)
                done[agent_id] = True
                agent.decay_epsilon()  # Decay epsilon after each episode or terminal step

            elif agent_id in decision_steps:
                next_state = np.concatenate([obs.flatten() for obs in decision_steps[agent_id].obs])
                reward = decision_steps[agent_id].reward
                episode_rewards[agent_id] = episode_rewards.get(agent_id, 0) + reward
                agent_id_map[agent_id].update(state, action, reward, False, next_state)

        if sum(episode_rewards.values()) > 0:
            print(f"Rewards so far: {sum(episode_rewards.values())}")
        current_ids = list(decision_steps.agent_id) + list(terminal_steps.agent_id)

    print(f"End of episode {episode + 1}, rewards: {sum(episode_rewards.values())}")
    agents_pool.extend(agent_id_map.values())
    agent_id_map.clear()

env.close()