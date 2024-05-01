import mlagents
import numpy as np
from DQNAgent import DQNAgent
from mlagents_envs.environment import UnityEnvironment as UE 
from mlagents_envs.environment import ActionTuple

env = UE(file_name="BasicEnv", seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

num_actions = spec.action_spec.discrete_branches[0]
num_states = sum([obs.shape[0] for obs in spec.observation_specs])
print(num_states)

# Hyperparameters
learning_rate = 0.01
n_episodes = 100000
batch_size = 64
memory_threshold = 64
start_epsilon = 1.0
epsilon_decay = (start_epsilon / (n_episodes)) * 5 # reduce the exploration over time
final_epsilon = 0.05

# Initialize a fixed number of QAgents based on the maximum number expected to be active
max_agents = 1  # Adjust this number based on the maximum number of agents ever observed at once
agents_pool = [DQNAgent(num_states, num_actions) for _ in range(max_agents)]
agent_id_map = {}

def get_available_agent():
    if agents_pool:
        random_agent = np.random.randint(0, max_agents)
        return agents_pool[random_agent]
    return DQNAgent(num_states, num_actions)

last_reward_checkpoint = 0 # Helper var for printing running episode rewards

for episode in range(n_episodes):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # Reset or reassign agent IDs to existing QAgent instances
    current_ids = list(decision_steps.agent_id)
    agent_id_map = {agent_id: get_available_agent() for agent_id in current_ids}

    done = {agent_id: False for agent_id in current_ids}
    episode_rewards = {agent_id: 0 for agent_id in current_ids}

    print(f"Starting episode {episode + 1}")
    
    episode_count = 0
    updateTargetNetwork = 1000

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
                action = agent.act(state)
                action_tuple = ActionTuple(discrete=np.array([[action]]))
                env.set_action_for_agent(behavior_name, agent_id, action_tuple)

        env.step()
        
        # Get new steps after action
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Update total rewards and train agent
        for agent_id in current_ids:
            if agent_id in terminal_steps:
                
                next_state = np.concatenate([obs.flatten() for obs in terminal_steps[agent_id].obs])
                reward = terminal_steps[agent_id].reward
                episode_rewards[agent_id] = episode_rewards.get(agent_id, 0) + reward
                
                agent_id_map[agent_id].remember(state, action, reward, next_state, 1)
                done[agent_id] = True
                episode_count += 1

            elif agent_id in decision_steps:
                next_state = np.concatenate([obs.flatten() for obs in decision_steps[agent_id].obs])
                reward = decision_steps[agent_id].reward
                episode_rewards[agent_id] = episode_rewards.get(agent_id, 0) + reward
                
                agent_id_map[agent_id].remember(state, action, reward, next_state, 0)
                
        # Periodically train agent from memory
        if len(agent.memory_bank) > batch_size and len(agent.memory_bank) > memory_threshold and last_reward_checkpoint % 20 == 0:
            print(len(agent.memory_bank))
            agent.replay(batch_size)
            agent.target_train()

        # Print the running episode rewards
        total_rewards = sum(episode_rewards.values())
        reward_checkpoint = total_rewards // 10 * 10
        if reward_checkpoint > last_reward_checkpoint:
            print(f"Rewards so far: {total_rewards}")
            last_reward_checkpoint = reward_checkpoint

        current_ids = list(decision_steps.agent_id) + list(terminal_steps.agent_id)

    print(f"End of episode {episode + 1}, rewards: {sum(episode_rewards.values())}")
    agents_pool.extend(agent_id_map.values())
    agent_id_map.clear()

env.close()