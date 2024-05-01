import mlagents
import numpy as np
from plot import plot_data
from DQNAgent import DQNAgent
from mlagents_envs.environment import UnityEnvironment as UE 
from mlagents_envs.environment import ActionTuple

def main():
    env = UE(file_name="BasicEnv", seed=1, side_channels=[])
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]
    print(f"Number of actions: ", spec.action_spec)

    num_actions = spec.action_spec.discrete_branches[0]
    num_states = sum([obs.shape[0] for obs in spec.observation_specs])

    # Hyperparameters
    n_episodes = 1000
    min_samples = 128
    batch_size = 64

    agent = DQNAgent(num_states, num_actions)

    rewards_over_episodes = []

    for episode in range(n_episodes):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        episode_rewards = 0
        done = False
        print(f"Starting episode {episode + 1}")
        
        while not done:
            for agent_id in decision_steps:
                state = np.concatenate([obs.flatten() for obs in decision_steps[agent_id].obs])
                action = agent.act(state)
                env.set_action_for_agent(behavior_name, agent_id, ActionTuple(discrete=np.array([[action]])))
            
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            for agent_id in terminal_steps:
                next_state = np.concatenate([obs.flatten() for obs in terminal_steps[agent_id].obs])
                reward = terminal_steps[agent_id].reward
                done = True
                episode_rewards += reward
                agent.remember(state, action, reward, next_state, done)
                
            for agent_id in decision_steps:
                next_state = np.concatenate([obs.flatten() for obs in decision_steps[agent_id].obs])
                reward = decision_steps[agent_id].reward
                episode_rewards += reward
                agent.remember(state, action, reward, next_state, done)

            if len(agent.memory_bank) > min_samples and episode % 20 == 0:
                agent.replay(batch_size)
                agent.target_train()
        
        rewards_over_episodes.append(episode_rewards)
        print(f"End of episode {episode + 1}, rewards: {episode_rewards}")

    plot_data(rewards_over_episodes)
    env.close()

main()