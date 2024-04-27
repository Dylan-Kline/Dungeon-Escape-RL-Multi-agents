import mlagents
from mlagents_envs.environment import UnityEnvironment as UE 

env = UE(file_name="UnityEnvironment", seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]
print(f"Number of actions: ", spec.action_spec)

print("Number of observations: ", len(spec.observation_specs))
print("Observation vector shape: ", spec.observation_specs)

# if spec.:
#     print("The action is discrete")

for episode in range(3):
    env.reset()
    
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    tracked_agent = -1 # -1 indicates not yet tracking
    done = False # For the tracked_agent
    episode_rewards = 0 # For the tracked agent
    
    
    