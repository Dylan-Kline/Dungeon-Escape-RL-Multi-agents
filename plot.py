from matplotlib import pyplot as plt

def plot_data(data):
    
    # unpack data
    rewards, episodes = zip(*data)
    
    # plot data
    fig = plt.figure(figsize=(15, 10))
    plt.plot(episodes, rewards)
    plt.title("Reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.tight_layout()
    plt.show()