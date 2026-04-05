import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Hyperparameters ---
ENV_NAME = "FrozenLake-v1"
LEARNING_RATE = 0.8
GAMMA = 0.95
MAX_EPISODES = 5000
MAX_STEPS = 100

# Exploration parameters for Epsilon-Greedy strategy
EPSILON = 1.0       # Initial exploration rate
MAX_EPSILON = 1.0   # Maximum exploration rate
MIN_EPSILON = 0.01  # Minimum exploration rate
DECAY_RATE = 0.005  # Exponential decay rate

def create_env(render_mode=None):
    """
    Initializes and returns the FrozenLake environment.
    Note: Rewards are sparse (+1 for reaching the goal, 0 otherwise).
    """
    # is_slippery=True adds stochasticity to the environment
    return gym.make(ENV_NAME, is_slippery=True, render_mode=render_mode)


if __name__ == "__main__":
    env = create_env()
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    
    # Initialize Q-table to zeros
    # Shape: (16, 4) because there are 16 states and 4 actions
    q_table = np.zeros((state_dim, action_dim))
    
    rewards_history = []
    avg_rewards = []
    
    print(f"Starting Q-Learning on {ENV_NAME}...")

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        done = False
        current_ep_reward = 0
        
        for step in range(MAX_STEPS):
            # 1. Choose action via Epsilon-greedy strategy
            exp_exp_tradeoff = np.random.uniform(0, 1)
            
            if exp_exp_tradeoff > EPSILON:
                # Exploit: choose the action with the highest Q-value for this state
                action = np.argmax(q_table[state, :])
            else:
                # Explore: choose a random action
                action = env.action_space.sample()

            # 2. Take step in the environment
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Update Q-table using the Bellman equation
            # Q(s,a) = Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            temporal_difference = reward + GAMMA * np.max(q_table[new_state, :]) - q_table[state, action]
            q_table[state, action] = q_table[state, action] + LEARNING_RATE * temporal_difference
            
            # 4. Update state and accumulate reward
            state = new_state
            current_ep_reward += reward
            
            if done:
                break
                
        # Decay epsilon to progressively explore less and exploit more
        EPSILON = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
        
        # Tracking metrics
        rewards_history.append(current_ep_reward)
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards.append(avg_reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode} \t Avg Reward: {avg_reward:.2f} \t Epsilon: {EPSILON:.3f}")

        # FrozenLake-v1 is officially considered solved at ~0.74 due to slippery mechanics
        if avg_reward >= 0.74:
            print(f"Solved! Average reward: {avg_reward:.2f} at Episode {episode}")
            break

    env.close()

    # Save training results plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label='Episode Reward', alpha=0.3)
    plt.plot(avg_rewards, label='Average Reward (100 eps)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Tabular Q-Learning - {ENV_NAME}')
    plt.legend()
    plt.savefig('frozenlake_qlearning_results.png')
    print("Results saved to frozenlake_qlearning_results.png")

    # ==========================
    # Test the final trained agent
    # ==========================
    test_env = gym.make(ENV_NAME, render_mode="human")
    state, _ = test_env.reset()
    total_reward = 0
    done = False
    
    print("\n--- Running Trained Agent ---")
    
    while not done:
        # For testing, we only exploit (epsilon = 0)
        action = np.argmax(q_table[state, :])
        state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        test_env.render()
        time.sleep(0.5) # Slow down to watch the agent

    print(f"Final Test Reward: {total_reward}")
    test_env.close()
