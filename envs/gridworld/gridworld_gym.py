import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldGym(gym.Env):
    """
    Gymnasium-compatible version of the user's basic GridWorld.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, width=4, height=4, render_mode=None):
        super().__init__()
        self.width = width
        self.height = height
        
        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 2D coordinate [row, col]
        # We use float32 Box because typical deep RL algorithms (like those in firedup) 
        # expect continuous observation tensors by default.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([float(self.height - 1), float(self.width - 1)], dtype=np.float32),
            dtype=np.float32
        )
        
        self.agent_pos = [0, 0]  # Start at top-left
        self.goal_pos = [self.height - 1, self.width - 1]  # Goal at bottom-right
        self.render_mode = render_mode
        
        # Q-table: stores Q-values for each state-action pair (height, width, num_actions)
        self.q_table = np.zeros((self.height, self.width, self.action_space.n))
        
        # Value table: stores the expected return for each state (height, width)
        self.v_table = np.zeros((self.height, self.width))

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.agent_pos = [0, 0]
        
        # Return observation as numpy float32 array and empty info dict
        obs = np.array(self.agent_pos, dtype=np.float32)
        info = {}
        
        self.render()
            
        return obs, info

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0 and self.agent_pos[0] > 0: # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.height - 1: # Down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0: # Left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.width - 1: # Right
            self.agent_pos[1] += 1

        # Check if goal reached
        terminated = (self.agent_pos == self.goal_pos)
        truncated = False  # The environment doesn't have an internal step limit right now
        
        reward = 10.0 if terminated else -1.0
        
        obs = np.array(self.agent_pos, dtype=np.float32)
        info = {}
        
        self.render()
            
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            grid = np.full((self.height, self.width), ".", dtype=str)
            grid[self.goal_pos[0], self.goal_pos[1]] = "G"
            grid[self.agent_pos[0], self.agent_pos[1]] = "A"
            print("\n" + "\n".join([" ".join(row) for row in grid]))
            print("-" * (self.width * 2))

if __name__ == "__main__":
    # Smoke test
    env = GridWorldGym(render_mode="human")
    print("=== INITIAL STATE ===")
    obs, info = env.reset()
    print(f"Observation: {obs}")
    
    print("\n=== STEP DOWN ===")
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"Observation: {obs}, Reward: {reward}, Terminated: {terminated}")

    env = GridWorldGym()
    print("=== INITIAL STATE ===")
    obs, info = env.reset()
    print(f"Observation: {obs}")

    # Q-Learning Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.99 # Discount factor
    epsilon = 1.0 # Exploration rate
    epsilon_decay = 0.995 # Decay rate for exploration
    epsilon_min = 0.01 # Minimum exploration rate
    episodes = 500 # Number of training episodes

    def get_action(state, eps):
        """Epsilon-greedy action selection."""
        # state is a float array, cast indices to int for Q-table
        s_r, s_c = int(state[0]), int(state[1])
        if np.random.rand() < eps:
            return np.random.randint(4)  # Explore: random action
        else:
            return np.argmax(env.q_table[s_r, s_c])  # Exploit: best action from Q-table

    print("Training Q-Learning agent...")
    for episode in range(episodes):
        # Gym reset returns (obs, info)
        state, info = env.reset()
        done = False
        
        while not done:
            action = get_action(state, epsilon)
            # Gym step returns next_state, reward, terminated, truncated, info
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            s_r, s_c = int(state[0]), int(state[1])
            ns_r, ns_c = int(next_state[0]), int(next_state[1])
            
            # Bellman update for Q-table
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(env.q_table[ns_r, ns_c])
            
            td_error = td_target - env.q_table[s_r, s_c, action]
            env.q_table[s_r, s_c, action] += alpha * td_error
            
            # Update V-table (Value function is the max Q-value for the state)
            env.v_table[s_r, s_c] = np.max(env.q_table[s_r, s_c])
            
            state = next_state
            
        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print("Training finished.\n")
    print("Learned Value Table (V_table):")
    print(np.round(env.v_table, 2))

    # Test the learned policy
    print("\nTesting learned policy:")
    env.render_mode = "human"  # Enable rendering to visualize final agent movement
    state, info = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 20: # Limit steps to prevent infinite loop
        # Always exploit during testing
        s_r, s_c = int(state[0]), int(state[1])
        action = np.argmax(env.q_table[s_r, s_c])
        
        action_names = ["Up", "Down", "Left", "Right"]
        print(f"\nAgent chose to move: {action_names[action]}")
        
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    if done:
        print("Goal reached successfully!")
    else:
        print("Failed to reach the goal within the step limit.")