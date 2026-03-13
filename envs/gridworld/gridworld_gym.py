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

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.agent_pos = [0, 0]
        
        # Return observation as numpy float32 array and empty info dict
        obs = np.array(self.agent_pos, dtype=np.float32)
        info = {}
        
        if self.render_mode == "human":
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
        
        if self.render_mode == "human":
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
