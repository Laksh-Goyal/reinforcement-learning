import numpy as np

class GridWorld:
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Start at top-left
        self.goal_pos = [width-1, height-1]  # Goal at bottom-right

    def reset(self):
        self.agent_pos = [0, 0]
        return tuple(self.agent_pos)

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
        done = self.agent_pos == self.goal_pos
        reward = 10 if done else -1
        
        return tuple(self.agent_pos), reward, done

# Usage
env = GridWorld()
state = env.reset()
next_state, reward, done = env.step(1) # Move Down
print(f"Moved to: {next_state}, Reward: {reward}")