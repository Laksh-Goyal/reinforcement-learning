import numpy as np

class GridWorld:
    def __init__(self, width=4, height=4):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]  # Start at top-left
        self.goal_pos = [height-1, width-1]  # Goal at bottom-right
        
        # Q-table: stores Q-values for each state-action pair (height, width, num_actions)
        self.q_table = np.zeros((height, width, 4))
        
        # Value table: stores the expected return for each state (height, width)
        self.v_table = np.zeros((height, width))

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

    def render(self):
        grid = np.full((self.height, self.width), ".", dtype=str)
        grid[self.goal_pos[0], self.goal_pos[1]] = "G"
        grid[self.agent_pos[0], self.agent_pos[1]] = "A"
        print("\n" + "\n".join([" ".join(row) for row in grid]))
        print("-" * (self.width * 2))

# Q-Learning Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99 # Discount factor
epsilon = 1.0 # Exploration rate
epsilon_decay = 0.995 # Decay rate for exploration
epsilon_min = 0.01 # Minimum exploration rate
episodes = 500 # Number of training episodes

env = GridWorld()

def get_action(state, eps):
    """Epsilon-greedy action selection."""
    if np.random.rand() < eps:
        return np.random.randint(4)  # Explore: random action
    else:
        return np.argmax(env.q_table[state[0], state[1]])  # Exploit: best action from Q-table

print("Training Q-Learning agent...")
for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = get_action(state, epsilon)
        next_state, reward, done = env.step(action)
        
        # Bellman update for Q-table
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * np.max(env.q_table[next_state[0], next_state[1]])
        
        td_error = td_target - env.q_table[state[0], state[1], action]
        env.q_table[state[0], state[1], action] += alpha * td_error
        
        # Update V-table (Value function is the max Q-value for the state)
        env.v_table[state[0], state[1]] = np.max(env.q_table[state[0], state[1]])
        
        state = next_state
        
    # Decay epsilon after each episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("Training finished.\n")
print("Learned Value Table (V_table):")
print(np.round(env.v_table, 2))

# Test the learned policy
print("\nTesting learned policy:")
state = env.reset()
env.render()
done = False
step_count = 0

while not done and step_count < 20: # Limit steps to prevent infinite loop
    # Always exploit during testing
    action = np.argmax(env.q_table[state[0], state[1]])
    
    action_names = ["Up", "Down", "Left", "Right"]
    print(f"Agent chose to move: {action_names[action]}")
    
    state, reward, done = env.step(action)
    env.render()
    step_count += 1

if done:
    print("Goal reached successfully!")
else:
    print("Failed to reach the goal within the step limit.")