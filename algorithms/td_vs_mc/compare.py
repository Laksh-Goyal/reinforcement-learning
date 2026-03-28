import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from envs.gridworld.gridworld_gym import GridWorldGym

class ValueEstimationAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.v_table = np.zeros((env.height, env.width))

    def get_action(self, state):
        # Random policy for value estimation comparison
        return self.env.action_space.sample()

class MonteCarloAgent(ValueEstimationAgent):
    def update(self, trajectory):
        """
        trajectory: list of (state, reward)
        """
        g = 0
        visited_states = set()
        # First-visit MC
        for i in range(len(trajectory) - 1, -1, -1):
            state, reward = trajectory[i]
            g = self.gamma * g + reward
            state_tuple = (int(state[0]), int(state[1]))
            
            # Check if this is the first visit to the state in this episode
            if state_tuple not in [ (int(s[0]), int(s[1])) for s, r in trajectory[:i]]:
                r, c = state_tuple
                self.v_table[r, c] += self.alpha * (g - self.v_table[r, c])

class TDZeroAgent(ValueEstimationAgent):
    def update(self, state, reward, next_state, done):
        r, c = int(state[0]), int(state[1])
        nr, nc = int(next_state[0]), int(next_state[1])
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.v_table[nr, nc]
            
        self.v_table[r, c] += self.alpha * (target - self.v_table[r, c])

class TDLambdaAgent(ValueEstimationAgent):
    def __init__(self, env, alpha=0.1, gamma=0.99, lbda=0.9):
        super().__init__(env, alpha, gamma)
        self.lbda = lbda
        self.e_table = np.zeros((env.height, env.width))

    def reset_traces(self):
        self.e_table.fill(0)

    def update(self, state, reward, next_state, done):
        r, c = int(state[0]), int(state[1])
        nr, nc = int(next_state[0]), int(next_state[1])
        
        # TD error
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.v_table[nr, nc]
        
        td_error = td_target - self.v_table[r, c]
        
        # Update eligibility trace (replacing trace)
        self.e_table[r, c] = 1.0
        
        # Update all values based on current TD error and traces
        self.v_table += self.alpha * td_error * self.e_table
        
        # Decay traces
        self.e_table *= self.gamma * self.lbda

def compute_true_v(env, gamma=0.99, threshold=1e-6):
    """Compute true value function for a random policy using iterative policy evaluation."""
    v = np.zeros((env.height, env.width))
    while True:
        delta = 0
        for r in range(env.height):
            for c in range(env.width):
                if [r, c] == env.goal_pos:
                    continue
                old_v = v[r, c]
                new_v = 0
                for action in range(4): # Up, Down, Left, Right
                    # Simulate step
                    nr, nc = r, c
                    if action == 0 and r > 0: nr -= 1
                    elif action == 1 and r < env.height - 1: nr += 1
                    elif action == 2 and c > 0: nc -= 1
                    elif action == 3 and c < env.width - 1: nc += 1
                    
                    reward = 10.0 if [nr, nc] == env.goal_pos else -1.0
                    new_v += 0.25 * (reward + gamma * v[nr, nc])
                v[r, c] = new_v
                delta = max(delta, abs(old_v - v[r, c]))
        if delta < threshold:
            break
    return v

def run_comparison(episodes=500):
    env = GridWorldGym()
    true_v = compute_true_v(env)
    
    agents = {
        'MC': MonteCarloAgent(env),
        'TD(0)': TDZeroAgent(env),
        'TD(0.5)': TDLambdaAgent(env, lbda=0.5),
        'TD(0.9)': TDLambdaAgent(env, lbda=0.9)
    }
    
    errors = {name: [] for name in agents}
    
    for ep in range(episodes):
        # Monte Carlo Agent (Batch update)
        state, info = env.reset()
        trajectory = []
        done = False
        while not done:
            action = agents['MC'].get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, reward))
            state = next_state
            done = terminated or truncated
        agents['MC'].update(trajectory)
        
        # TD-based Agents (Online updates)
        for name, agent in agents.items():
            if name == 'MC': continue
            
            state, info = env.reset()
            if hasattr(agent, 'reset_traces'):
                agent.reset_traces()
            done = False
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(state, reward, next_state, terminated or truncated)
                state = next_state
                done = terminated or truncated
            
        # Record RMS Error for all agents
        for name, agent in agents.items():
            rmse = np.sqrt(np.mean((agent.v_table - true_v)**2))
            errors[name].append(rmse)
        
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, err_list in errors.items():
        plt.plot(err_list, label=name)
        
    plt.xlabel('Episodes')
    plt.ylabel('RMS Error')
    plt.title('Convergence Analysis: TD(0) vs TD(λ) vs Monte Carlo')
    plt.legend()
    plt.grid(True)
    
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'td_vs_mc_results.png')
    plt.savefig(results_path)
    print(f"Comparison results saved to {results_path}")
    
    # Print final V-tables
    print("\nTrue Value Function:")
    print(np.round(true_v, 2)) 
    for name, agent in agents.items():
        print(f"\nFinal {name} Value Function:")
        print(np.round(agent.v_table, 2))

if __name__ == "__main__":
    run_comparison()
