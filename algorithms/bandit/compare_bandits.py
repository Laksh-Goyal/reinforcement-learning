import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.bandit.k_armed_bandit import KArmedBandit
from algorithms.bandit.agents import EpsilonGreedyAgent, UCBAgent

def run_experiment(agent_func, env_func, runs=2000, steps=1000, k=10):
    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))
    
    for r in range(runs):
        if (r+1) % 500 == 0:
            print(f"Run {r+1}/{runs}...")
        env = env_func(k)
        agent = agent_func(k)
        
        for t in range(steps):
            action = agent.get_action()
            reward, is_optimal = env.step(action)
            agent.update(action, reward)
            
            rewards[r, t] = reward
            optimal_actions[r, t] = is_optimal
            
    return rewards.mean(axis=0), optimal_actions.mean(axis=0)

if __name__ == "__main__":
    k = 10
    runs = 2000
    steps = 1000
    
    print(f"Running Epsilon-Greedy (epsilon=0.1) over {runs} runs of {steps} steps...")
    eps_rewards, eps_optimal = run_experiment(
        lambda k: EpsilonGreedyAgent(k, epsilon=0.1),
        lambda k: KArmedBandit(k),
        runs=runs, steps=steps, k=k
    )
    
    print(f"\nRunning UCB (c=2.0) over {runs} runs of {steps} steps...")
    ucb_rewards, ucb_optimal = run_experiment(
        lambda k: UCBAgent(k, c=2.0),
        lambda k: KArmedBandit(k),
        runs=runs, steps=steps, k=k
    )
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Average Reward Plot
    ax1.plot(eps_rewards, label='Epsilon-Greedy (eps=0.1)', color='blue')
    ax1.plot(ucb_rewards, label='UCB (c=2.0)', color='red')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward over 2000 Runs')
    ax1.legend()
    
    # % Optimal Action Plot
    ax2.plot(eps_optimal * 100, label='Epsilon-Greedy (eps=0.1)', color='blue')
    ax2.plot(ucb_optimal * 100, label='UCB (c=2.0)', color='red')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal Action')
    ax2.set_title('Percentage of Optimal Action Selections')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('bandit_results.png')
    print("\nFinished! Plot saved to bandit_results.png")
