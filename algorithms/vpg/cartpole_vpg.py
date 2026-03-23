import os
import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Determine if we have a GPU (Metal/MPS for Apple Silicon, CUDA for NVIDIA, or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class PolicyNetwork(nn.Module):
    """Simple Multi-Layer Perceptron (MLP) for Policy Network."""
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # VPG needs probabilities, so we use softmax on the output layer
        # Categorical distribution takes logits or probs, but doing softmax
        # makes it explicit. However, raw logits are often better for numerical stability.
        # We'll just output the logits and use them in Categorical.
        return self.fc3(x)

# --- Hyperparameters ---
GAMMA = 0.99
LR = 1e-3
EPISODES = 1000

# Setup Environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize Policy Network
policy_net = PolicyNetwork(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

def select_action(state):
    """Samples an action from the policy probability distribution."""
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    logits = policy_net(state_tensor)
    
    # Create a categorical distribution based on the action probabilities/logits
    dist = Categorical(logits=logits)
    action = dist.sample()
    
    # Also return the log probability of the selected action, needed for VPG loss calculation
    return action.item(), dist.log_prob(action)

def calculate_returns(rewards):
    """Calculates discounted cumulative returns for each time step."""
    returns = []
    G_t = 0
    # Iterate backwards through the rewards
    for r in reversed(rewards):
        G_t = r + GAMMA * G_t
        returns.insert(0, G_t)
    return returns

# --- Training Loop ---
episode_rewards = []
episode_losses = []

consecutive_wins = 0

print(f"Starting Vanilla Policy Gradient (REINFORCE) training on {env.spec.id}...")

for episode in range(EPISODES):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    
    done = False
    total_reward = 0
    
    # 1. Collect a trajectory
    while not done:
        action, log_prob = select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        
        state = next_state
        
    episode_rewards.append(total_reward)
    
    # 2. Calculate discounted returns
    returns = calculate_returns(rewards)
    returns = torch.tensor(returns).to(device)
    
    # 3. Normalize returns (variance reduction)
    # Adding a small epsilon to standard deviation to avoid division by zero
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    # 4. Calculate policy gradient loss
    # Loss = - E [ log(prob) * return ]
    policy_loss = []
    for log_prob, G_t in zip(log_probs, returns):
        policy_loss.append(-log_prob * G_t)
        
    # Sum the loss for the entire episode
    policy_loss = torch.cat(policy_loss).sum()
    episode_losses.append(policy_loss.item())

    # 5. Perform backpropagation and parameter update
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    # Logging and Early Stopping
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES} - Total Reward: {total_reward}")

    if total_reward >= 450: # CartPole env max is 500, consider "solved" at high reward
        consecutive_wins += 1
    else:
        consecutive_wins = 0
        
    if consecutive_wins >= 10:
        print(f"\nEnvironment solved in {episode + 1} episodes! (Achieved near max reward for 10 consecutive episodes)")
        break

print("Training finished!\n")
env.close()

# Plot the results
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(episode_rewards, color='blue')
axs[0].set_title('Episode Rewards')
axs[0].set_ylabel('Total Reward')

axs[1].plot(episode_losses, color='red')
axs[1].set_title('Episode Loss (Policy Gradient)')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Loss')

plt.tight_layout()
plt.savefig('cartpole_vpg_results.png')
print("Saved plot to cartpole_vpg_results.png\n")

# --- Evaluation ---
print("Testing trained agent...")
# We initialize it with render_mode=None for headless, change to 'human' if you want GUI
test_env = gym.make("CartPole-v1", render_mode=None)
state, _ = test_env.reset()
done = False
total_reward = 0
step_count = 0

while not done and step_count < 1000:
    # During testing we can just take the action with highest probability 
    # (argmax of logits) to act deterministically rather than sampling
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits = policy_net(state_tensor)
        action = torch.argmax(logits, dim=-1).item()
        
    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1

print(f"Test final reward: {total_reward}")
test_env.close()
