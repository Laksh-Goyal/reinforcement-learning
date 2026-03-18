import sys
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Add project root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.replay_buffer import ReplayBuffer

# Determine if we have a GPU (Metal/MPS for Apple Silicon, CUDA for NVIDIA, or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

class QNetwork(nn.Module):
    """Simple Multi-Layer Perceptron (MLP) for CartPole."""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Hyperparameters ---
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10 # Sync target network every 10 episodes
LR = 5e-3
EPISODES = 1000
MEMORY_CAPACITY = 20000

# Setup Environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize Main and Target Networks
policy_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # Target network is never explicitly trained

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_CAPACITY)

def select_action(state, eps):
    """Epsilon-greedy action selection."""
    if random.random() < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # Select action with max expected return
            return policy_net(state_tensor).max(1)[1].item()

def optimize_model():
    """Sample from experience replay and perform a gradient descent step."""
    if len(memory) < BATCH_SIZE:
        return # Not enough samples yet

    # Sample a batch
    b_states, b_actions, b_rewards, b_next_states, b_dones = memory.sample(BATCH_SIZE)

    # Convert to PyTorch tensors
    state_batch = torch.FloatTensor(b_states).to(device)
    action_batch = torch.LongTensor(b_actions).unsqueeze(1).to(device)
    # Give rewards a dimensionality of (BATCH_SIZE, 1)
    reward_batch = torch.FloatTensor(b_rewards).unsqueeze(1).to(device) 
    next_state_batch = torch.FloatTensor(b_next_states).to(device)
    done_batch = torch.BoolTensor(b_dones).unsqueeze(1).to(device)

    # 1. Compute Q(s_t, a) from the Main Network
    # Gather takes the Q-values corresponding to the actions actually taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 2. Compute V(s_{t+1}) using the Frozen Target Network
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
        # If the state is a terminal state, next_state_value should be 0
        next_state_values[done_batch] = 0.0

    # 3. Calculate TD Targets
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    # 4. Compute Loss (Huber Loss is less sensitive to outliers than MSE)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # 5. Optimize the Model
    optimizer.zero_grad()
    loss.backward()
    # Optional: Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# --- Training Loop ---
epsilon = EPS_START
consecutive_wins = 0 # Counter for early stopping

print(f"Starting DQN training on {env.spec.id}...")

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store experience
        memory.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # Perform optimization step
        optimize_model()

    # Apply epsilon decay
    epsilon = max(EPS_END, EPS_DECAY * epsilon)

    # Periodic Sync (Hard Update)
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Logging and Early Stopping
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{EPISODES} - Total Reward: {total_reward} - Epsilon: {epsilon:.3f}")

    if total_reward >= 450: # CartPole is considered "solved" at 475~500
        consecutive_wins += 1
    else:
        consecutive_wins = 0
        
    if consecutive_wins >= 10:
        print(f"\nEnvironment solved in {episode + 1} episodes! (Achieved near max reward for 10 consecutive episodes)")
        break

print("Training finished!\n")
env.close()

# --- Evaluation ---
print("Testing trained agent...")
# We initialize it with render_mode='human' to watch it play
test_env = gym.make("CartPole-v1", render_mode="human")
state, _ = test_env.reset()
done = False
total_reward = 0
step = 0

while not done and step < 1000:
    # Always exploit during evaluation
    action = select_action(state, 0.0) 
    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    total_reward += reward
    step += 1

print(f"Test final reward: {total_reward}")
test_env.close()
