import sys
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

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
    """Simple Multi-Layer Perceptron (MLP) for Q-values."""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # TODO: Implement the Q-network architecture
        pass

    def forward(self, x):
        # TODO: Implement the forward pass
        pass

class EnvironmentModel(nn.Module):
    """Predicts next state and reward given current state and action."""
    def __init__(self, state_dim, action_dim):
        super(EnvironmentModel, self).__init__()
        # TODO: Implement the Environment Model architecture. 
        # Note: the input should take both state and action. You can use an embedding for discrete actions.
        # The output should predict the next_state and the reward.
        pass

    def forward(self, state, action):
        # TODO: Implement the forward pass. Return (next_state_prediction, reward_prediction)
        pass

# --- Hyperparameters ---
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10 
LR_Q = 1e-3
LR_MODEL = 1e-3
EPISODES = 500
MEMORY_CAPACITY = 10000
PLANNING_STEPS = 10 # Number of planning steps per real interaction

def select_action(state, eps, policy_net, action_dim):
    """Epsilon-greedy action selection."""
    if random.random() < eps:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_tensor).max(1)[1].item()

def optimize_q_network(policy_net, target_net, optimizer, transitions):
    """
    Perform a DQN update using either real or simulated transitions.
    transitions is a tuple of (states, actions, rewards, next_states, dones).
    """
    # TODO: Implement standard DQN loss calculation and backpropagation
    pass

def optimize_environment_model(env_model, optimizer, transitions):
    """
    Train the environment model to predict next states and rewards.
    """
    # TODO: Calculate MSE loss between predicted (next_states, rewards) and actual targets.
    # Backpropagate to train env_model.
    pass

def planning_step(policy_net, target_net, q_optimizer, env_model, memory, batch_size, planning_steps):
    """
    The Dyna-Q planning phase:
    1. Sample states and actions that have been visited.
    2. Use the model to predict next states and rewards.
    3. Update Q-network with these simulated transitions.
    """
    if len(memory) < batch_size:
        return
    
    # TODO: Implement the planning loop. Run `planning_steps` times.
    # In each step, sample a batch from memory (just need state and action, or sample random actions).
    # Pass them to env_model to get simulated next_state and reward.
    # Package into simulated transitions and call optimize_q_network.
    pass

def main():
    # Setup Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize Networks
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    env_model = EnvironmentModel(state_dim, action_dim).to(device)

    # Optimizers
    q_optimizer = optim.Adam(policy_net.parameters(), lr=LR_Q)
    model_optimizer = optim.Adam(env_model.parameters(), lr=LR_MODEL)
    
    memory = ReplayBuffer(MEMORY_CAPACITY)

    epsilon = EPS_START
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 1. Select and take action
            action = select_action(state, epsilon, policy_net, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store in replay buffer
            memory.add(state, action, reward, next_state, done)
            
            # Sample a batch of REAL experience
            if len(memory) >= BATCH_SIZE:
                real_transitions = memory.sample(BATCH_SIZE)
                
                # 2. Optimize Q-Network with real experience
                optimize_q_network(policy_net, target_net, q_optimizer, real_transitions)
                
                # 3. Optimize Environment Model with real experience
                optimize_environment_model(env_model, model_optimizer, real_transitions)
                
                # 4. Planning: Optimize Q-Network with SIMULATED experience
                planning_step(policy_net, target_net, q_optimizer, env_model, memory, BATCH_SIZE, PLANNING_STEPS)

            state = next_state
            total_reward += reward

        epsilon = max(EPS_END, EPS_DECAY * epsilon)
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} - Total Reward: {total_reward} - Epsilon: {epsilon:.3f}")

    env.close()

if __name__ == "__main__":
    main()
