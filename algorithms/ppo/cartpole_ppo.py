import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 1e-3 # Increased LR for faster convergence in simple env
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10
MINI_BATCH_SIZE = 64 # Using mini-batches
MAX_STEPS_PER_BATCH = 512 # Update more frequently
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_EPISODES = 1000 # More episodes to allow learning

class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network with a shared backbone."""
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared Backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(64, action_dim)
        
        # Critic head
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        shared_features = self.backbone(x)
        
        # Actor: logits for actions
        logits = self.actor(shared_features)
        
        # Critic: state value estimate
        value = self.critic(shared_features)
        
        return logits, value

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            logits, value = self.policy_old(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item(), value.item()

    def update(self, memory):
        # Convert memory lists to tensors
        old_states = torch.FloatTensor(np.array(memory.states)).to(device)
        old_actions = torch.LongTensor(np.array(memory.actions)).to(device)
        old_logprobs = torch.FloatTensor(np.array(memory.logprobs)).to(device)
        old_values = torch.FloatTensor(np.array(memory.values)).to(device)
        rewards = memory.rewards
        is_terminals = memory.is_terminals

        # 1. Calculate GAE and Returns
        returns = []
        advantages = []
        gae = 0
        
        next_value = 0 # Batch usually ends with a terminal state or dummy value
        
        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(is_terminals[i])
            delta = rewards[i] + GAMMA * next_value * mask - old_values[i]
            gae = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages.insert(0, gae)
            next_value = old_values[i]
            
        returns = torch.tensor(advantages, dtype=torch.float32).to(device) + old_values
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        
        # 2. Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 3. Optimize policy for K epochs:
        dataset_size = old_states.size(0)
        
        for _ in range(K_EPOCHS):
            # Shuffle indices for mini-batch updates
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states = old_states[batch_indices]
                batch_actions = old_actions[batch_indices]
                batch_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions and values
                logits, state_values = self.policy(batch_states)
                state_values = torch.squeeze(state_values)
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy()
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - batch_logprobs.detach())

                # Finding Surrogate Loss:
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * batch_advantages
                
                # Final loss of PPO
                loss = -torch.min(surr1, surr2) + VALUE_LOSS_COEF * self.MseLoss(state_values, batch_returns) - ENTROPY_COEF * dist_entropy
                
                # Take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]

def main():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    ppo = PPO(state_dim, action_dim)
    memory = Memory()
    
    rewards_history = []
    avg_rewards = []
    
    time_step = 0
    update_timestep = MAX_STEPS_PER_BATCH
    
    print(f"Starting PPO training on {ENV_NAME}...")
    
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        current_ep_reward = 0
        done = False
        
        while not done:
            time_step += 1
            
            # Running policy_old
            action, log_prob, state_val = ppo.select_action(state)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.values.append(state_val)
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            current_ep_reward += reward
            
            # Update PPO if time_step reaches update_timestep
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear()
                time_step = 0
        
        rewards_history.append(current_ep_reward)
        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards.append(avg_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode} \t Avg Reward: {avg_reward:.2f}")

        if avg_reward >= 495:
            print(f"Solved! Average reward: {avg_reward:.2f}")
            break
            
    # Save training results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label='Episode Reward', alpha=0.3)
    plt.plot(avg_rewards, label='Average Reward (100 eps)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'PPO - {ENV_NAME}')
    plt.legend()
    plt.savefig('algorithms/ppo/cartpole_ppo_results.png')
    print("Results saved to algorithms/ppo/cartpole_ppo_results.png")
    
    # Test final agent
    test_env = gym.make(ENV_NAME, render_mode=None)
    state, _ = test_env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _, _ = ppo.select_action(state)
        state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Test Reward: {total_reward}")
    test_env.close()

if __name__ == "__main__":
    main()
