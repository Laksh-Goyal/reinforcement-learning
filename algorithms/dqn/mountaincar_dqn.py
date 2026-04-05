import sys
import os
import argparse
import glob
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Add project root to path to import utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
from utils.replay_buffer import ReplayBuffer

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ── Model save directory (models/<algorithm>/<environment>/) ──────────────────
ALGORITHM  = "dqn"
ENVIRONMENT = "mountaincar"
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models", ALGORITHM, ENVIRONMENT)


# ── Network ───────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """Simple Multi-Layer Perceptron (MLP) for MountainCar."""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE        = 128
GAMMA             = 0.99
EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY         = 0.995
TARGET_UPDATE     = 10
LR                = 1e-3
EPISODES          = 2000
MEMORY_CAPACITY   = 50000
SUCCESS_THRESHOLD = -110   # Considered solved


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_env(render: bool = False):
    return gym.make("MountainCar-v0", render_mode="human" if render else None)


def select_action(policy_net, env, state, eps: float) -> int:
    if random.random() < eps:
        return env.action_space.sample()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        return policy_net(state_tensor).max(1)[1].item()


def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return None

    b_states, b_actions, b_rewards, b_next_states, b_dones = memory.sample(BATCH_SIZE)

    state_batch      = torch.FloatTensor(b_states).to(device)
    action_batch     = torch.LongTensor(b_actions).unsqueeze(1).to(device)
    reward_batch     = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
    next_state_batch = torch.FloatTensor(b_next_states).to(device)
    done_batch       = torch.BoolTensor(b_dones).unsqueeze(1).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
        next_state_values[done_batch] = 0.0

    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    criterion = nn.HuberLoss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()


def get_latest_model() -> str | None:
    """Return the path of the most recently saved model in MODEL_DIR, or None."""
    pattern = os.path.join(MODEL_DIR, "*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def save_model(policy_net, episode: int) -> str:
    """Save policy_net weights; return the path."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"policy_ep{episode}_{timestamp}.pt"
    path      = os.path.join(MODEL_DIR, filename)
    torch.save(policy_net.state_dict(), path)
    print(f"Model saved → {path}")
    return path


def load_model(path: str, state_dim: int, action_dim: int) -> QNetwork:
    """Load a QNetwork from a checkpoint file."""
    net = QNetwork(state_dim, action_dim).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    print(f"Model loaded ← {path}")
    return net


# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    env = make_env(render=False)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory    = ReplayBuffer(MEMORY_CAPACITY)

    epsilon          = EPS_START
    consecutive_wins = 0
    episode_rewards  = []
    episode_losses   = []

    print(f"Starting DQN training on {env.spec.id}...")

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_true_reward = 0
        done = False
        step_losses = []

        while not done:
            action = select_action(policy_net, env, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_true_reward += reward

            # ── Reward shaping ────────────────────────────────────────────────
            position        = next_state[0]
            velocity        = next_state[1]
            velocity_bonus  = abs(velocity) * 10
            position_bonus  = (position + 1.2) ** 2
            goal_bonus      = 50.0 if terminated else 0.0
            shaped_reward   = reward + velocity_bonus + position_bonus + goal_bonus
            # ─────────────────────────────────────────────────────────────────

            memory.add(state, action, shaped_reward, next_state, done)
            state = next_state

            loss_val = optimize_model(policy_net, target_net, optimizer, memory)
            if loss_val is not None:
                step_losses.append(loss_val)

        epsilon = max(EPS_END, EPS_DECAY * epsilon)

        episode_rewards.append(total_true_reward)
        episode_losses.append(np.mean(step_losses) if step_losses else 0.0)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{EPISODES} | Avg(10) Reward: {avg_reward:.1f} | Epsilon: {epsilon:.3f}")

        if total_true_reward >= SUCCESS_THRESHOLD:
            consecutive_wins += 1
        else:
            consecutive_wins = 0

        if consecutive_wins >= 10:
            print(f"\nEnvironment solved in {episode + 1} episodes! "
                  f"(>= {SUCCESS_THRESHOLD} reward for 10 consecutive episodes)")
            break

    print("Training finished!\n")
    env.close()

    # Save the trained model
    save_model(policy_net, episode + 1)

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(episode_rewards, color="blue", alpha=0.6)
    window = 20
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        axs[0].plot(range(window - 1, len(episode_rewards)), moving_avg, color="navy", linewidth=2)
    axs[0].set_title("Episode Rewards (Unshaped)")
    axs[0].set_ylabel("Reward")

    axs[1].plot(episode_losses, color="red", alpha=0.6)
    axs[1].set_title("Episode Average Loss")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Loss")

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_results.png")
    plt.savefig(plot_path)
    print(f"Saved training plot → {plot_path}\n")


# ── Test ──────────────────────────────────────────────────────────────────────
def test(model_path: str | None = None):
    # Resolve model path
    if model_path is None:
        model_path = get_latest_model()
        if model_path is None:
            print(f"No trained models found in '{MODEL_DIR}'.")
            print("Run with --mode train first, or pass --model-path <path>.")
            sys.exit(1)
        print(f"No --model-path given; using latest model.")

    env = make_env(render=True)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = load_model(model_path, state_dim, action_dim)

    print(f"\nTesting agent on {env.spec.id}...")
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action(policy_net, env, state, eps=0.0)  # pure exploit
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        time.sleep(0.01)

    print(f"Test episode reward: {total_reward}")
    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DQN agent for MountainCar-v0",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        required=True,
        help="'train'  – train a new agent and save it\n"
             "'test'   – load and visualise a trained agent",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="(test mode only) path to a .pt checkpoint.\n"
             "If omitted, the latest model in models/dqn/mountaincar/ is used.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        test(model_path=args.model_path)
