# Reinforcement Learning

A clean workspace for reinforcement learning algorithms and experiments.

## Overview

This project is dedicated to implementing and testing various Reinforcement Learning (RL) techniques. It uses **fireup** (kashif's PyTorch port of OpenAI Spinning Up) as the core algorithm library.

**Included algorithms:** VPG, PPO, TRPO, DDPG, TD3, SAC

## Getting Started

### Prerequisites

- macOS with [pyenv](https://github.com/pyenv/pyenv) installed
- Python 3.10.18 available via pyenv (`pyenv install 3.10.18`)
- [Homebrew](https://brew.sh/) and `swig` (`brew install swig`) — required for Box2D environments

### Setup

#### Step 1 — Create the virtual environment

```bash
pyenv virtualenv 3.10.18 spinningup-env
pyenv local spinningup-env   # writes .python-version
```

#### Step 2 — Upgrade pip & install build tools

```bash
pip install --upgrade pip setuptools wheel
```

#### Step 3 — Install PyTorch (CPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Step 4 — Install Gymnasium with extras

```bash
pip install "gymnasium[classic-control,box2d]"
```

#### Step 5 — Clone and install firedup (Spinning Up)

```bash
git clone https://github.com/kashif/firedup.git spinningup
pip install --no-deps -e ./spinningup
pip install scipy matplotlib seaborn pandas ipython joblib tqdm psutil mpi4py
```

> **Note:** We use `--no-deps` to avoid pulling in the old `gym` + `box2d-py` that firedup's setup.py requests. Gymnasium 1.x is already installed and compatible.

#### Step 6 — Verify installation

```bash
python -c "import fireup; import gymnasium; import torch; print('OK')"
```

### Running an algorithm

Train PPO on CartPole for 50 epochs:

```bash
python -m fireup.algos.ppo.ppo --env CartPole-v1 --epochs 50 --cpu 1
```

Train VPG on LunarLander (Box2D):

```bash
python -m fireup.algos.vpg.vpg --env LunarLander-v3 --epochs 100 --cpu 1
```

Train SAC on a continuous control environment:

```bash
python -m fireup.algos.sac.sac --env Pendulum-v1 --epochs 50
```

### Activating the environment

The `.python-version` file in the project root makes pyenv automatically use `spinningup-env`. You can also activate manually:

```bash
pyenv activate spinningup-env
```

## Project Structure

- `spinningup/` — Firedup source (gitignored, cloned from kashif/firedup)
- `src/` — Custom algorithm implementations
- `envs/` — Custom environment definitions
- `notebooks/` — Exploration and visualization
- `tests/` — Unit and integration tests

## Compatibility Notes

- **Library:** `kashif/firedup` (PyTorch port) — not the original `openai/spinningup` (broken on Python 3.8+)
- **Python:** 3.10.18 via pyenv
- **Gymnasium:** 1.2.3 (modern fork of OpenAI Gym)
- **PyTorch:** 2.10.0 (CPU build)
- **MuJoCo:** Not installed. Classic-control and Box2D environments are fully supported.

## License

MIT
