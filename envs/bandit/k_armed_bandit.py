import numpy as np

class KArmedBandit:
    """
    Standard K-Armed Bandit environment.
    Each arm has a true mean reward drawn from a standard normal distribution.
    When stepping an arm, the reward is drawn from a normal distribution 
    with that arm's true mean and variance 1.
    """
    def __init__(self, k=10):
        self.k = k
        self.action_space_n = k
        self.reset()
        
    def reset(self):
        # True values of the k arms, q*(a) ~ N(0, 1)
        self.q_true = np.random.randn(self.k)
        # Optimal action is the arm with the highest true mean
        self.optimal_action = np.argmax(self.q_true)
        return 0 # bandits have no "state"
        
    def step(self, action):
        assert 0 <= action < self.k
        # Reward R_t ~ N(q*(action), 1)
        reward = np.random.randn() + self.q_true[action]
        is_optimal = int(action == self.optimal_action)
        return reward, is_optimal
