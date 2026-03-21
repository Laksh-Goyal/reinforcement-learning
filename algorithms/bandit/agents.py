import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_estimates = np.zeros(k)
        self.action_counts = np.zeros(k)
        
    def get_action(self):
        if np.random.rand() < self.epsilon:
            # Explore randomly
            return np.random.randint(self.k)
        else:
            # Exploit (break ties randomly)
            max_val = np.max(self.q_estimates)
            best_actions = np.where(self.q_estimates == max_val)[0]
            return np.random.choice(best_actions)
            
    def update(self, action, reward):
        self.action_counts[action] += 1
        # Sample-average update: Q(a) = Q(a) + 1/N(a) * [R - Q(a)]
        self.q_estimates[action] += (1.0 / self.action_counts[action]) * (reward - self.q_estimates[action])

class UCBAgent:
    def __init__(self, k=10, c=2.0):
        self.k = k
        self.c = c
        self.q_estimates = np.zeros(k)
        self.action_counts = np.zeros(k)
        self.t = 0
        
    def get_action(self):
        self.t += 1
        # If there are unexplored actions, prioritize them first
        unexplored = np.where(self.action_counts == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored)
            
        # Calculate UCB values for all actions
        ucb_values = self.q_estimates + self.c * np.sqrt(np.log(self.t) / self.action_counts)
        max_val = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_val)[0]
        return np.random.choice(best_actions)
        
    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_estimates[action] += (1.0 / self.action_counts[action]) * (reward - self.q_estimates[action])
