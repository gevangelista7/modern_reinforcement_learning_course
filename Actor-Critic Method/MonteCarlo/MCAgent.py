import itertools
import numpy as np

N_games = 500000

class MCAgent:
        def __init__(self, obs_space, act_space):
            self.Q = {}
            self.policy = {}
            self.actions = range(act_space.n)

            obs_dims_ranges = [range(dim.n) for dim in obs_space]
            obs_combinations = list(itertools.product(*obs_dims_ranges))
            for obs in obs_combinations:
                self.policy[obs] = 1
                for act in self.actions:
                    self.Q[obs, act] = 0.0

            self.returns = []

        def choose_action(self, state):
            return self.policy[state]

        def insert_memory(self, state, action, reward):
            self.returns.append([(state, action), reward])

        def update_Q(self):
            keys = [mem[0] for mem in self.returns]
            keys = list(set(keys))
            for key in keys:
                mean_rwd = np.mean([mem[-1] for mem in self.returns if mem[0] == key])
                self.Q[key] = mean_rwd

        def update_policy(self):
            rewards = []
            for state in self.policy.keys():
                for act in self.actions:
                    rewards.append(self.Q[state, act])
                self.policy[state] = np.argmax(rewards)
                rewards.clear()

        def clear_memory(self):
            self.returns.clear()


