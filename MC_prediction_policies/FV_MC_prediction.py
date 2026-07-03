import numpy as np
import play_game as plf

class FVMCPrediction:
    def __init__(self, MDP, discount, num_episodes):
        self.MDP = MDP
        self.discount = discount
        self.num_episodes = num_episodes
        self.V = np.zeros((10, 10, 2))
        self.num_returns = np.zeros((10, 10, 2))
        self.sum_returns = np.zeros((10, 10, 2))

        self.returns = np.empty((10, 10, 2), dtype=object)
        for index in np.ndindex(self.returns.shape):
            self.returns[index] = []

        self.policy_pi = np.zeros((10, 10, 2, 2))
        for i in range(0, 8):
            self.policy_pi[i, :, :] = [1, 0]
        self.policy_pi[8, :, :] = [0, 1]
        self.policy_pi[9, :, :] = [0, 1]
    
    def run(self):
        for _ in range(self.num_episodes):
            S_0 = self.MDP.states[np.random.randint(0, 200)]
            A_0 = 1 if S_0[0] == 20 or S_0[0] == 21 else 0

            episode_steps = plf.play_game(self.MDP, self.policy_pi, S_0, A_0)

            G = 0

            for t in range(len(episode_steps) - 2, -1, -1):
                step = episode_steps[t]
                next_step = episode_steps[t+1]

                G = self.discount * G + next_step["reward"]
                s = tuple(step["state"])
                    
                self.returns[s[0]-12, s[1]-1, s[2]].append(G)
                self.num_returns[s[0]-12, s[1]-1, s[2]] += 1
                self.sum_returns[s[0]-12, s[1]-1, s[2]] += G
                self.V[s[0]-12, s[1]-1, s[2]] = (self.sum_returns[s[0]-12, s[1]-1, s[2]] 
                                                    / self.num_returns[s[0]-12, s[1]-1, s[2]])