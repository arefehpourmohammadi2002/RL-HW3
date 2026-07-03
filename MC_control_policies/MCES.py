import numpy as np
import random 
from play_game import play_game

class MCES:
    def __init__(self, MDP, discount, num_episodes, policy_pi, action_value_fun):
        self.MDP = MDP
        self.discount = discount
        self.policy_pi = policy_pi
        self.action_value_fun = action_value_fun
        self.returns = np.empty((10, 10, 2, 2), dtype=object)

        for index in np.ndindex(self.returns.shape):
            self.returns[index] = []

        self.num_returns = np.zeros((10, 10, 2, 2))
        self.sum_returns = np.zeros((10, 10, 2, 2))
        self.num_episodes = num_episodes
            
    def MCES_model(self):
        for _ in range(0, self.num_episodes):
            
            S_0 = self.MDP.states[np.random.randint(0, 200)]
            A_0 = random.randint(0, 1)

            episode_steps = play_game(self.MDP, self.policy_pi, S_0, A_0)

            G = 0 
                
            for t in range(len(episode_steps)-2, -1, -1): 
                step = episode_steps[t]
                next_step = episode_steps[t+1]
                G = self.discount * G + next_step["reward"]
                
                state = step["state"]
                action = step["action"]
            
                self.returns[state[0] - 12, state[1] - 1, state[2], action].append(G)

                self.num_returns[state[0] - 12, state[1] - 1, state[2], action] += 1
                self.sum_returns[state[0] - 12, state[1] - 1, state[2], action] += G

                num = self.num_returns[state[0] - 12, state[1] - 1, state[2], action]
                sum = self.sum_returns[state[0] - 12, state[1] - 1, state[2], action]

                self.action_value_fun[state[0] - 12, state[1] - 1, state[2], action] = sum / num

                if (self.action_value_fun[state[0] - 12, state[1] - 1, state[2], 0] <
                    self.action_value_fun[state[0] - 12, state[1] - 1, state[2], 1]):
                    
                    self.policy_pi[state[0] - 12, state[1] - 1, state[2], 0] = 0
                    self.policy_pi[state[0] - 12, state[1] - 1, state[2], 1] = 1
                
                elif (self.action_value_fun[state[0] - 12, state[1] - 1, state[2], 1] <
                    self.action_value_fun[state[0] - 12 , state[1] - 1, state[2], 0]):

                    self.policy_pi[state[0] - 12, state[1] - 1, state[2], 0] = 1
                    self.policy_pi[state[0] - 12, state[1] - 1, state[2], 1] = 0
                
                else:

                    self.policy_pi[state[0] - 12, state[1] - 1, state[2], 0] = 0.5
                    self.policy_pi[state[0] - 12, state[1] - 1, state[2], 1] = 0.5

                        

            

    