import numpy as np
import statistics
import random 
from play_game import play_game

'''In this file the problem is solved with Mont Carlo Exploring Start'''
class MCES:
    def __init__(self, MDP, discount, num_episodes, policy_pi, action_value_fun):
        self.MDP = MDP
        self.discount = discount
        self.policy_pi = policy_pi
        self.policy_pi[9, :, :, 1] = 1
        self.policy_pi[9, :, :, 0] = 0
        self.action_value_fun = np.zeros((10, 10, 2, 2))
        self.returns = np.empty((10, 10, 2, 2), dtype=object)

        for index in np.ndindex(self.returns.shape):
            self.returns[index] = []

        self.num_episodes = num_episodes

    def is_first_visit(self, state, action, previouse_steps):

        for step in previouse_steps:
            if np.array_equal(step["state"], state) and step["action"] == action:
                return False
            
        return True
            
    def MCES_model(self):
        for i in range(0, self.num_episodes):
            
            S_0 = self.MDP.states[np.random.randint(0, 200)]
            A_0 = random.randint(0, 1)

            episode_steps = play_game(self.MDP, self.policy_pi, S_0, A_0)

            G = 0 
                
            for t in range(len(episode_steps)-2, -1, -1): 
                step = episode_steps[t]
                next_step = episode_steps[t+1]
                G = self.discount * G + next_step["reward"]
                
                if self.is_first_visit(step["state"], step["action"], episode_steps[0:t]):
                    state = step["state"]
                    action = step["action"]
                
                    self.returns[state[0] - 12, state[1] - 1, state[2], action].append(G)

                    self.action_value_fun[state[0] - 12, state[1] - 1, state[2], action] = statistics.mean(self.returns[state[0] - 12, state[1] - 1, state[2], action])

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

                        

            

    