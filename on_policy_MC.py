import numpy as np
from play_game import play_game
import random
import statistics
''' 
in this file i implemented the solution for the problem with on policy first visist MC e-greedy policy
'''
class OnPoicyMC:
    def __init__(self, MDP, discount, num_episodes, policy_pi, action_value_fun, epsilon):
        self.MDP = MDP
        self.discount = discount
        self.policy_pi = policy_pi
        self.policy_pi[9, :, :, 1] = 1
        self.policy_pi[9, :, :, 0] = 0
        self.epsilon = epsilon
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
            
    def on_poicy_MC_model(self):
        for i in range(0, self.num_episodes):
            usable_ace = False
            player_first_card = random.randint(1, 10)
            players_sum = player_first_card
            if player_first_card == 1:
                    usable_ace = True
                    players_sum = 11
            
            while players_sum < 12:
                player_second_card = random.randint(1, 10)
                if (players_sum != 11 and player_second_card == 1):
                    usable_ace = True
                    players_sum += 11
                else:
                    players_sum += player_second_card

            dealer_first_card = random.randint(1, 10)
            dealer_second_card = random.randint(1, 10)
            
            if usable_ace:
                S_0 = np.array([players_sum, dealer_second_card, 1], dtype=int)
            else:
                S_0 = np.array([players_sum, dealer_second_card, 0], dtype=int)

            ############### not corret no ace considered
            # self.MDP.dealer_hidden_sum = dealer_first_card + dealer_second_card

            if random.randint(0,1) <= self.policy_pi[S_0[0]-12, S_0[1]-1, S_0[2], 0]:
                A_0 = 0 
            else:
                A_0 = 1

            episode_steps = play_game(self.MDP, self.policy_pi, S_0, A_0)

            G = 0 

            for t in range(len(episode_steps)-2, -1, -1): # does it iterate correctly????
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
                        
                        self.policy_pi[state[0] - 12, state[1] - 1, state[2], 0] = self.epsilon/2
                        self.policy_pi[state[0] - 12, state[1] - 1, state[2], 1] = 1-self.epsilon + self.epsilon/2
                    
                    elif (self.action_value_fun[state[0] - 12, state[1] - 1, state[2], 1] <
                        self.action_value_fun[state[0] - 12 , state[1] - 1, state[2], 0]):
                        
                        self.policy_pi[state[0] - 12, state[1] - 1, state[2], 0] = 1-self.epsilon + self.epsilon/2
                        self.policy_pi[state[0] - 12, state[1] - 1, state[2], 1] = self.epsilon/2
                    
                    else:
                        
                        self.policy_pi[state[0] - 12, state[1] - 1, state[2], 0] = 0.5
                        self.policy_pi[state[0] - 12, state[1] - 1, state[2], 1] = 0.5