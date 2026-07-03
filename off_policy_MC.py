import numpy as np
from play_game import play_game
import random
import statistics
''' 
in this file i implemented the solution for the problem with off policy MC 
'''
class OffPoicyMC:
    def __init__(self, MDP, discount, num_episodes, policy_pi, action_value_fun):
        self.MDP = MDP
        self.discount = discount
        self.policy_pi = policy_pi
        self.policy_pi[9, :, :, 1] = 1
        self.policy_pi[9, :, :, 0] = 0
        self.action_value_fun = np.zeros((10, 10, 2, 2))
        self.C = np.zeros((10, 10, 2, 2))
        self.num_episodes = num_episodes
    
    def generate_soft_policy(self):
        b = np.zeros((10, 10, 2, 2))

        hit_prob = random.uniform(0, 1)
        flip_coin = random.random(0, 1)
        if flip_coin == 1:
            b[:,:,:,0] = hit_prob
            b[:,:,:,1] = 1-hit_prob
        
        else:
            b[:,:,:,0] = 1-hit_prob
            b[:,:,:,1] = hit_prob

    
        return b


    def off_poicy_MC_model(self):
        for i in range(0, self.num_episodes):

            policy_b = self.generate_soft_policy()

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

            self.MDP.dealer_hidden_sum = dealer_first_card + dealer_second_card

            if random.random() <= self.policy_b[S_0[0]-12, S_0[1]-1, S_0[2], 0]:
                A_0 = 0 
            else:
                A_0 = 1

            episode_steps = play_game(self.MDP, policy_b, S_0, A_0)

            G = 0 
            W = 1

            for t in range(len(episode_steps)-2, -1, -1): 
                step = episode_steps[t]
                next_step = episode_steps[t+1]

                state = step["state"]
                action = step["action"]

                G = self.discount * G + next_step["reward"]
                self.C[state[0] - 12, state[1] - 1, state[2], action] += W

                Q_S_A = self.action_value_fun[state[0] - 12, state[1] - 1, state[2], action]
                C_S_A = self.C[state[0] - 12, state[1] - 1, state[2], action]

                Q_S_A = Q_S_A + (W/C_S_A)*(G - Q_S_A)
                
                self.action_value_fun[state[0] - 12, state[1] - 1, state[2], action] = Q_S_A

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

                if self.policy_pi[state[0] - 12, state[1] - 1, state[2], action] == 0:
                    break

                W = W*(1/policy_b[state[0] - 12, state[1] - 1, state[2], action]) 