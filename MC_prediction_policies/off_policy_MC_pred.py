import numpy as np
from play_game import play_game
import random

class OffPoicyMCPred:
    def __init__(self, MDP, discount, num_episodes, action_value_fun):
        self.MDP = MDP
        self.discount = discount
        self.action_value_fun = action_value_fun
        self.C = np.zeros((10, 10, 2, 2))
        self.num_episodes = num_episodes

        self.policy_pi = np.zeros((10, 10, 2, 2))
        for i in range(0, 8):
            self.policy_pi[i, :, :] = [1, 0]
        self.policy_pi[8, :, :] = [0, 1]
        self.policy_pi[9, :, :] = [0, 1]
        
    def generate_soft_policy(self):
        b = np.full((10, 10, 2, 2), 0.5)
        return b

    def run(self):
        for _ in range(0, self.num_episodes):

            policy_b = self.generate_soft_policy()

            usable_ace = False
            player_first_card = min(random.randint(1, 13), 10)
            players_sum = player_first_card

            if player_first_card == 1:
                    usable_ace = True
                    players_sum = 11
            
            while players_sum < 12:
                player_second_card = min(random.randint(1, 13), 10)
                if (players_sum != 11 and player_second_card == 1):
                    usable_ace = True
                    players_sum += 11
                else:
                    players_sum += player_second_card

            dealer_second_card = min(random.randint(1, 13), 10)
            
            if usable_ace:
                S_0 = np.array([players_sum, dealer_second_card, 1], dtype=int)
            else:
                S_0 = np.array([players_sum, dealer_second_card, 0], dtype=int)

            self.MDP.dealer_hidden_sum = dealer_second_card

            if random.random() <= policy_b[S_0[0]-12, S_0[1]-1, S_0[2], 0]:
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

                if C_S_A > 0:
                    Q_S_A = self.action_value_fun[state[0]-12, state[1]-1, state[2], action]
                    self.action_value_fun[state[0]-12, state[1]-1, state[2], action] = (
                        Q_S_A + (W / C_S_A) * (G - Q_S_A)
                    )
                b_A_S = policy_b[state[0] - 12, state[1] - 1, state[2], action]
                pi_A_S = self.policy_pi[state[0] - 12, state[1] - 1, state[2], action]
                W = W*(pi_A_S/b_A_S) 