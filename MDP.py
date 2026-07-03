import numpy as np
import random
class MDP:
    def __init__(self):

        self.dealer_hidden_sum = 0
        self.dealer_usable_ace = 0
        self.states = np.zeros((200, 3), dtype= int)
        index = 0
        for self_cards in range(0,10):
            for dealer in range(1, 11):

                self.states[index] = [12 + self_cards, dealer, 1]
                self.states[index+1] = [12 + self_cards, dealer, 0]
                index += 2

    def state_transition(self, state, action):
        next_state = state.copy()

        if action == 0: # action is hit
            next_state[0] += min(random.randint(1, 13), 10)
            if next_state[2] == 1 and next_state[0] > 21:
                next_state[0] -= 10
                next_state[2] = 0
        
        if action == 1: #action is stick

            while self.dealer_hidden_sum < 17:
                random_card = min(random.randint(1, 13), 10)
                if random_card == 1 and self.dealer_hidden_sum < 11:
                    self.dealer_hidden_sum += 11
                    self.dealer_usable_ace = 1
                else:
                    self.dealer_hidden_sum += random_card

                if self.dealer_hidden_sum > 21 and self.dealer_usable_ace == 1:
                    self.dealer_hidden_sum -= 10
                    self.dealer_usable_ace = 0
  
        return next_state
                
    def reward(self, state):
        if state[0] > 21:
            return -1
        
        elif self.dealer_hidden_sum > 21:
            return 1
        
        elif 17 <= self.dealer_hidden_sum <= 21:
            if state[0] > self.dealer_hidden_sum:
                return 1
            elif state[0] < self.dealer_hidden_sum:
                return -1
            else:
                return 0
        
        elif state[0] == 21:
            return 1
    
    def is_finished(self, state, action):
        if state[0] > 21 or self.dealer_hidden_sum > 21:
            return True
        elif 17 <= self.dealer_hidden_sum and self.dealer_hidden_sum <= 21 and action == 1:
            return True
        else:
            return False
            
    
        

g = MDP()