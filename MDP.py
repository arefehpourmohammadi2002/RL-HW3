import numpy as np
import random
class MDP:
    def __init__(self):
        self.states = np.zeros((200, 3), dtype= int)
        index = 0
        for self_cards in range(0,10):
            for dealer in range(0, 10):

                self.states[index] = [12 + self_cards, dealer, 1]
                self.states[index+1] = [12 + self_cards, dealer, 0]
                index += 2
    def state_transition(state, action):
        next_state = state.copy()
        if action == 0: # action is hit
            next_state[0] += random.randint(1, 10)
            
            

    def reward(self):
        pass 

g = MDP()