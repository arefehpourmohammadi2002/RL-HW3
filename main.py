import numpy as np

from MDP import MDP
from MCES import MCES
from on_policy_MC import OnPoicyMC
from off_policy_MC import OffPoicyMC
import graph

number_of_episodes = 500000
policy = np.full((10, 10, 2, 2), 0.5)
action_value_fun  = np.full((10, 10, 2, 2), 0)

# MDP_obj = MDP()
# MCES_obj = MCES(MDP_obj, 1, number_of_episodes, policy, action_value_fun)
# MCES_obj.MCES_model()
# for i in range(0, 10):
#     for j in range(0, 10):
#         print(i+12, j+1, 0, MCES_obj.policy_pi[i][j][0])

# for i in range(0, 10):
#     for j in range(0, 10): 
#         print(i+12, j+1, 1, MCES_obj.policy_pi[i][j][1]) 

# graph.create_blackjack_policy_chart(MCES_obj.policy_pi)


# print("---------------------------- on policy -------------------------")


# MDP_on_policy_obj = MDP()
# on_policy_obj = OnPoicyMC(MDP_on_policy_obj, 1, number_of_episodes, policy, action_value_fun, 0.05)
# on_policy_obj.on_poicy_MC_model()

# for i in range(0, 10):
#     for j in range(0, 10):
#         print(i+12, j+1, 0, on_policy_obj.policy_pi[i][j][0]) 
#         print(i+12, j+1, 1, on_policy_obj.policy_pi[i][j][1]) 


print("---------------------------- off policy -------------------------")

MDP_off_policy_obj = MDP()
MDP_off_policy_obj = MDP()
off_policy_obj = OffPoicyMC(MDP_off_policy_obj, 1, number_of_episodes, policy, action_value_fun)
off_policy_obj.off_poicy_MC_model()

for i in range(0, 10):
    for j in range(0, 10):
        print(i+12, j+1, 0, off_policy_obj.policy_pi[i][j][0])

for i in range(0, 10):
    for j in range(0, 10): 
        print(i+12, j+1, 1, off_policy_obj.policy_pi[i][j][1]) 


