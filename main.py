import numpy as np

from MDP import MDP
from MCES import MCES

MDP_obj = MDP()

policy = np.full((10, 10, 2, 2), 0.5)
action_value_fun  = np.full((10, 10, 2, 2), 0)

MCES_obj = MCES(MDP_obj, 1, 5000, policy, action_value_fun)
optimal_policy_ES = MCES_obj.MCES_model()
print(optimal_policy_ES)