import numpy as np

from MDP import MDP
from MC_control_policies.MCES import MCES
from MC_control_policies.on_policy_MC import OnPoicyMC
from MC_control_policies.off_policy_MC import OffPoicyMC
import MC_control_policies.graph as graph

from MC_prediction_policies.FV_MC_prediction import FVMCPrediction
from MC_prediction_policies.off_policy_MC_pred import OffPoicyMCPred
import MC_prediction_policies.graph_prediction as gp

number_of_episodes = 50000000
action_value_fun  = np.zeros((10, 10, 2, 2))
policy = np.zeros((10, 10, 2, 2))
for s in range(10):
    for d in range(10):
        for ace in range(2):
            policy[s, d, ace] = [0, 1] if s + 12 >= 20 else [1, 0]

print("---------------------------- MC Controles -------------------------")
print("---------------------------- MCES -------------------------")

MDP_obj = MDP()
MCES_obj = MCES(MDP_obj, 1, number_of_episodes, policy.copy(), action_value_fun.copy())
MCES_obj.MCES_model()

graph.plot_blackjack_results(MCES_obj.policy_pi, MCES_obj.action_value_fun,
                            save_path="MC_control_policies/MCES.png")


print("---------------------------- on policy -------------------------")


MDP_on_policy_obj = MDP()
on_policy_obj = OnPoicyMC(MDP_on_policy_obj, 1, number_of_episodes, policy.copy(), action_value_fun.copy(), 0.05)
on_policy_obj.on_poicy_MC_model()

graph.plot_blackjack_results(on_policy_obj.policy_pi, on_policy_obj.action_value_fun,
                            save_path="MC_control_policies/on_policy.png")


print("---------------------------- off policy -------------------------")

MDP_off_policy_obj = MDP()
MDP_off_policy_obj = MDP()
off_policy_obj = OffPoicyMC(MDP_off_policy_obj, 1, number_of_episodes, policy.copy(), action_value_fun.copy())
off_policy_obj.off_poicy_MC_model()

graph.plot_blackjack_results(off_policy_obj.policy_pi, off_policy_obj.action_value_fun,
                            save_path="MC_control_policies/off_policy.png")

print("---------------------------- MC Prediction -------------------------")
print("---------------------------- FV Prediction -------------------------")

MDP_FV_prediction = MDP()
FVMCP_obj_bad = FVMCPrediction(MDP_FV_prediction, 1, 10000)
FVMCP_obj_bad.run()

MDP_FV_prediction_good = MDP()
FVMCP_obj_good = FVMCPrediction(MDP_FV_prediction_good, 1, 500000)
FVMCP_obj_good.run()

gp.plot_blackjack_results(FVMCP_obj_bad.V,FVMCP_obj_good.V, "MC_prediction_policies/FVMC_Predict.png")

print("---------------------------- Off-policy Prediction -------------------------")


MDP_off_prediction = MDP()
offMCP_obj_bad = OffPoicyMCPred(MDP_off_prediction, 1, 10000, action_value_fun)
offMCP_obj_bad.run()

v_bad = np.zeros((10, 10, 2))
for i in range(0, 8):
    v_bad[i,:,:] = offMCP_obj_bad.action_value_fun[i,:,:,0]

v_bad[8,:,:] = offMCP_obj_bad.action_value_fun[8,:,:,1]
v_bad[9,:,:] = offMCP_obj_bad.action_value_fun[9,:,:,1]

MDP_off_prediction_good = MDP()
offMCP_obj_good = OffPoicyMCPred(MDP_off_prediction_good, 1, 500000, action_value_fun)
offMCP_obj_good.run()

v_good = np.zeros((10, 10, 2))
for i in range(0, 8):
    v_good[i,:,:] = offMCP_obj_good.action_value_fun[i,:,:,0]
    
v_good[8,:,:] = offMCP_obj_good.action_value_fun[8,:,:,1]
v_good[9,:,:] = offMCP_obj_good.action_value_fun[9,:,:,1]

gp.plot_blackjack_results(v_bad, v_good, "MC_prediction_policies/offMC_Predict.png")
