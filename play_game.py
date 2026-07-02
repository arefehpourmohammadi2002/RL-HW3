import random

def play_game(MDP, policy, first_state, first_action):
    
    episode_steps = []
    current_state = first_state
    current_action = first_action
    first_step = {"state": current_state, "action": current_action}
    episode_steps.append(first_step)


    step = 1
    while True:
        next_state = MDP.state_transiction(current_state, current_action)

        if not MDP.is_finished(next_state):

            action_prob = policy.state_action(next_state)
            episode_steps[step]["state"] = next_state
            
            rand = random.random()
            if rand <= action_prob[0]:
                
                episode_steps[step]["action"] = 0
                
            else:
                episode_steps[step]["action"] = 0

            episode_steps[step]["reward"] = 0

        else:
            reward = MDP.reward(next_state)
            episode_steps[step+1]["reward"] = reward
            break
        
        step += 1
        
    return episode_steps
        
