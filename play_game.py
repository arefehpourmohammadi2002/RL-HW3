import random

def play_game(MDP, policy, first_state, first_action):
    
    episode_steps = []
    current_state = first_state
    current_action = first_action

    dealer_other_card = random_card = min(random.randint(1, 13), 10)
    if ((current_state[1] == 1 and dealer_other_card == 10)
        or current_state[1] == 10 and dealer_other_card == 1):
        first_step = {"state": current_state, "action": current_action, "reward ": -1}
        episode_steps.append(first_step)
        return episode_steps
    
    if current_state[0] == 21:
        first_step = {"state": current_state, "action": current_action, "reward ": 1}
        episode_steps.append(first_step)
        return episode_steps
    
    if dealer_other_card == 1: 
        MDP.dealer_hidden_sum = current_state[1] + 11
        MDP.dealer_usable_ace = 1
    
    elif current_state[1] == 1:
        MDP.dealer_hidden_sum = dealer_other_card + 11
        MDP.dealer_usable_ace = 1
    
    else:
        MDP.dealer_hidden_sum = current_state[1] + dealer_other_card
        MDP.dealer_usable_ace = 0

    first_step = {"state": current_state, "action": current_action}
    episode_steps.append(first_step)


    step = 1
    while True:
        next_state = MDP.state_transition(current_state, current_action)

        if not MDP.is_finished(next_state, current_action):
            
            action_prob = policy[next_state[0] - 12, next_state[1] - 1, next_state[2]]
            
            rand = random.random()
            action = 1
            if rand <= action_prob[0]:
               action = 0

            this_step = {"state": next_state, "action": action, "reward":0}
            episode_steps.append(this_step)

            current_state = next_state
            current_action = action

        else:
            reward = MDP.reward(next_state)
            this_step = {"state": next_state, "reward":reward}
            episode_steps.append(this_step)
            break

        step += 1
        
    return episode_steps
        
