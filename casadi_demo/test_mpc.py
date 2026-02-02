import numpy as np
from generate_training_data import generate_controlled_actions

# Test with small number of steps
current_state = np.zeros(12)
target_state = np.array([1, 2, 3, 4, 5, 6])  # Very close to target
num_steps = 10
action_dim = 6

print('Testing target reaching functionality...')
actions, actual_steps = generate_controlled_actions(current_state, target_state, num_steps, action_dim)
print(f'Generated {len(actions)} actions')
print(f'Actual MPC-calculated steps: {actual_steps}')
print(f'First few actions: {actions[:3]}')
print(f'Last few actions: {actions[-3:]}')

# Check if remaining actions are filled with the same value
if len(actions) > 3:
    all_same = np.allclose(actions[3:], actions[3])
    print(f'All remaining actions are the same: {all_same}')
    
    if all_same:
        print(f'Action used for filling: {actions[3]}')