import os
import numpy as np
import matplotlib.pyplot as plt
from envball_gnn import BallGNNEnvironment

def build_eval_agent(model_path, target_pos):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    from train_gnn_ball import load_agent_from_checkpoint
    agent = load_agent_from_checkpoint(model_path)
    agent.target_pos = np.array(target_pos, dtype=np.float32)
    return agent

def test_gnn_ball(model_path, num_tests=10, max_steps=500, plot_path="gnn_trajectories.png", show_plot=False):
    env = BallGNNEnvironment(max_steps=max_steps)
    agent = build_eval_agent(model_path, env.target_pos)
    all_trajectories = []
    print(f"Testing GNN agent for {num_tests} runs...")
    for test_idx in range(num_tests):
        obs = env.reset()
        trajectory = [obs['ball_pos'].copy()]
        total_reward = 0
        steps_taken = 0
        success = False
        for _ in range(max_steps):
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps_taken += 1
            trajectory.append(obs['ball_pos'].copy())
            if info.get('success', False):
                success = True
                break
            if done:
                break
        print(f"Test {test_idx+1}: Success={success}, Steps={steps_taken}, Reward={total_reward:.2f}")
        all_trajectories.append(np.array(trajectory))
    plot_trajectories(all_trajectories, env.target_pos, plot_path, show_plot)

def plot_trajectories(trajectories, target_pos, plot_path, show_plot):
    plt.figure(figsize=(6,6))
    for traj in trajectories:
        plt.plot(traj[:,0], traj[:,1], marker='o')
    plt.scatter([target_pos[0]], [target_pos[1]], c='red', marker='*', s=200, label='Target')
    plt.title('GNN Ball Trajectories')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    if show_plot:
        plt.show()
    plt.close()

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "models", "gnn_sac_best.npz")
    test_gnn_ball(model_path)
