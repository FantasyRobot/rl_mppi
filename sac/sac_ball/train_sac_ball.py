#!/usr/bin/env python3

import os
import numpy as np
import torch
import argparse
from env.envball_utils import BallEnvironment

# Import SAC components from sac_utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sac_utils import SACAgent, ReplayBuffer

def train_sac_ball(data_dir, epochs, batch_size, save_path, target_pos=None):
    """
    Train SAC agent on ball environment data.
    
    Args:
        data_dir (str): Directory containing training data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        save_path (str): Path to save the trained model
        target_pos (list): Target position for the environment
    """
    # Load training data
    print("Loading training data...")
    states = np.load(os.path.join(data_dir, "states.npy"))
    actions = np.load(os.path.join(data_dir, "actions.npy"))
    next_states = np.load(os.path.join(data_dir, "next_states.npy"))
    rewards = np.load(os.path.join(data_dir, "rewards.npy"))
    dones = np.load(os.path.join(data_dir, "dones.npy"))
    
    print(f"Data loaded successfully!")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Dones shape: {dones.shape}")
    
    # Create environment to get dimensions
    env = BallEnvironment(target_pos=target_pos, max_steps=100)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Create replay buffer and populate with data
    print("Populating replay buffer...")
    replay_buffer = ReplayBuffer(max_size=len(states))
    
    for i in range(len(states)):
        replay_buffer.add(
            state=states[i],
            action=actions[i],
            reward=rewards[i],
            next_state=next_states[i],
            done=dones[i]
        )
    
    print(f"Replay buffer populated with {len(replay_buffer.buffer)} transitions")
    
    # Initialize SAC agent
    print("Initializing SAC agent...")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=True
    )
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Update agent
        update_results = agent.update(replay_buffer, batch_size=batch_size)
        
        # Print training progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            q1_loss = update_results["q1_loss"]
            q2_loss = update_results["q2_loss"]
            policy_loss = update_results["policy_loss"]
            alpha = update_results["alpha"]
            
            total_loss = q1_loss + q2_loss + policy_loss
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Q1 Loss: {q1_loss:.4f}")
            print(f"  Q2 Loss: {q2_loss:.4f}")
            print(f"  Policy Loss: {policy_loss:.4f}")
            print(f"  Total Loss: {total_loss:.4f}")
            print(f"  Alpha: {alpha:.4f}")
            
            # Save best model
            if total_loss < best_loss:
                best_loss = total_loss
                save_model(agent, save_path)
                print(f"  Saved best model with loss {best_loss:.4f}")
    
    # Save final model
    save_model(agent, save_path.replace(".pth", "_final.pth"))
    print(f"Training completed! Final model saved to {save_path.replace('.pth', '_final.pth')}")
    return agent

def save_model(agent, save_path):
    """Save the trained SAC agent model."""
    model_dir = os.path.dirname(save_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    torch.save({
        'policy_state_dict': agent.policy_net.state_dict(),
        'q1_state_dict': agent.q_net1.state_dict(),
        'q2_state_dict': agent.q_net2.state_dict(),
        'target_q1_state_dict': agent.target_q_net1.state_dict(),
        'target_q2_state_dict': agent.target_q_net2.state_dict(),
        'alpha': agent.alpha,
        'log_alpha': agent.log_alpha if agent.auto_entropy_tuning else None
    }, save_path)
    
    print(f"Model saved to {save_path}")

def load_model(save_path, state_dim, action_dim):
    """Load a trained SAC agent model."""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sac_utils import SACAgent
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        learning_rate=3e-4,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=True
    )
    
    checkpoint = torch.load(save_path)
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.q_net1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q_net2.load_state_dict(checkpoint['q2_state_dict'])
    agent.target_q_net1.load_state_dict(checkpoint['target_q1_state_dict'])
    agent.target_q_net2.load_state_dict(checkpoint['target_q2_state_dict'])
    agent.alpha = checkpoint['alpha']
    if agent.auto_entropy_tuning:
        agent.log_alpha = checkpoint['log_alpha']
    
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent for ball environment")
    parser.add_argument("--data_dir", type=str, default="train_data", help="Directory containing training data")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--save_path", type=str, default="sac_ball_model.pth", help="Path to save the trained model")
    parser.add_argument("--target_x", type=float, default=3.0, help="Target x position")
    parser.add_argument("--target_y", type=float, default=3.0, help="Target y position")
    
    args = parser.parse_args()
    
    target_pos = [args.target_x, args.target_y]
    
    # Train the model
    train_sac_ball(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        target_pos=target_pos
    )