#!/usr/bin/env python3

import os
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import optuna
import time

from algorithms.gnn.env_gnn import MultiRobotGNNEnv
from algorithms.gnn.gnn_agent import GNNRLAgent, GNNReplayBuffer
from algorithms.gnn.compute_features import make_graph_features_prescaled, FeatureConfig

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
_RESULTS_DIR = os.path.join(_ROOT_DIR, "experiments", "results")


def train_gnn(
    xml_path: str,
    save_path: str,
    total_steps: int = 10_000_000,
    batch_size: int = 128,
    max_ep_steps: int = 2500,
    seed: int = 42,
    eval_every: int = 20_000,
    log_path: str = None,
    plot_path: str = None,
    show_plot: bool = False,
):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'tb_logs', time.strftime('%Y%m%d-%H%M%S')))

    save_path = os.path.expanduser(os.path.expandvars(str(save_path)))
    if log_path is None:
        log_path = save_path + "_train_log.npz"
    if plot_path is None:
        plot_path = save_path + "_train.png"

    print("[INFO] 初始化环境...")
    env = MultiRobotGNNEnv(xml_path=xml_path, max_steps=max_ep_steps, seed=seed)
    obs = env.reset()

    print("[INFO] 环境初始化完成，开始特征提取...")
    # 生成graph特征，推断维度
    feature_config = FeatureConfig(robot_relative_tip_features=True, robot_base_features=True)
    g = make_graph_features_prescaled(obs, feature_config)
    node_dim = g.nodes.shape[1]
    edge_dim = g.edges.shape[1]
    print(f"[DEBUG] g.globals.shape: {g.globals.shape}")
    global_dim = g.globals.shape[1]
    print(f"[DEBUG] global_dim: {global_dim}")
    action_dim = env.model.nu

    print(f"[INFO] 特征维度: node_dim={node_dim}, edge_dim={edge_dim}, global_dim={global_dim}, action_dim={action_dim}")
    agent = GNNRLAgent(
        input_node_dim=node_dim,
        input_edge_dim=edge_dim,
        input_global_dim=global_dim,  # Fixed: match actual global feature dimension
        node_dim=512,
        edge_dim=256,
        global_dim=global_dim,
        node_hidden_dims=[512]*7,
        edge_hidden_dims=[256,6],
        action_dim=action_dim,
        rounds=5,
        use_attention=True,
        lr=5e-5,
        gamma=0.94,
        tau=4e-5,
    )
    replay = GNNReplayBuffer(max_size=20_000_000)

    episode_return = []
    episode_len = []
    eval_step = []
    eval_mean_return = []
    print("[INFO] Agent 初始化完成，开始训练循环...")

    loss_log = []
    for t in range(int(total_steps)):
        obs = env.reset()
        total_reward = 0.0
        episode_transitions = []  # 存储本episode所有transition
        for step in range(max_ep_steps):
            g = make_graph_features_prescaled(obs, feature_config)
            nodes = torch.tensor(np.array(g.nodes), dtype=torch.float32)
            edges = torch.tensor(np.array(g.edges), dtype=torch.float32)
            globals_ = torch.tensor(np.array(g.globals.squeeze(0)), dtype=torch.float32)
            senders = torch.tensor(np.array(g.senders), dtype=torch.long)
            receivers = torch.tensor(np.array(g.receivers), dtype=torch.long)
            robot_node_indices = torch.arange(nodes.shape[0], dtype=torch.long)
            action = agent.select_action(nodes, edges, globals_, senders, receivers, robot_node_indices)
            next_obs, reward, done, info = env.step(action)
            g_next = make_graph_features_prescaled(next_obs, feature_config)
            transition = (
                {'nodes': g.nodes, 'edges': g.edges, 'globals': g.globals, 'senders': g.senders, 'receivers': g.receivers, 'robot_node_indices': np.arange(1)},
                action,
                reward,
                {'nodes': g_next.nodes, 'edges': g_next.edges, 'globals': g_next.globals, 'senders': g_next.senders, 'receivers': g_next.receivers, 'robot_node_indices': np.arange(1)},
                float(done),
                obs,  # 额外存储原始obs用于HER
                next_obs
            )
            episode_transitions.append(transition)
            replay.add(*transition[:5])  # 只存前5项
            total_reward += reward
            obs = next_obs
            if done:
                break
            if step % 100 == 0:
                print(f"[DEBUG] t={t+1} step={step} reward={reward:.3f}")
                writer.add_scalar('train/reward', reward, t * max_ep_steps + step)
            # 更新
            if len(replay) > batch_size:
                loss = agent.update(replay, batch_size=batch_size)
                loss_log.append(loss)
                if step % 100 == 0:
                    print(f"[LOSS] t={t+1} step={step} q1_loss={loss['q1_loss']:.4f} q2_loss={loss['q2_loss']:.4f} policy_loss={loss['policy_loss']:.4f}")
                    writer.add_scalar('train/q1_loss', loss['q1_loss'], t * max_ep_steps + step)
                    writer.add_scalar('train/q2_loss', loss['q2_loss'], t * max_ep_steps + step)
                    writer.add_scalar('train/policy_loss', loss['policy_loss'], t * max_ep_steps + step)

        # --- HER relabeling ---
        # 采样若干her_goal（如4个），对episode内每个transition重构reward/done并追加到replay
        her_k = 4
        if len(episode_transitions) > 0:
            episode_length = len(episode_transitions)
            for i, (obs_dict, action, reward, next_obs_dict, done, obs_raw, next_obs_raw) in enumerate(episode_transitions):
                # 随机采样her_k个future step作为her_goal
                future_idx = np.random.choice(np.arange(i, episode_length), size=min(her_k, episode_length - i), replace=False)
                for fidx in future_idx:
                    # 取future step的tip位置作为her_goal
                    her_goal = episode_transitions[fidx][6]['robots']['tip_relative_poses'][0, :3, 3].copy()
                    # 修改obs/next_obs的goal为her_goal
                    obs_her = obs_raw.copy()
                    next_obs_her = next_obs_raw.copy()
                    obs_her = dict(obs_her)
                    next_obs_her = dict(next_obs_her)
                    obs_her['targets'] = dict(obs_her['targets'])
                    next_obs_her['targets'] = dict(next_obs_her['targets'])
                    # 修改goal
                    obs_her['targets']['poses'] = obs_her['targets']['poses'].copy()
                    next_obs_her['targets']['poses'] = next_obs_her['targets']['poses'].copy()
                    obs_her['targets']['poses'][0, :3, 3] = her_goal
                    next_obs_her['targets']['poses'][0, :3, 3] = her_goal
                    # 重新提取graph特征
                    g_her = make_graph_features_prescaled(obs_her, feature_config)
                    g_next_her = make_graph_features_prescaled(next_obs_her, feature_config)
                    # 计算新的reward和done
                    tip = obs_her['robots']['tip_relative_poses'][0, :3, 3]
                    dist = np.linalg.norm(tip - her_goal)
                    her_reward = -dist
                    her_done = float(dist < env.reach_tol)
                    # 存入replay
                    replay.add(
                        {'nodes': g_her.nodes, 'edges': g_her.edges, 'globals': g_her.globals, 'senders': g_her.senders, 'receivers': g_her.receivers, 'robot_node_indices': np.arange(1)},
                        action,
                        her_reward,
                        {'nodes': g_next_her.nodes, 'edges': g_next_her.edges, 'globals': g_next_her.globals, 'senders': g_next_her.senders, 'receivers': g_next_her.receivers, 'robot_node_indices': np.arange(1)},
                        her_done
                    )

        episode_return.append(float(total_reward))
        episode_len.append(int(step + 1))
        # 定期保存模型
        model_save_interval = 500  # 每5万步保存一次
        if (t + 1) % model_save_interval == 0:
            model_path = os.path.join(save_path, f"gnn_model.pth")
            save_model(model_path)
            
        # 评估流程
        if (t + 1) % int(eval_every) == 0:
            eval_returns = []
            for _ in range(5):
                obs = env.reset()
                eval_total = 0.0
                for _ in range(max_ep_steps):
                    g = make_graph_features_prescaled(obs, feature_config)
                    nodes = torch.tensor(np.array(g.nodes), dtype=torch.float32)
                    edges = torch.tensor(np.array(g.edges), dtype=torch.float32)
                    globals_ = torch.tensor(np.array(g.globals.squeeze(0)), dtype=torch.float32)
                    senders = torch.tensor(np.array(g.senders), dtype=torch.long)
                    receivers = torch.tensor(np.array(g.receivers), dtype=torch.long)
                    robot_node_indices = torch.arange(nodes.shape[0], dtype=torch.long)
                    action = agent.select_action(nodes, edges, globals_, senders, receivers, robot_node_indices, evaluate=True)
                    obs, reward, done, _ = env.step(action)
                    eval_total += reward
                    if done:
                        break
                eval_returns.append(eval_total)
            eval_step.append(int(t + 1))
            eval_mean_return.append(np.mean(eval_returns))
            print(f"[EVAL] step={t+1} mean_return={np.mean(eval_returns):.4f}")
            writer.add_scalar('eval/mean_return', np.mean(eval_returns), t * max_ep_steps)
    # 保存训练日志
    log_path = os.path.join(save_path, "gnn_training_log.npz")
    np.savez(log_path,
        episode_return=np.asarray(episode_return, dtype=np.float32),
        episode_len=np.asarray(episode_len, dtype=np.int32),
        eval_step=np.asarray(eval_step, dtype=np.int32),
        eval_mean_return=np.asarray(eval_mean_return, dtype=np.float32),
        loss_log=np.array(loss_log, dtype=object)
    )
    print(f"[LOG] GNN training log saved: {log_path}")
    print("[INFO] 训练完成。")
    writer.close()

    # 定期保存模型
    def save_model(path: str):
        model_dir = os.path.dirname(path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save({
            'policy_net': agent.policy_net.state_dict(),
            'q_net1': agent.q_net1.state_dict(),
            'q_net2': agent.q_net2.state_dict(),
            'target_q_net1': agent.target_q_net1.state_dict(),
            'target_q_net2': agent.target_q_net2.state_dict(),
        }, path)
        print(f"[MODEL] Saved GNN checkpoint: {path}")
        
# Optuna自动调参入口
def optuna_objective(trial):
    # 搜索空间示例，可根据实际需求扩展
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.99)
    tau = trial.suggest_float('tau', 1e-5, 1e-3, log=True)
    node_dim = trial.suggest_categorical('node_dim', [256, 512, 1024])
    edge_dim = trial.suggest_categorical('edge_dim', [128, 256, 512])
    rounds = trial.suggest_int('rounds', 3, 7)
    use_attention = trial.suggest_categorical('use_attention', [True, False])
    # 训练参数
    xml_path = os.path.join(_ROOT_DIR, "algorithms", "gnn", "urdf", "t12a_14_clear.xml")
    save_path = os.path.join(_RESULTS_DIR, "gnn_optuna")
    # agent参数传递
    agent_params = dict(
        input_node_dim=62,
        input_edge_dim=18,
        input_global_dim=2,
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=2,
        node_hidden_dims=[node_dim]*7,
        edge_hidden_dims=[edge_dim,6],
        action_dim=6,
        rounds=rounds,
        use_attention=use_attention,
        lr=lr,
        gamma=gamma,
        tau=tau,
    )
    # 只训练少量step用于评测
    total_steps = 2000
    batch_size = 128
    max_ep_steps = 500
    seed = 42
    eval_every = 500
    # 训练并返回评估结果
    env = MultiRobotGNNEnv(xml_path=xml_path, max_steps=max_ep_steps, seed=seed)
    obs = env.reset()
    feature_config = FeatureConfig(robot_relative_tip_features=True, robot_base_features=True)
    g = make_graph_features_prescaled(obs, feature_config)
    agent = GNNRLAgent(**agent_params)
    replay = GNNReplayBuffer(max_size=100000)
    eval_mean_return = []
    for t in range(total_steps):
        obs = env.reset()
        total_reward = 0.0
        loss_log = []
        for step in range(max_ep_steps):
            g = make_graph_features_prescaled(obs, feature_config)
            nodes = torch.tensor(np.array(g.nodes), dtype=torch.float32)
            edges = torch.tensor(np.array(g.edges), dtype=torch.float32)
            globals_ = torch.tensor(np.array(g.globals.squeeze(0)), dtype=torch.float32)
            senders = torch.tensor(np.array(g.senders), dtype=torch.long)
            receivers = torch.tensor(np.array(g.receivers), dtype=torch.long)
            robot_node_indices = torch.arange(nodes.shape[0], dtype=torch.long)
            action = agent.select_action(nodes, edges, globals_, senders, receivers, robot_node_indices)
            next_obs, reward, done, info = env.step(action)
            g_next = make_graph_features_prescaled(next_obs, feature_config)
            replay.add({'nodes': g.nodes, 'edges': g.edges, 'globals': g.globals, 'senders': g.senders, 'receivers': g.receivers, 'robot_node_indices': np.arange(1)}, action, reward, {'nodes': g_next.nodes, 'edges': g_next.edges, 'globals': g_next.globals, 'senders': g_next.senders, 'receivers': g_next.receivers, 'robot_node_indices': np.arange(1)}, float(done))
            total_reward += reward
            obs = next_obs
            if done:
                print(f"[TRIAL LOG] trial={trial.number} t={t+1} step={step} episode_reward={total_reward:.3f}")
                break
            if len(replay) > batch_size:
                loss = agent.update(replay, batch_size=batch_size)
                loss_log.append(loss)
                if step % 100 == 0:
                    print(f"[TRIAL LOSS] trial={trial.number} t={t+1} step={step} q1_loss={loss['q1_loss']:.4f} q2_loss={loss['q2_loss']:.4f} policy_loss={loss['policy_loss']:.4f}")
        # 评估
        if (t + 1) % eval_every == 0:
            eval_returns = []
            for _ in range(3):
                obs = env.reset()
                eval_total = 0.0
                for _ in range(max_ep_steps):
                    g = make_graph_features_prescaled(obs, feature_config)
                    nodes = torch.tensor(np.array(g.nodes), dtype=torch.float32)
                    edges = torch.tensor(np.array(g.edges), dtype=torch.float32)
                    globals_ = torch.tensor(np.array(g.globals.squeeze(0)), dtype=torch.float32)
                    senders = torch.tensor(np.array(g.senders), dtype=torch.long)
                    receivers = torch.tensor(np.array(g.receivers), dtype=torch.long)
                    robot_node_indices = torch.arange(nodes.shape[0], dtype=torch.long)
                    action = agent.select_action(nodes, edges, globals_, senders, receivers, robot_node_indices, evaluate=True)
                    obs, reward, done, _ = env.step(action)
                    eval_total += reward
                    if done:
                        break
                eval_returns.append(eval_total)
            mean_eval = np.mean(eval_returns)
            eval_mean_return.append(mean_eval)
            print(f"[TRIAL EVAL] trial={trial.number} t={t+1} mean_eval_return={mean_eval:.4f}")
    # 返回最后一次评估均值
    return eval_mean_return[-1] if eval_mean_return else -999

def run_optuna():
    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=20)
    print('Best trial:', study.best_trial.params)

if __name__ == "__main__":
    # xml_path = os.path.join(_ROOT_DIR, "algorithms", "gnn", "urdf", "t12a_14_clear.xml")
    # save_path = os.path.join(_RESULTS_DIR, "gnn")
    # train_gnn(xml_path=xml_path, save_path=save_path)

    run_optuna()
