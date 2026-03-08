#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from contextlib import contextmanager

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from env.envmujoco_t12a_14 import T12A14MuJoCoEnv
from algorithms.gnn.gnn_agent import GNNRLAgent
from algorithms.gnn.compute_features import make_graph_features_prescaled, FeatureConfig

def load_gnn_policy(model_path, node_dim, edge_dim, global_dim, action_dim):
    agent = GNNRLAgent(
        input_node_dim=node_dim,
        input_edge_dim=edge_dim,
        input_global_dim=global_dim,
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
    ckpt = torch.load(model_path, map_location="cpu")
    agent.policy_net.load_state_dict(ckpt['policy_net'])
    agent.q_net1.load_state_dict(ckpt['q_net1'])
    agent.q_net2.load_state_dict(ckpt['q_net2'])
    agent.target_q_net1.load_state_dict(ckpt['target_q_net1'])
    agent.target_q_net2.load_state_dict(ckpt['target_q_net2'])
    return agent

def test_gnn_t12a_14(model_path, xml_path, max_steps=2500, action_repeat=5, init_qpos=None):
    env = T12A14MuJoCoEnv(
        xml_path=xml_path,
        max_steps=max_steps,
        action_repeat=action_repeat,
    )
    obs = env.reset(init_qpos=init_qpos)
    # 提取特征维度
    feature_config = FeatureConfig(robot_relative_tip_features=True, robot_base_features=True)
    g = make_graph_features_prescaled(obs, feature_config)
    node_dim = g.nodes.shape[1]
    edge_dim = g.edges.shape[1]
    global_dim = g.globals.shape[1]
    action_dim = env.model.nu
    agent = load_gnn_policy(model_path, node_dim, edge_dim, global_dim, action_dim)
    eef_traj = []
    dist_traj = []
    for _ in range(max_steps):
        g = make_graph_features_prescaled(obs, feature_config)
        nodes = torch.tensor(np.array(g.nodes), dtype=torch.float32)
        edges = torch.tensor(np.array(g.edges), dtype=torch.float32)
        globals_ = torch.tensor(np.array(g.globals.squeeze(0)), dtype=torch.float32)
        senders = torch.tensor(np.array(g.senders), dtype=torch.long)
        receivers = torch.tensor(np.array(g.receivers), dtype=torch.long)
        robot_node_indices = torch.arange(nodes.shape[0], dtype=torch.long)
        action = agent.select_action(nodes, edges, globals_, senders, receivers, robot_node_indices, evaluate=True)
        obs, _r, done, info = env.step(action)
        eef_traj.append(np.asarray(info["eef"], dtype=np.float32))
        dist_traj.append(float(info["dist"]))
        if done:
            break
    return {
        "eef": np.asarray(eef_traj, dtype=np.float32),
        "dist": np.asarray(dist_traj, dtype=np.float32),
        "success": bool(info.get("success", False)),
        "final_dist": float(dist_traj[-1]) if dist_traj else float("inf"),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="GNN模型路径")
    parser.add_argument("--xml", type=str, required=True, help="MuJoCo XML路径")
    parser.add_argument("--max_steps", type=int, default=2500)
    parser.add_argument("--action_repeat", type=int, default=5)
    args = parser.parse_args()
    result = test_gnn_t12a_14(args.model, args.xml, args.max_steps, args.action_repeat)
    print("Test result:", result)
