import argparse
from dataclasses import asdict, dataclass
import os
import sys

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envball_hrsga import HRSGABallEnvironment
from hrsga_ball_model import HRSGAPolicyNetwork, HRSGAQNetwork, StructuredReplayBuffer, snapshot_to_model_inputs


@dataclass
class EpisodeStats:
    total_reward: float = 0.0
    success: int = 0
    completed_tasks: int = 0
    deadline_satisfaction: float = 0.0
    missed_deadlines: int = 0
    collisions: int = 0
    mean_task_distance: float = 0.0
    min_pair_distance: float = 0.0
    episode_length: int = 0


def _task_status_counts(snapshot):
    completed = 0
    completed_before_deadline = 0
    missed = 0
    for task in snapshot["tasks"]:
        if task["completed"]:
            completed += 1
            if task["completed_step"] <= task["deadline_step"]:
                completed_before_deadline += 1
        elif snapshot["step"] > task["deadline_step"]:
            missed += 1
    return completed, completed_before_deadline, missed


def _mean_active_task_distance(snapshot):
    distances = []
    for agent in snapshot["agents"]:
        task = snapshot["tasks"][agent["task_id"]]
        dx = agent["pos"][0] - task["pos"][0]
        dy = agent["pos"][1] - task["pos"][1]
        distances.append(float(np.hypot(dx, dy)))
    return float(np.mean(distances)) if distances else 0.0


def compute_reward(prev_snapshot, next_snapshot, info, done):
    prev_completed, prev_deadline_ok, prev_missed = _task_status_counts(prev_snapshot)
    next_completed, next_deadline_ok, next_missed = _task_status_counts(next_snapshot)
    prev_distance = _mean_active_task_distance(prev_snapshot)
    next_distance = _mean_active_task_distance(next_snapshot)

    reward = 0.6 * (prev_distance - next_distance)
    reward += 2.5 * float(next_completed - prev_completed)
    reward += 1.2 * float(next_deadline_ok - prev_deadline_ok)
    reward -= 0.35 * float(info.get("collisions", 0))
    reward -= 0.25 * float(next_missed - prev_missed)
    reward -= 0.01
    if info.get("success", False):
        reward += 6.0
    elif done:
        reward -= 1.0
    return float(reward)


def aggregate_episode_stats(stats_list):
    if not stats_list:
        return {}
    stats_dicts = [asdict(stats) if isinstance(stats, EpisodeStats) else dict(stats) for stats in stats_list]
    keys = stats_dicts[0].keys()
    aggregated = {f"avg_{key}": float(np.mean([float(stats[key]) for stats in stats_dicts])) for key in keys}
    aggregated["success_rate"] = aggregated["avg_success"]
    aggregated["avg_return"] = aggregated["avg_total_reward"]
    aggregated["avg_length"] = aggregated["avg_episode_length"]
    return aggregated


def format_eval_log(step, stats):
    return (
        f"[EVAL] step={step} "
        f"success={stats.get('success_rate', 0.0):.2f} "
        f"completed={stats.get('avg_completed_tasks', 0.0):.2f} "
        f"deadline={stats.get('avg_deadline_satisfaction', 0.0):.2f} "
        f"dist={stats.get('avg_mean_task_distance', 0.0):.3f} "
        f"coll={stats.get('avg_collisions', 0.0):.2f} "
        f"len={stats.get('avg_length', 0.0):.1f}"
    )


def format_train_log(step, stats, losses):
    return (
        f"[TRAIN] step={step} "
        f"success={stats.get('success_rate', 0.0):.2f} "
        f"completed={stats.get('avg_completed_tasks', 0.0):.2f} "
        f"deadline={stats.get('avg_deadline_satisfaction', 0.0):.2f} "
        f"dist={stats.get('avg_mean_task_distance', 0.0):.3f} "
        f"coll={stats.get('avg_collisions', 0.0):.2f} "
        f"loss(q1/q2/pi)={losses.get('q1_loss', 0.0):.4f}/{losses.get('q2_loss', 0.0):.4f}/{losses.get('policy_loss', 0.0):.4f}"
    )


def _better_eval_candidate(candidate_stats, incumbent_stats):
    if incumbent_stats is None:
        return True
    candidate_key = (
        float(candidate_stats.get("success_rate", 0.0)),
        float(candidate_stats.get("avg_completed_tasks", 0.0)),
        float(candidate_stats.get("avg_deadline_satisfaction", 0.0)),
        float(candidate_stats.get("avg_return", 0.0)),
        -float(candidate_stats.get("avg_mean_task_distance", 0.0)),
    )
    incumbent_key = (
        float(incumbent_stats.get("success_rate", 0.0)),
        float(incumbent_stats.get("avg_completed_tasks", 0.0)),
        float(incumbent_stats.get("avg_deadline_satisfaction", 0.0)),
        float(incumbent_stats.get("avg_return", 0.0)),
        -float(incumbent_stats.get("avg_mean_task_distance", 0.0)),
    )
    return candidate_key > incumbent_key


class TD3HRSGAAgent:
    def __init__(
        self,
        num_balls,
        max_steps,
        hidden_dim=128,
        num_heads=4,
        topk_robot=2,
        topk_task=2,
        topk_obstacle=1,
        actor_lr=3e-4,
        critic_lr=3e-4,
        discount=0.98,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.4,
        device=None,
    ):
        self.num_balls = int(num_balls)
        self.max_steps = int(max_steps)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.topk_robot = int(topk_robot)
        self.topk_task = int(topk_task)
        self.topk_obstacle = int(topk_obstacle)
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs = dict(
            robot_dim=11,
            task_dim=8,
            obstacle_dim=3,
            rr_edge_dim=8,
            tr_edge_dim=8,
            or_edge_dim=5,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            topk_robot=self.topk_robot,
            topk_task=self.topk_task,
            topk_obstacle=self.topk_obstacle,
        )
        self.actor = HRSGAPolicyNetwork(**model_kwargs).to(self.device)
        self.actor_target = HRSGAPolicyNetwork(**model_kwargs).to(self.device)
        self.critic1 = HRSGAQNetwork(**model_kwargs).to(self.device)
        self.critic2 = HRSGAQNetwork(**model_kwargs).to(self.device)
        self.critic1_target = HRSGAQNetwork(**model_kwargs).to(self.device)
        self.critic2_target = HRSGAQNetwork(**model_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.last_losses = {}

    def _to_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        dtype = torch.bool if value.dtype == np.bool_ else torch.float32
        return torch.from_numpy(value).to(self.device, dtype=dtype)

    def _state_tensors(self, batch):
        return {
            "robot_features": self._to_tensor(batch["robot_features"]),
            "task_features": self._to_tensor(batch["task_features"]),
            "obstacle_features": self._to_tensor(batch["obstacle_features"]),
            "rr_edges": self._to_tensor(batch["rr_edges"]),
            "tr_edges": self._to_tensor(batch["tr_edges"]),
            "or_edges": self._to_tensor(batch["or_edges"]),
            "rr_mask": self._to_tensor(batch["rr_mask"]),
            "tr_mask": self._to_tensor(batch["tr_mask"]),
            "or_mask": self._to_tensor(batch["or_mask"]),
            "active_mask": self._to_tensor(batch["active_mask"]),
        }

    def select_action(self, snapshot, noise_sigma=0.0):
        batch = snapshot_to_model_inputs(snapshot, max_steps=self.max_steps)
        batch = {key: value[None, ...] for key, value in batch.items()}
        state = self._state_tensors(batch)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(**state)[0].cpu().numpy()
        if noise_sigma > 0.0:
            action = action + np.random.normal(0.0, noise_sigma, size=action.shape).astype(np.float32)
        return np.clip(action, -1.4, 1.4).astype(np.float32)

    def update(self, batch, policy_update=True):
        state = self._state_tensors(batch)
        next_state = self._state_tensors({key[5:]: value for key, value in batch.items() if key.startswith("next_")})
        actions = self._to_tensor(batch["actions"])
        rewards = self._to_tensor(batch["rewards"])
        dones = self._to_tensor(batch["dones"])

        with torch.no_grad():
            next_actions = self.actor_target(**next_state)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1.4, 1.4)
            target_q1 = self.critic1_target(**next_state, actions=next_actions)
            target_q2 = self.critic2_target(**next_state, actions=next_actions)
            target_q = rewards + (1.0 - dones) * self.discount * torch.min(target_q1, target_q2)

        current_q1 = self.critic1(**state, actions=actions)
        current_q2 = self.critic2(**state, actions=actions)
        q1_loss = nn.functional.mse_loss(current_q1, target_q)
        q2_loss = nn.functional.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=2.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=2.0)
        self.critic2_optimizer.step()

        policy_loss = torch.tensor(0.0, device=self.device)
        if policy_update:
            policy_actions = self.actor(**state)
            policy_loss = -self.critic1(**state, actions=policy_actions).mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        self.last_losses = {
            "q1_loss": float(q1_loss.detach().cpu().item()),
            "q2_loss": float(q2_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
        }

    def _soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(source_param.data, alpha=self.tau)

    def save(self, path, training_step=0, best_success_rate=-np.inf, best_avg_return=-np.inf, best_completed_tasks=-np.inf, best_deadline_satisfaction=-np.inf):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic1_target": self.critic1_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic1_optimizer": self.critic1_optimizer.state_dict(),
                "critic2_optimizer": self.critic2_optimizer.state_dict(),
                "training_step": int(training_step),
                "best_success_rate": float(best_success_rate),
                "best_avg_return": float(best_avg_return),
                "best_completed_tasks": float(best_completed_tasks),
                "best_deadline_satisfaction": float(best_deadline_satisfaction),
                "num_balls": self.num_balls,
                "max_steps": self.max_steps,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "topk_robot": self.topk_robot,
                "topk_task": self.topk_task,
                "topk_obstacle": self.topk_obstacle,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic1_optimizer" in checkpoint:
            self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        if "critic2_optimizer" in checkpoint:
            self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        return checkpoint


def load_checkpoint_metadata(path):
    if not path or not os.path.exists(path):
        return {
            "training_step": 0,
            "best_success_rate": -np.inf,
            "best_avg_return": -np.inf,
            "best_completed_tasks": -np.inf,
            "best_deadline_satisfaction": -np.inf,
        }
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "training_step": int(checkpoint.get("training_step", 0)),
        "best_success_rate": float(checkpoint.get("best_success_rate", -np.inf)),
        "best_avg_return": float(checkpoint.get("best_avg_return", -np.inf)),
        "best_completed_tasks": float(checkpoint.get("best_completed_tasks", -np.inf)),
        "best_deadline_satisfaction": float(checkpoint.get("best_deadline_satisfaction", -np.inf)),
    }


def load_agent_from_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    agent = TD3HRSGAAgent(
        num_balls=int(checkpoint.get("num_balls", 4)),
        max_steps=int(checkpoint.get("max_steps", 180)),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_heads=int(checkpoint.get("num_heads", 4)),
        topk_robot=int(checkpoint.get("topk_robot", 2)),
        topk_task=int(checkpoint.get("topk_task", 2)),
        topk_obstacle=int(checkpoint.get("topk_obstacle", 1)),
    )
    agent.load(path)
    return agent


def run_episode(env, agent, episode_idx, global_step, random_exploration_episodes, policy_sigma, policy_sigma_decay_rate):
    snapshot = env.reset(seed=4300 + episode_idx)
    total_reward = 0.0
    transitions = []
    info = {}
    while True:
        if episode_idx < random_exploration_episodes:
            action = np.random.uniform(-1.0, 1.0, size=(env.num_balls, 2)).astype(np.float32) * env.max_accel
        else:
            sigma = float(policy_sigma * (policy_sigma_decay_rate ** (global_step / 10000.0)))
            action = agent.select_action(snapshot, noise_sigma=sigma)
        next_snapshot, done, info = env.step(action)
        reward = compute_reward(snapshot, next_snapshot, info, done)
        transitions.append(
            {
                "state": snapshot_to_model_inputs(snapshot, max_steps=env.max_steps),
                "actions": np.asarray(action, dtype=np.float32),
                "rewards": np.asarray(reward, dtype=np.float32),
                "next_state": snapshot_to_model_inputs(next_snapshot, max_steps=env.max_steps),
                "dones": np.asarray(float(done), dtype=np.float32),
            }
        )
        total_reward += reward
        snapshot = next_snapshot
        if done:
            break
    return transitions, EpisodeStats(
        total_reward=float(total_reward),
        success=int(bool(info.get("success", False))),
        completed_tasks=int(info.get("completed_tasks", 0)),
        deadline_satisfaction=float(info.get("deadline_satisfaction", 0.0)),
        missed_deadlines=int(info.get("missed_deadlines", 0)),
        collisions=int(info.get("total_collisions", 0)),
        mean_task_distance=float(info.get("mean_task_distance", 0.0)),
        min_pair_distance=float(info.get("min_pair_distance", 0.0)),
        episode_length=int(snapshot["step"]),
    )


def evaluate_agent(agent, num_episodes=8):
    stats = []
    for episode_idx in range(num_episodes):
        env = HRSGABallEnvironment(num_balls=agent.num_balls, max_steps=agent.max_steps)
        snapshot = env.reset(seed=1234 + episode_idx)
        total_reward = 0.0
        info = {}
        while True:
            action = agent.select_action(snapshot, noise_sigma=0.0)
            next_snapshot, done, info = env.step(action)
            reward = compute_reward(snapshot, next_snapshot, info, done)
            total_reward += reward
            snapshot = next_snapshot
            if done:
                break
        stats.append(
            EpisodeStats(
                total_reward=float(total_reward),
                success=int(bool(info.get("success", False))),
                completed_tasks=int(info.get("completed_tasks", 0)),
                deadline_satisfaction=float(info.get("deadline_satisfaction", 0.0)),
                missed_deadlines=int(info.get("missed_deadlines", 0)),
                collisions=int(info.get("total_collisions", 0)),
                mean_task_distance=float(info.get("mean_task_distance", 0.0)),
                min_pair_distance=float(info.get("min_pair_distance", 0.0)),
                episode_length=int(snapshot["step"]),
            )
        )
    return aggregate_episode_stats(stats)


def _transition_to_buffer_entry(transition):
    entry = {
        "actions": transition["actions"],
        "rewards": transition["rewards"],
        "dones": transition["dones"],
    }
    for key, value in transition["state"].items():
        entry[key] = value
    for key, value in transition["next_state"].items():
        entry[f"next_{key}"] = value
    return entry


def train(
    *,
    total_steps=30000,
    batch_size=128,
    update_after=1500,
    updates_per_step=2,
    buffer_size=250000,
    random_exploration_episodes=12,
    policy_sigma=0.25,
    policy_sigma_decay_rate=0.8,
    resume_path=None,
    save_interval=2500,
    eval_interval=2500,
    log_interval=250,
    eval_episodes=6,
    seed=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = HRSGABallEnvironment()
    best_success_rate = -np.inf
    best_avg_return = -np.inf
    best_completed_tasks = -np.inf
    best_deadline_satisfaction = -np.inf
    best_eval_stats = None
    start_step = 0

    if resume_path:
        metadata = load_checkpoint_metadata(resume_path)
        start_step = metadata["training_step"]
        best_success_rate = metadata["best_success_rate"]
        best_avg_return = metadata["best_avg_return"]
        best_completed_tasks = metadata["best_completed_tasks"]
        best_deadline_satisfaction = metadata["best_deadline_satisfaction"]
        best_eval_stats = {
            "success_rate": best_success_rate,
            "avg_return": best_avg_return,
            "avg_completed_tasks": best_completed_tasks,
            "avg_deadline_satisfaction": best_deadline_satisfaction,
        }
        agent = load_agent_from_checkpoint(resume_path)
        print(f"[RESUME] loaded {resume_path} at step={start_step}")
    else:
        agent = TD3HRSGAAgent(num_balls=env.num_balls, max_steps=env.max_steps)

    buffer = StructuredReplayBuffer(max_size=buffer_size)
    episode_idx = 0
    global_step = start_step
    last_eval = None
    completed_stats = []
    while global_step < start_step + total_steps:
        episode_transitions, episode_stats = run_episode(
            env=env,
            agent=agent,
            episode_idx=episode_idx,
            global_step=global_step,
            random_exploration_episodes=random_exploration_episodes,
            policy_sigma=policy_sigma,
            policy_sigma_decay_rate=policy_sigma_decay_rate,
        )
        for transition_idx, transition in enumerate(episode_transitions):
            buffer.add(_transition_to_buffer_entry(transition))
            global_step += 1
            if buffer.size >= max(batch_size, update_after):
                for update_idx in range(updates_per_step):
                    agent.update(buffer.sample(batch_size), policy_update=(update_idx == updates_per_step - 1))
            if eval_interval > 0 and global_step % eval_interval == 0 and buffer.size >= max(batch_size, update_after):
                last_eval = evaluate_agent(agent, num_episodes=eval_episodes)
                print(format_eval_log(global_step, last_eval))
            if save_interval > 0 and global_step % save_interval == 0:
                agent.save(
                    os.path.join(ROOT_DIR, "models", "td3_hrsga_latest.pt"),
                    training_step=global_step,
                    best_success_rate=best_success_rate,
                    best_avg_return=best_avg_return,
                    best_completed_tasks=best_completed_tasks,
                    best_deadline_satisfaction=best_deadline_satisfaction,
                )
                print(f"[SAVE] step={global_step} model saved.")
            if global_step >= start_step + total_steps:
                break

        completed_stats.append(episode_stats)
        if log_interval > 0 and global_step % log_interval == 0 and buffer.size >= max(batch_size, update_after):
            recent_count = min(10, len(completed_stats))
            recent_stats = aggregate_episode_stats(completed_stats[-recent_count:]) if recent_count > 0 else {}
            selection_stats = last_eval if last_eval is not None else recent_stats
            if selection_stats and _better_eval_candidate(selection_stats, best_eval_stats):
                best_eval_stats = dict(selection_stats)
                best_success_rate = float(selection_stats.get("success_rate", best_success_rate))
                best_avg_return = float(selection_stats.get("avg_return", best_avg_return))
                best_completed_tasks = float(selection_stats.get("avg_completed_tasks", best_completed_tasks))
                best_deadline_satisfaction = float(selection_stats.get("avg_deadline_satisfaction", best_deadline_satisfaction))
                agent.save(
                    os.path.join(ROOT_DIR, "models", "td3_hrsga_best.pt"),
                    training_step=global_step,
                    best_success_rate=best_success_rate,
                    best_avg_return=best_avg_return,
                    best_completed_tasks=best_completed_tasks,
                    best_deadline_satisfaction=best_deadline_satisfaction,
                )
            print(format_train_log(global_step, recent_stats, agent.last_losses))
        episode_idx += 1

    agent.save(
        os.path.join(ROOT_DIR, "models", "td3_hrsga_latest.pt"),
        training_step=global_step,
        best_success_rate=best_success_rate,
        best_avg_return=best_avg_return,
        best_completed_tasks=best_completed_tasks,
        best_deadline_satisfaction=best_deadline_satisfaction,
    )
    print("训练完成，模型已保存。")


def main():
    parser = argparse.ArgumentParser(description="HRSGA multi-ball TD3 training")
    parser.add_argument("--total_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--update_after", type=int, default=1500)
    parser.add_argument("--updates_per_step", type=int, default=2)
    parser.add_argument("--buffer_size", type=int, default=250000)
    parser.add_argument("--random_exploration_episodes", type=int, default=12)
    parser.add_argument("--policy_sigma", type=float, default=0.25)
    parser.add_argument("--policy_sigma_decay_rate", type=float, default=0.8)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=2500)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--log_interval", type=int, default=250)
    parser.add_argument("--eval_episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        update_after=args.update_after,
        updates_per_step=args.updates_per_step,
        buffer_size=args.buffer_size,
        random_exploration_episodes=args.random_exploration_episodes,
        policy_sigma=args.policy_sigma,
        policy_sigma_decay_rate=args.policy_sigma_decay_rate,
        resume_path=args.resume_path,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()