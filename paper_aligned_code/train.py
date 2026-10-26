"""
Distributed DRL Training & Evaluation strictly conforming to Tang & Wong (IEEE TMC 2020 / 2022).
Implements:
- Algorithm 1: Device-side Action Selection & Local Experience Gathering
- Algorithm 2: Edge-side Distributed Neural Network Training & Target Net Updates
- Paper Section 5 Parameters: batch_size=16, lr=0.001, gamma=0.9, epsilon decay 1.0 -> 0.01
- Cost Minimization (argmin)
- Explicit Epsilon Reset Mechanism for Non-Stationary Shift
- Baseline Heuristic Evaluators (All-Local, Random, Greedy Queue-Aware)
- Configuration file (.cfg) loading support
"""

import os
import sys
import argparse
import time
import json
import random
import numpy as np

# Ensure local packages (including torch) in .venv_pkgs are found
if sys.version_info >= (3, 8):
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv_pkgs')
    if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

from mec_env import MECEnvironment
from d3qn_agent import D3QNPaperAgent
from workload import WorkloadGenerator
from config_parser import load_config


def train_d3qn(num_episodes=50,
               num_devices=20,
               num_edge_nodes=5,
               workload_pattern='stationary',
               enable_epsilon_reset=False,
               reset_at_episode=25,
               learning_rate=0.001,
               batch_size=16,
               discount_factor=0.9,
               replace_threshold=200,
               memory_capacity=500,
               f_device=2.5,
               f_edge=41.8,
               r_tran=14.0,
               comp_density=0.297,
               penalty_cost=20.0,
               config_dict=None,
               verbose=True):

    # If config_dict is passed, extract all section parameters
    if config_dict is not None:
        env_cfg = config_dict.get('ENVIRONMENT', {})
        agent_cfg = config_dict.get('DRL_AGENT', {})
        wl_cfg = config_dict.get('WORKLOAD', {})
        exp_cfg = config_dict.get('EXPERIMENT', {})

        num_devices = env_cfg.get('num_devices', num_devices)
        num_edge_nodes = env_cfg.get('num_edge_nodes', num_edge_nodes)
        f_device = env_cfg.get('f_device', f_device)
        f_edge = env_cfg.get('f_edge', f_edge)
        r_tran = env_cfg.get('r_tran', r_tran)
        comp_density = env_cfg.get('comp_density', comp_density)
        penalty_cost = env_cfg.get('penalty_cost', penalty_cost)

        learning_rate = agent_cfg.get('learning_rate', learning_rate)
        discount_factor = agent_cfg.get('discount_factor', discount_factor)
        batch_size = agent_cfg.get('batch_size', batch_size)
        replace_threshold = agent_cfg.get('replace_threshold', replace_threshold)
        memory_capacity = agent_cfg.get('memory_capacity', memory_capacity)

        workload_pattern = wl_cfg.get('pattern', workload_pattern)

        num_episodes = exp_cfg.get('episodes', num_episodes)
        enable_epsilon_reset = exp_cfg.get('enable_epsilon_reset', enable_epsilon_reset)
        reset_at_episode = exp_cfg.get('reset_at_episode', reset_at_episode)

    print("==========================================================================")
    print("      D3QN MEC TASK OFFLOADING (STRICTLY PAPER CONFORMING VERSION)        ")
    print("==========================================================================")
    print(f"Algorithm: Double Dueling DQN + LSTM | Optimization: Cost Minimization (argmin)")
    print(f"Hyperparameters: Batch Size={batch_size} | LR={learning_rate} | Gamma={discount_factor} | RMSProp")
    print(f"Devices (M): {num_devices} | Edge Nodes (N): {num_edge_nodes} | Pattern: {workload_pattern.upper()}")
    print("==========================================================================\n")

    env = MECEnvironment(
        num_devices=num_devices,
        num_edge_nodes=num_edge_nodes,
        num_time_slots=100,
        max_deadline=10,
        time_slot_duration=0.1,
        f_device=f_device,
        f_edge=f_edge,
        r_tran=r_tran,
        comp_density=comp_density,
        penalty_cost=penalty_cost
    )

    workload_gen = WorkloadGenerator(
        pattern=workload_pattern,
        num_devices=num_devices,
        base_arrival_prob=0.3,
        seed=42
    )

    agents = [
        D3QNPaperAgent(
            num_local_features=env.num_local_features,
            num_edge_nodes=num_edge_nodes,
            num_actions=env.num_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            replace_threshold=replace_threshold,
            memory_capacity=memory_capacity,
            epsilon_init=1.0,
            epsilon_min=0.01,
            epsilon_decay_steps=num_episodes * 40
        ) for _ in range(num_devices)
    ]

    metrics = {
        'episodes': [],
        'average_cost': [],
        'average_delay': [],
        'drop_ratio': [],
        'drop_local': [],
        'drop_tran': [],
        'drop_edge': [],
        'action_distribution': [],
        'epsilon': [],
        'training_loss': [],
        'slot_delays': []
    }

    t_start = time.time()

    for episode in range(num_episodes):

        # Tang & Wong Section 5 reset mechanism if enabled
        if enable_epsilon_reset and episode == reset_at_episode:
            print(f"\n>>> [WORKLOAD SHIFT DETECTED] Resetting epsilon to 1.0 (Tang & Wong Section 5) <<<\n")
            for agent in agents:
                agent.reset_exploration(new_epsilon=1.0)

        # Generate discrete task arrivals
        bit_arrive = workload_gen.generate(env.T, env.tau, episode=episode)
        local_s, H_s = env.reset(bit_arrive)

        history = [{} for _ in range(env.T + env.tau + 1)]
        action_counts = np.zeros(env.num_actions)

        while True:
            t = env.time_slot
            actions = np.zeros(num_devices, dtype=int)

            for m in range(num_devices):
                task_bits = env.bit_arrive[t, m] if t < len(env.bit_arrive) else 0.0

                if task_bits > 0.0:
                    actions[m] = agents[m].select_action(local_s[m], H_s[m])
                    action_counts[actions[m]] += 1
                    history[t][m] = {
                        'local_s': local_s[m].copy(),
                        'H_s': H_s[m].copy(),
                        'action': actions[m]
                    }
                else:
                    actions[m] = 0

            next_local_s, next_H_s, finished_tasks_per_device, done = env.step(actions)

            for m in range(num_devices):
                for finished_t in finished_tasks_per_device[m]:
                    if finished_t in history and m in history[finished_t]:
                        exp = history[finished_t][m]
                        cost = env.task_cost[finished_t, m]

                        agents[m].store_experience(
                            local_s=exp['local_s'],
                            H_s=exp['H_s'],
                            action=exp['action'],
                            cost=cost,
                            local_s_next=next_local_s[m],
                            H_s_next=next_H_s[m]
                        )
                        agents[m].train_step()

            local_s = next_local_s
            H_s = next_H_s

            if done:
                break

        total_tasks = int(np.count_nonzero(bit_arrive[:env.T, :]))
        dropped_tasks = int(np.sum(env.task_dropped[:env.T, :]))
        drop_rate = float(dropped_tasks / max(1, total_tasks))

        finished_mask = (env.task_delay[:env.T, :] > 0) & (~env.task_dropped[:env.T, :])
        mean_delay = float(np.mean(env.task_delay[:env.T, :][finished_mask])) if np.any(finished_mask) else float(env.tau)
        mean_cost = float(np.mean(env.task_cost[:env.T, :]))
        mean_loss = float(np.mean([a.latest_loss for a in agents]))
        curr_eps = float(agents[0].epsilon)

        total_acts = max(1, int(np.sum(action_counts)))
        act_dist = (action_counts / total_acts).tolist()

        slot_delays = []
        for s_idx in range(env.T):
            active_mask = env.task_delay[s_idx, :] > 0
            if np.any(active_mask):
                slot_delays.append(float(np.mean(env.task_delay[s_idx, active_mask])))
            else:
                slot_delays.append(0.0)

        metrics['episodes'].append(episode)
        metrics['average_cost'].append(mean_cost)
        metrics['average_delay'].append(mean_delay)
        metrics['drop_ratio'].append(drop_rate)
        metrics['drop_local'].append(int(env.drop_local_count))
        metrics['drop_tran'].append(int(env.drop_tran_count))
        metrics['drop_edge'].append(int(env.drop_edge_count))
        metrics['action_distribution'].append(act_dist)
        metrics['epsilon'].append(curr_eps)
        metrics['training_loss'].append(mean_loss)
        metrics['slot_delays'].append(slot_delays)

        # Decay exploration epsilon by episode schedule (1.0 -> 0.01)
        for agent in agents:
            agent.decay_epsilon_by_episode(episode, num_episodes)

        if verbose and (episode % max(1, num_episodes // 10) == 0 or episode == num_episodes - 1):
            elapsed = time.time() - t_start
            print(f"Episode {episode:3d}/{num_episodes} | "
                  f"Avg Cost: {mean_cost:5.2f} | "
                  f"Avg Delay: {mean_delay:4.2f} slots | "
                  f"Drop Ratio: {drop_rate * 100:5.1f}% | "
                  f"Epsilon: {curr_eps:5.3f} | "
                  f"Elapsed: {elapsed:5.1f}s")

    return metrics, agents, env


def evaluate_policy(env, agents, workload_gen, num_eval_episodes=15):
    old_eps = [a.epsilon for a in agents]
    for a in agents:
        a.epsilon = 0.0

    delays = []
    drop_ratios = []
    costs = []

    for ep in range(num_eval_episodes):
        bit_arrive = workload_gen.generate(env.T, env.tau, episode=ep)
        local_s, H_s = env.reset(bit_arrive)

        while True:
            t = env.time_slot
            actions = np.zeros(env.M, dtype=int)
            for m in range(env.M):
                if env.bit_arrive[t, m] > 0.0:
                    actions[m] = agents[m].select_action(local_s[m], H_s[m])
                else:
                    actions[m] = 0

            local_s, H_s, _, done = env.step(actions)
            if done:
                break

        total_tasks = int(np.count_nonzero(bit_arrive[:env.T, :]))
        dropped = int(np.sum(env.task_dropped[:env.T, :]))
        drop_ratios.append(float(dropped / max(1, total_tasks)))

        finished_mask = (env.task_delay[:env.T, :] > 0) & (~env.task_dropped[:env.T, :])
        mean_delay = float(np.mean(env.task_delay[:env.T, :][finished_mask])) if np.any(finished_mask) else float(env.tau)
        delays.append(mean_delay)
        costs.append(float(np.mean(env.task_cost[:env.T, :])))

    for idx, a in enumerate(agents):
        a.epsilon = old_eps[idx]

    return {
        'avg_delay': float(np.mean(delays)),
        'avg_drop_ratio': float(np.mean(drop_ratios)),
        'avg_cost': float(np.mean(costs))
    }


def evaluate_baseline(env, workload_gen, mode='local', num_episodes=15):
    delays = []
    drop_ratios = []
    costs = []

    for ep in range(num_episodes):
        bit_arrive = workload_gen.generate(env.T, env.tau, episode=ep)
        local_s, H_s = env.reset(bit_arrive)

        while True:
            t = env.time_slot
            actions = np.zeros(env.M, dtype=int)

            for m in range(env.M):
                if env.bit_arrive[t, m] > 0.0:
                    if mode == 'local':
                        actions[m] = 0
                    elif mode == 'random':
                        actions[m] = random.randint(0, env.num_actions - 1)
                    elif mode == 'greedy':
                        q_edge = local_s[m, 3:3 + env.N]
                        best_edge = int(np.argmin(q_edge)) + 1
                        actions[m] = best_edge
                else:
                    actions[m] = 0

            local_s, H_s, _, done = env.step(actions)
            if done:
                break

        total_tasks = int(np.count_nonzero(bit_arrive[:env.T, :]))
        dropped = int(np.sum(env.task_dropped[:env.T, :]))
        drop_ratios.append(float(dropped / max(1, total_tasks)))

        finished_mask = (env.task_delay[:env.T, :] > 0) & (~env.task_dropped[:env.T, :])
        mean_delay = float(np.mean(env.task_delay[:env.T, :][finished_mask])) if np.any(finished_mask) else float(env.tau)
        delays.append(mean_delay)
        costs.append(float(np.mean(env.task_cost[:env.T, :])))

    return {
        'avg_delay': float(np.mean(delays)),
        'avg_drop_ratio': float(np.mean(drop_ratios)),
        'avg_cost': float(np.mean(costs))
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Paper-Conforming D3QN MEC Training with .cfg Support")
    parser.add_argument("--config", type=str, default=None, help="Path to .cfg configuration file (e.g. config.cfg)")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes (overrides config)")
    parser.add_argument("--devices", type=int, default=None, help="Number of devices M (overrides config)")
    parser.add_argument("--edge_nodes", type=int, default=None, help="Number of edge nodes N (overrides config)")
    parser.add_argument("--workload", type=str, default=None,
                        choices=["stationary", "sinusoidal", "sudden_shift", "bursty", "markovian"],
                        help="Workload pattern (overrides config)")
    parser.add_argument("--enable_reset", action="store_true", default=None, help="Enable epsilon reset adaptation")
    args = parser.parse_args()

    # Load from .cfg file
    cfg = load_config(args.config)

    # CLI overrides
    if args.episodes is not None: cfg['EXPERIMENT']['episodes'] = args.episodes
    if args.devices is not None: cfg['ENVIRONMENT']['num_devices'] = args.devices
    if args.edge_nodes is not None: cfg['ENVIRONMENT']['num_edge_nodes'] = args.edge_nodes
    if args.workload is not None: cfg['WORKLOAD']['pattern'] = args.workload
    if args.enable_reset: cfg['EXPERIMENT']['enable_epsilon_reset'] = True

    metrics, agents, env = train_d3qn(config_dict=cfg)

    wl_name = cfg['WORKLOAD']['pattern']
    output_path = os.path.join(cfg['EXPERIMENT'].get('output_dir', 'paper_aligned_code/reports'), f"metrics_{wl_name}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {output_path}")
