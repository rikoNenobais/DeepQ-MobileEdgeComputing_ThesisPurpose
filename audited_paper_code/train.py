"""
Audited Training & Evaluation Engine for MEC D3QN Task Offloading.
Features:
1. Strict two-phase execution: Active arrival phase (0..T-1) followed by Drain period (T..T+tau-1).
2. Verifiable conservation: total_arrived == completed + dropped.
3. Strict category consistency: dropped == drop_local + drop_tran + drop_edge.
4. Clean separation of successful task delay vs penalty cost.
5. Deterministic trace-paired evaluation across all baseline policies.
"""

import os
import sys
import time
import json
import argparse
import numpy as np

# Ensure local package path support
if sys.version_info >= (3, 8):
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv_pkgs')
    if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

from mec_env import AuditedMECEnvironment
from workload import AuditedWorkloadGenerator
from d3qn_agent import D3QNAgent
from config_parser import load_config, get_default_config


def train_d3qn(num_episodes=30,
               num_devices=20,
               num_edge_nodes=5,
               num_time_slots=100,
               max_deadline=10,
               workload_pattern='stationary',
               base_arrival_prob=0.30,
               learning_rate=0.001,
               discount_factor=0.9,
               batch_size=16,
               replace_threshold=200,
               memory_capacity=500,
               epsilon_init=1.0,
               epsilon_min=0.01,
               enable_epsilon_reset=False,
               reset_at_episode=15,
               config_dict=None,
               verbose=True):

    if config_dict is not None:
        env_cfg = config_dict.get('ENVIRONMENT', {})
        drl_cfg = config_dict.get('DRL_AGENT', {})
        work_cfg = config_dict.get('WORKLOAD', {})
        exp_cfg = config_dict.get('EXPERIMENT', {})

        num_devices = env_cfg.get('num_devices', num_devices)
        num_edge_nodes = env_cfg.get('num_edge_nodes', num_edge_nodes)
        num_time_slots = env_cfg.get('num_time_slots', num_time_slots)
        max_deadline = env_cfg.get('max_deadline', max_deadline)

        learning_rate = drl_cfg.get('learning_rate', learning_rate)
        discount_factor = drl_cfg.get('discount_factor', discount_factor)
        batch_size = drl_cfg.get('batch_size', batch_size)
        replace_threshold = drl_cfg.get('replace_threshold', replace_threshold)
        memory_capacity = drl_cfg.get('memory_capacity', memory_capacity)
        epsilon_init = drl_cfg.get('epsilon_init', epsilon_init)
        epsilon_min = drl_cfg.get('epsilon_min', epsilon_min)

        workload_pattern = work_cfg.get('pattern', workload_pattern)
        base_arrival_prob = work_cfg.get('base_arrival_prob', base_arrival_prob)

        num_episodes = exp_cfg.get('episodes', num_episodes)
        enable_epsilon_reset = exp_cfg.get('enable_epsilon_reset', enable_epsilon_reset)
        reset_at_episode = exp_cfg.get('reset_at_episode', reset_at_episode)

    # Initialize audited environment
    env = AuditedMECEnvironment(
        num_devices=num_devices,
        num_edge_nodes=num_edge_nodes,
        num_time_slots=num_time_slots,
        max_deadline=max_deadline,
        task_arrival_prob=base_arrival_prob
    )

    # Initialize audited workload generator
    workload_gen = AuditedWorkloadGenerator(
        pattern=workload_pattern,
        num_devices=num_devices,
        base_arrival_prob=base_arrival_prob,
        seed=42
    )

    # Initialize D3QN agents (one per mobile device)
    agents = [
        D3QNAgent(
            num_local_features=env.num_local_features,
            num_edge_nodes=env.N,
            num_actions=env.num_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            replace_threshold=replace_threshold,
            memory_capacity=memory_capacity,
            epsilon_init=epsilon_init,
            epsilon_min=epsilon_min
        )
        for m in range(num_devices)
    ]

    metrics = {
        'episodes': [],
        'total_arrived': [],
        'completed_count': [],
        'dropped_count': [],
        'drop_ratio': [],
        'avg_delay_successful': [],
        'avg_delay_with_penalty': [],
        'avg_cost': [],
        'average_delay': [],
        'average_cost': [],
        'drop_local': [],
        'drop_tran': [],
        'drop_edge': [],
        'epsilon': [],
        'training_loss': [],
        'action_distribution': [],
        'slot_delays': [],
        'slot_diagnostics_last_ep': []
    }

    t_start = time.time()

    for episode in range(num_episodes):

        # Tang & Wong exploration reset if enabled
        if enable_epsilon_reset and episode == reset_at_episode:
            if verbose:
                print(f"\n>>> [WORKLOAD SHIFT DETECTED] Resetting epsilon to 1.0 (Tang & Wong Section 5) <<<\n")
            for agent in agents:
                agent.reset_exploration(new_epsilon=1.0)

        # Generate discrete arrivals with shape [T + tau, M]
        bit_arrive = workload_gen.generate(env.T, env.tau, episode=episode)
        local_s, H_s = env.reset(bit_arrive)

        # Experience storage history per time slot
        history = [{} for _ in range(env.total_slots + 1)]
        action_counts = np.zeros(env.num_actions)

        while True:
            t = env.time_slot
            actions = np.zeros(num_devices, dtype=int)

            # Active Arrival Phase (t < T): Agents select actions for arriving tasks
            if t < env.T:
                for m in range(num_devices):
                    task_bits = env.bit_arrive[t, m]
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
            else:
                # Drain Period (T <= t < T + tau): No new tasks arrive, dummy action
                actions[:] = 0

            # Step environment through physical queue processing
            next_local_s, next_H_s, finished_experiences, done, info = env.step(actions)

            # Deliver feedback to experience replays for finished tasks
            for m in range(num_devices):
                for finished_t in finished_experiences[m]:
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

        # Extract audited summary metrics
        ep_summary = env.get_summary_metrics()

        curr_eps = float(agents[0].epsilon)
        mean_loss = float(np.mean([a.latest_loss for a in agents]))
        total_acts = max(1, int(np.sum(action_counts)))
        act_dist = (action_counts / total_acts).tolist()

        slot_delays = []
        for s_idx in range(env.T):
            active_mask = (env.task_delay[s_idx, :] > 0) & (~env.task_dropped[s_idx, :])
            if np.any(active_mask):
                slot_delays.append(float(np.mean(env.task_delay[s_idx, active_mask])))
            else:
                slot_delays.append(0.0)

        metrics['episodes'].append(episode)
        metrics['total_arrived'].append(ep_summary['total_arrived'])
        metrics['completed_count'].append(ep_summary['completed_count'])
        metrics['dropped_count'].append(ep_summary['dropped_count'])
        metrics['drop_ratio'].append(ep_summary['drop_ratio'])
        metrics['avg_delay_successful'].append(ep_summary['avg_delay_successful'])
        metrics['avg_delay_with_penalty'].append(ep_summary['avg_delay_with_penalty'])
        metrics['avg_cost'].append(ep_summary['avg_cost'])
        metrics['average_delay'].append(ep_summary['avg_delay_successful'])
        metrics['average_cost'].append(ep_summary['avg_cost'])
        metrics['drop_local'].append(ep_summary['drop_local'])
        metrics['drop_tran'].append(ep_summary['drop_tran'])
        metrics['drop_edge'].append(ep_summary['drop_edge'])
        metrics['epsilon'].append(curr_eps)
        metrics['training_loss'].append(mean_loss)
        metrics['action_distribution'].append(act_dist)
        metrics['slot_delays'].append(slot_delays)

        if episode == num_episodes - 1:
            metrics['slot_diagnostics_last_ep'] = ep_summary['diagnostic_logs']

        # Epsilon decay schedule per episode
        for agent in agents:
            agent.decay_epsilon_by_episode(episode, num_episodes)

        if verbose and (episode % max(1, num_episodes // 10) == 0 or episode == num_episodes - 1):
            elapsed = time.time() - t_start
            print(f"Episode {episode:2d}/{num_episodes} | "
                  f"Arrived: {ep_summary['total_arrived']:3d} | "
                  f"Cost: {ep_summary['avg_cost']:5.2f} | "
                  f"Delay(succ): {ep_summary['avg_delay_successful']:4.2f}s | "
                  f"Drop: {ep_summary['drop_ratio']*100:5.1f}% | "
                  f"Eps: {curr_eps:4.2f} | "
                  f"Elapsed: {elapsed:4.1f}s")

    return metrics, agents, env


def evaluate_policy_on_traces(env, agents, traces):
    """
    Evaluate trained D3QN greedily (epsilon=0) on a pre-generated set of identical traces.
    """
    old_eps = [a.epsilon for a in agents]
    for a in agents:
        a.epsilon = 0.0

    ep_delays = []
    ep_drops = []
    ep_costs = []
    ep_arrived = []
    ep_completed = []

    for trace in traces:
        local_s, H_s = env.reset(trace)

        while True:
            t = env.time_slot
            actions = np.zeros(env.M, dtype=int)

            if t < env.T:
                for m in range(env.M):
                    if env.bit_arrive[t, m] > 0.0:
                        actions[m] = agents[m].select_action(local_s[m], H_s[m])
                    else:
                        actions[m] = 0
            else:
                actions[:] = 0

            local_s, H_s, _, done, _ = env.step(actions)
            if done:
                break

        summary = env.get_summary_metrics()
        ep_delays.append(summary['avg_delay_successful'])
        ep_drops.append(summary['drop_ratio'])
        ep_costs.append(summary['avg_cost'])
        ep_arrived.append(summary['total_arrived'])
        ep_completed.append(summary['completed_count'])

    for idx, a in enumerate(agents):
        a.epsilon = old_eps[idx]

    return {
        'mean_delay': float(np.mean(ep_delays)),
        'std_delay': float(np.std(ep_delays)),
        'mean_drop_ratio': float(np.mean(ep_drops)),
        'std_drop_ratio': float(np.std(ep_drops)),
        'mean_cost': float(np.mean(ep_costs)),
        'std_cost': float(np.std(ep_costs)),
        'per_episode_delays': ep_delays,
        'per_episode_drops': ep_drops,
        'per_episode_costs': ep_costs
    }


def evaluate_baseline_on_traces(env, traces, mode='local', random_seed=42):
    """
    Evaluate baseline heuristics on the EXACT SAME pre-generated traces.
    mode: 'local' (All-Local), 'random' (Random Offload), 'greedy' (Greedy Queue-Aware).
    """
    ep_delays = []
    ep_drops = []
    ep_costs = []
    ep_arrived = []
    ep_completed = []

    rng = np.random.RandomState(random_seed)

    for trace in traces:
        local_s, H_s = env.reset(trace)

        while True:
            t = env.time_slot
            actions = np.zeros(env.M, dtype=int)

            if t < env.T:
                for m in range(env.M):
                    if env.bit_arrive[t, m] > 0.0:
                        if mode == 'local':
                            actions[m] = 0
                        elif mode == 'random':
                            actions[m] = rng.randint(0, env.num_actions)
                        elif mode == 'greedy':
                            # Edge queue lengths q_edge: features 3..3+N
                            q_edge = local_s[m, 3:3 + env.N]
                            actions[m] = int(np.argmin(q_edge)) + 1
                    else:
                        actions[m] = 0
            else:
                actions[:] = 0

            local_s, H_s, _, done, _ = env.step(actions)
            if done:
                break

        summary = env.get_summary_metrics()
        ep_delays.append(summary['avg_delay_successful'])
        ep_drops.append(summary['drop_ratio'])
        ep_costs.append(summary['avg_cost'])
        ep_arrived.append(summary['total_arrived'])
        ep_completed.append(summary['completed_count'])

    return {
        'mean_delay': float(np.mean(ep_delays)),
        'std_delay': float(np.std(ep_delays)),
        'mean_drop_ratio': float(np.mean(ep_drops)),
        'std_drop_ratio': float(np.std(ep_drops)),
        'mean_cost': float(np.mean(ep_costs)),
        'std_cost': float(np.std(ep_costs)),
        'per_episode_delays': ep_delays,
        'per_episode_drops': ep_drops,
        'per_episode_costs': ep_costs
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audited D3QN MEC Training")
    parser.add_argument("--config", type=str, default=None, help="Path to .cfg configuration file")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--workload", type=str, default="stationary", help="Workload pattern")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    metrics, agents, env = train_d3qn(num_episodes=args.episodes, workload_pattern=args.workload, config_dict=cfg)
    print("\nTraining completed successfully with strict accounting validation.")
