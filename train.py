"""
Training Script for Deep Reinforcement Learning in Mobile Edge Computing (MEC).
Supports both Stationary and Non-Stationary Workload Models.
Integrates with PyTorch-based Double Dueling DQN with LSTM.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import random

# Ensure local packages (including torch) in .venv_pkgs are found if not system-wide
pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv_pkgs')
if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

# Ensure matplotlib config directory is writable
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

from fog_env import Offload
from RL_brain import DeepQNetwork
from workload import NonStationaryWorkload

np.set_printoptions(threshold=np.inf)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def reward_fun(delay, max_delay, unfinish_indi):
    # Penalty for unfinished/dropped task
    penalty = - max_delay * 2
    if unfinish_indi:
        reward = penalty
    else:
        reward = - delay
    return reward


def train(iot_RL_list, NUM_EPISODE, env_instance=None, workload_generator=None, verbose=True):
    """
    Train DRL agents on the MEC environment.
    Fully backward-compatible with original train(iot_RL_list, NUM_EPISODE).
    Returns a comprehensive metrics dictionary.
    """
    global env
    if env_instance is not None:
        current_env = env_instance
    elif 'env' in globals() and globals()['env'] is not None:
        current_env = globals()['env']
    else:
        current_env = Offload(len(iot_RL_list), 5, 110, 10)

    RL_step = 0

    # Metrics history
    metrics = {
        'workload_pattern': workload_generator.pattern if workload_generator else 'stationary',
        'episodes': [],
        'episode_reward': [],
        'avg_delay': [],
        'drop_ratio': [],
        'drop_iot': [],
        'drop_fog': [],
        'drop_trans': [],
        'total_tasks': [],
        'finished_tasks': [],
        'action_distribution': [],  # [pct_local, pct_fog...]
        'epsilon': [],
        'time_slot_delays': []
    }

    t_start = time.time()

    for episode in range(NUM_EPISODE):

        # WORKLOAD ARRIVAL
        if workload_generator is not None:
            bitarrive = workload_generator.generate(
                current_env.n_time, current_env.n_iot, current_env.max_delay, episode
            )
        else:
            # Baseline stationary arrival
            bitarrive = np.random.uniform(
                current_env.min_bit_arrive, current_env.max_bit_arrive,
                size=[current_env.n_time, current_env.n_iot]
            )
            task_prob = current_env.task_arrive_prob
            bitarrive = bitarrive * (np.random.uniform(0, 1, size=[current_env.n_time, current_env.n_iot]) < task_prob)
            bitarrive[-current_env.max_delay:, :] = np.zeros([current_env.max_delay, current_env.n_iot])

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(current_env.n_time):
            history.append(list())
            for iot_index in range(current_env.n_iot):
                tmp_dict = {
                    'observation': np.zeros(current_env.n_features),
                    'lstm': np.zeros(current_env.n_lstm_state),
                    'action': np.nan,
                    'observation_': np.zeros(current_env.n_features),
                    'lstm_': np.zeros(current_env.n_lstm_state)
                }
                history[time_index].append(tmp_dict)

        reward_indicator = np.zeros([current_env.n_time, current_env.n_iot])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = current_env.reset(bitarrive)

        # Track actions in episode
        action_counts = np.zeros(current_env.n_actions)

        # TRAIN DRL
        while True:

            # PERFORM ACTION
            action_all = np.zeros([current_env.n_iot], dtype=int)
            for iot_index in range(current_env.n_iot):
                observation = np.squeeze(observation_all[iot_index, :])

                if np.sum(observation) == 0:
                    action_all[iot_index] = 0
                else:
                    action_all[iot_index] = iot_RL_list[iot_index].choose_action(observation)

                if observation[0] != 0:
                    iot_RL_list[iot_index].do_store_action(episode, current_env.time_count, action_all[iot_index])
                    action_counts[action_all[iot_index]] += 1

            # OBSERVE NEXT STATE AND REWARD
            observation_all_, lstm_state_all_, done = current_env.step(action_all)

            for iot_index in range(current_env.n_iot):
                iot_RL_list[iot_index].update_lstm(lstm_state_all_[iot_index, :])

            process_delay = current_env.process_delay
            unfinish_indi = current_env.process_delay_unfinish_ind

            # STORE TRANSITIONS
            for iot_index in range(current_env.n_iot):
                t_idx = current_env.time_count - 1
                history[t_idx][iot_index]['observation'] = observation_all[iot_index, :]
                history[t_idx][iot_index]['lstm'] = np.squeeze(lstm_state_all[iot_index, :])
                history[t_idx][iot_index]['action'] = action_all[iot_index]
                history[t_idx][iot_index]['observation_'] = observation_all_[iot_index]
                history[t_idx][iot_index]['lstm_'] = np.squeeze(lstm_state_all_[iot_index, :])

                update_index = np.where((1 - reward_indicator[:, iot_index]) * process_delay[:, iot_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        r_val = reward_fun(
                            process_delay[time_index, iot_index],
                            current_env.max_delay,
                            unfinish_indi[time_index, iot_index]
                        )
                        iot_RL_list[iot_index].store_transition(
                            history[time_index][iot_index]['observation'],
                            history[time_index][iot_index]['lstm'],
                            history[time_index][iot_index]['action'],
                            r_val,
                            history[time_index][iot_index]['observation_'],
                            history[time_index][iot_index]['lstm_']
                        )
                        iot_RL_list[iot_index].do_store_reward(episode, time_index, r_val)
                        iot_RL_list[iot_index].do_store_delay(
                            episode, time_index, process_delay[time_index, iot_index]
                        )
                        reward_indicator[time_index, iot_index] = 1

            RL_step += 1
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING FREQUENCY
            if (RL_step > 200) and (RL_step % 10 == 0):
                for iot in range(current_env.n_iot):
                    iot_RL_list[iot].learn()

            if done:
                break

        # EPISODE METRICS COMPUTATION
        total_tasks_arrived = int(np.count_nonzero(bitarrive))
        dropped_tasks = int(np.sum(current_env.process_delay_unfinish_ind))
        finished_tasks = max(0, total_tasks_arrived - dropped_tasks)
        drop_rate = float(dropped_tasks / max(1, total_tasks_arrived))

        # Average delay of finished tasks
        finished_mask = (current_env.process_delay > 0) & (current_env.process_delay_unfinish_ind == 0)
        mean_delay = float(np.mean(current_env.process_delay[finished_mask])) if np.any(finished_mask) else float(current_env.max_delay)

        # Average reward across all IoT devices
        ep_rewards = [np.sum(iot_RL_list[iot].reward_store[episode]) for iot in range(current_env.n_iot)]
        mean_ep_reward = float(np.mean(ep_rewards))

        # Action distribution percentages
        total_actions = max(1, int(np.sum(action_counts)))
        act_dist = (action_counts / total_actions).tolist()

        # Slot-wise average delay (first 100 slots)
        n_base = current_env.n_time - current_env.max_delay
        slot_delays = []
        for t in range(n_base):
            active_t = current_env.process_delay[t, :] > 0
            if np.any(active_t):
                slot_delays.append(float(np.mean(current_env.process_delay[t, active_t])))
            else:
                slot_delays.append(0.0)

        # Append to metrics
        metrics['episodes'].append(episode)
        metrics['episode_reward'].append(mean_ep_reward)
        metrics['avg_delay'].append(mean_delay)
        metrics['drop_ratio'].append(drop_rate)
        metrics['drop_iot'].append(int(current_env.drop_iot_count))
        metrics['drop_fog'].append(int(current_env.drop_fog_count))
        metrics['drop_trans'].append(int(current_env.drop_trans_count))
        metrics['total_tasks'].append(total_tasks_arrived)
        metrics['finished_tasks'].append(finished_tasks)
        metrics['action_distribution'].append(act_dist)
        metrics['epsilon'].append(float(iot_RL_list[0].epsilon))
        metrics['time_slot_delays'].append(slot_delays)

        if verbose and (episode % max(1, NUM_EPISODE // 10) == 0 or episode == NUM_EPISODE - 1):
            elapsed = time.time() - t_start
            print(f"Episode {episode:4d}/{NUM_EPISODE} | "
                  f"Reward: {mean_ep_reward:7.2f} | "
                  f"Avg Delay: {mean_delay:5.2f} slots | "
                  f"Drop Ratio: {drop_rate * 100:5.1f}% | "
                  f"Epsilon: {iot_RL_list[0].epsilon:5.3f} | "
                  f"Elapsed: {elapsed:5.1f}s")

    return metrics


def evaluate_policy(env_instance, iot_RL_list, workload_generator, num_eval_episodes=10):
    """
    Evaluate trained DRL policy (with exploration disabled) under non-stationary workload.
    """
    eval_metrics = {
        'rewards': [],
        'delays': [],
        'drop_ratios': [],
        'action_counts': np.zeros(env_instance.n_actions)
    }

    # Backup epsilons
    old_eps = [agent.epsilon for agent in iot_RL_list]
    for agent in iot_RL_list:
        agent.epsilon = 1.0  # pure exploitation

    for ep in range(num_eval_episodes):
        bitarrive = workload_generator.generate(
            env_instance.n_time, env_instance.n_iot, env_instance.max_delay, ep
        )
        obs_all, lstm_all = env_instance.reset(bitarrive)

        while True:
            action_all = np.zeros(env_instance.n_iot, dtype=int)
            for i in range(env_instance.n_iot):
                obs = obs_all[i, :]
                if np.sum(obs) == 0:
                    action_all[i] = 0
                else:
                    action_all[i] = iot_RL_list[i].choose_action(obs)
                if obs[0] != 0:
                    eval_metrics['action_counts'][action_all[i]] += 1

            obs_all_, lstm_all_, done = env_instance.step(action_all)
            for i in range(env_instance.n_iot):
                iot_RL_list[i].update_lstm(lstm_all_[i, :])
            obs_all = obs_all_
            lstm_all = lstm_all_
            if done:
                break

        total_tasks = int(np.count_nonzero(bitarrive))
        dropped = int(np.sum(env_instance.process_delay_unfinish_ind))
        drop_rate = float(dropped / max(1, total_tasks))

        finished_mask = (env_instance.process_delay > 0) & (env_instance.process_delay_unfinish_ind == 0)
        mean_delay = float(np.mean(env_instance.process_delay[finished_mask])) if np.any(finished_mask) else float(env_instance.max_delay)

        eval_metrics['delays'].append(mean_delay)
        eval_metrics['drop_ratios'].append(drop_rate)

    # Restore epsilons
    for i, agent in enumerate(iot_RL_list):
        agent.epsilon = old_eps[i]

    return {
        'avg_delay': float(np.mean(eval_metrics['delays'])),
        'avg_drop_ratio': float(np.mean(eval_metrics['drop_ratios'])),
        'action_distribution': (eval_metrics['action_counts'] / max(1, np.sum(eval_metrics['action_counts']))).tolist()
    }


def evaluate_baseline(env_instance, workload_generator, mode='local', num_episodes=10):
    """
    Evaluate heuristic baseline policies:
    - 'local': All tasks executed locally (action = 0)
    - 'random': Uniform random actions
    - 'greedy': Offload to the fog node with the minimum observed queue
    """
    delays = []
    drop_ratios = []

    for ep in range(num_episodes):
        bitarrive = workload_generator.generate(
            env_instance.n_time, env_instance.n_iot, env_instance.max_delay, ep
        )
        obs_all, lstm_all = env_instance.reset(bitarrive)

        while True:
            action_all = np.zeros(env_instance.n_iot, dtype=int)
            for i in range(env_instance.n_iot):
                obs = obs_all[i, :]
                if np.sum(obs) == 0:
                    action_all[i] = 0
                else:
                    if mode == 'local':
                        action_all[i] = 0
                    elif mode == 'random':
                        action_all[i] = random.randint(0, env_instance.n_actions - 1)
                    elif mode == 'greedy':
                        fog_queues = obs[3:3 + env_instance.n_fog]
                        best_fog = int(np.argmin(fog_queues)) + 1
                        action_all[i] = best_fog

            obs_all_, lstm_all_, done = env_instance.step(action_all)
            obs_all = obs_all_
            lstm_all = lstm_all_
            if done:
                break

        total_tasks = int(np.count_nonzero(bitarrive))
        dropped = int(np.sum(env_instance.process_delay_unfinish_ind))
        drop_rate = float(dropped / max(1, total_tasks))

        finished_mask = (env_instance.process_delay > 0) & (env_instance.process_delay_unfinish_ind == 0)
        mean_delay = float(np.mean(env_instance.process_delay[finished_mask])) if np.any(finished_mask) else float(env_instance.max_delay)

        delays.append(mean_delay)
        drop_ratios.append(drop_rate)

    return {
        'avg_delay': float(np.mean(delays)),
        'avg_drop_ratio': float(np.mean(drop_ratios))
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRL Task Offloading under Non-Stationary Workloads")
    parser.add_argument("--workload", type=str, default="sinusoidal",
                        choices=["stationary", "sinusoidal", "piecewise", "bursty", "markovian", "episode_drift"],
                        help="Workload arrival pattern")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--iot", type=int, default=20, help="Number of IoT devices")
    parser.add_argument("--fogs", type=int, default=5, help="Number of Fog nodes")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory to save metrics and report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    NUM_IOT = args.iot
    NUM_FOG = args.fogs
    NUM_EPISODE = args.episodes
    NUM_TIME_BASE = 100
    MAX_DELAY = 10
    NUM_TIME = NUM_TIME_BASE + MAX_DELAY

    print(f"=== Initializing MEC Environment ===")
    print(f"IoT Devices: {NUM_IOT} | Fog Servers: {NUM_FOG} | Time Horizon: {NUM_TIME} slots")
    print(f"Workload Pattern: {args.workload.upper()} | Episodes: {NUM_EPISODE}")

    # GENERATE ENVIRONMENT
    env = Offload(NUM_IOT, NUM_FOG, NUM_TIME, MAX_DELAY)

    # WORKLOAD GENERATOR
    workload_gen = NonStationaryWorkload(pattern=args.workload, seed=args.seed)

    # GENERATE DRL AGENTS
    iot_RL_list = list()
    for iot in range(NUM_IOT):
        iot_RL_list.append(DeepQNetwork(
            env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.99,
            replace_target_iter=200,
            memory_size=500,
            e_greedy_increment=0.001
        ))

    # TRAIN
    print("\n=== Starting Training ===")
    metrics = train(iot_RL_list, NUM_EPISODE, env_instance=env, workload_generator=workload_gen)
    print("=== Training Finished ===")

    # SAVE METRICS
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = os.path.join(args.output_dir, f"metrics_{args.workload}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
