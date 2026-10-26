"""
Automated Experiment Runner for MEC DRL Offloading with Non-Stationary Workload.
Runs:
1. Stationary baseline training
2. Non-Stationary training
3. Baseline policy benchmarks (Local, Random, Greedy, DRL)
4. Comprehensive visualization generation
5. Final report generation
"""

import os
import sys
import argparse
import json
import time
import shutil

# Ensure local packages in .venv_pkgs are found
pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv_pkgs')
if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

# Ensure matplotlib temporary directory is writable
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

from fog_env import Offload
from RL_brain import DeepQNetwork
from workload import NonStationaryWorkload
from train import train, evaluate_policy, evaluate_baseline
from generate_report import generate_all_visualizations, build_markdown_report


def run_full_experiment(episodes=60, num_iot=20, num_fog=5, pattern='sinusoidal', output_dir='reports'):
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("======================================================================")
    print("      MOBILE EDGE COMPUTING (MEC) DRL NON-STATIONARY EXPERIMENT       ")
    print("======================================================================")
    print(f"Episodes: {episodes} | IoT Devices: {num_iot} | Fog Servers: {num_fog}")
    print(f"Non-Stationary Pattern: {pattern.upper()}")
    print("======================================================================\n")

    # Generate workload comparison plot first
    print("[1/5] Generating Workload Patterns Comparison Chart...")
    NonStationaryWorkload.plot_comparison(save_path=os.path.join(figures_dir, 'workload_comparison.png'))

    # Environment setup
    num_time_base = 100
    max_delay = 10
    num_time = num_time_base + max_delay

    env_stat = Offload(num_iot, num_fog, num_time, max_delay)
    env_nonstat = Offload(num_iot, num_fog, num_time, max_delay)

    # 1. Train on Stationary Workload
    print("\n[2/5] Training DRL Agents on STATIONARY Workload...")
    workload_stat = NonStationaryWorkload(pattern='stationary', seed=42)
    agents_stat = [
        DeepQNetwork(
            env_stat.n_actions, env_stat.n_features, env_stat.n_lstm_state, env_stat.n_time,
            learning_rate=0.01, reward_decay=0.9, e_greedy=0.99,
            replace_target_iter=150, memory_size=500, e_greedy_increment=0.002
        ) for _ in range(num_iot)
    ]
    t0 = time.time()
    metrics_stat = train(agents_stat, episodes, env_instance=env_stat, workload_generator=workload_stat, verbose=True)
    print(f"Stationary training completed in {time.time() - t0:.1f}s")

    # 2. Train on Non-Stationary Workload
    print(f"\n[3/5] Training DRL Agents on NON-STATIONARY ({pattern.upper()}) Workload...")
    workload_nonstat = NonStationaryWorkload(pattern=pattern, seed=42)
    agents_nonstat = [
        DeepQNetwork(
            env_nonstat.n_actions, env_nonstat.n_features, env_nonstat.n_lstm_state, env_nonstat.n_time,
            learning_rate=0.01, reward_decay=0.9, e_greedy=0.99,
            replace_target_iter=150, memory_size=500, e_greedy_increment=0.002
        ) for _ in range(num_iot)
    ]
    t0 = time.time()
    metrics_nonstat = train(agents_nonstat, episodes, env_instance=env_nonstat, workload_generator=workload_nonstat, verbose=True)
    print(f"Non-stationary training completed in {time.time() - t0:.1f}s")

    # Save metrics JSON
    with open(os.path.join(output_dir, 'metrics_stationary.json'), 'w') as f:
        json.dump(metrics_stat, f, indent=2)
    with open(os.path.join(output_dir, f'metrics_{pattern}.json'), 'w') as f:
        json.dump(metrics_nonstat, f, indent=2)

    # 3. Benchmark Baselines under Non-Stationary Workload
    print("\n[4/5] Evaluating Baseline Policies under Non-Stationary Workload...")
    eval_env = Offload(num_iot, num_fog, num_time, max_delay)
    baseline_results = {}

    print("  Evaluating All-Local execution baseline...")
    baseline_results['All-Local'] = evaluate_baseline(eval_env, workload_nonstat, mode='local', num_episodes=15)
    print(f"    Delay: {baseline_results['All-Local']['avg_delay']:.2f} slots, Drop: {baseline_results['All-Local']['avg_drop_ratio']*100:.1f}%")

    print("  Evaluating Random offload baseline...")
    baseline_results['Random'] = evaluate_baseline(eval_env, workload_nonstat, mode='random', num_episodes=15)
    print(f"    Delay: {baseline_results['Random']['avg_delay']:.2f} slots, Drop: {baseline_results['Random']['avg_drop_ratio']*100:.1f}%")

    print("  Evaluating Greedy queue-aware baseline...")
    baseline_results['Greedy'] = evaluate_baseline(eval_env, workload_nonstat, mode='greedy', num_episodes=15)
    print(f"    Delay: {baseline_results['Greedy']['avg_delay']:.2f} slots, Drop: {baseline_results['Greedy']['avg_drop_ratio']*100:.1f}%")

    print("  Evaluating Trained DRL (Proposed)...")
    baseline_results['DRL (Proposed)'] = evaluate_policy(eval_env, agents_nonstat, workload_nonstat, num_eval_episodes=15)
    print(f"    Delay: {baseline_results['DRL (Proposed)']['avg_delay']:.2f} slots, Drop: {baseline_results['DRL (Proposed)']['avg_drop_ratio']*100:.1f}%")

    with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    # 4. Generate Visualizations & Final Report
    print("\n[5/5] Generating Visual Figures & Final Markdown Report...")
    generate_all_visualizations(metrics_stat, metrics_nonstat, baseline_results, output_dir=figures_dir)
    report_file = os.path.join(output_dir, 'FINAL_REPORT.md')
    build_markdown_report(metrics_stat, metrics_nonstat, baseline_results, report_path=report_file)

    print("\n======================================================================")
    print("                     EXPERIMENT SUCCESSFULLY COMPLETED!               ")
    print("======================================================================")
    print(f"Figures Directory: {figures_dir}")
    print(f"Final Report:      {report_file}")
    print("======================================================================\n")

    return metrics_stat, metrics_nonstat, baseline_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full MEC DRL Experiment with Non-Stationary Workload")
    parser.add_argument("--episodes", type=int, default=60, help="Number of episodes per training run")
    parser.add_argument("--iot", type=int, default=20, help="Number of IoT devices")
    parser.add_argument("--fogs", type=int, default=5, help="Number of Fog servers")
    parser.add_argument("--pattern", type=str, default="sinusoidal",
                        choices=["stationary", "sinusoidal", "piecewise", "bursty", "markovian", "episode_drift"],
                        help="Non-stationary workload pattern")
    parser.add_argument("--output_dir", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    run_full_experiment(
        episodes=args.episodes,
        num_iot=args.iot,
        num_fog=args.fogs,
        pattern=args.pattern,
        output_dir=args.output_dir
    )
