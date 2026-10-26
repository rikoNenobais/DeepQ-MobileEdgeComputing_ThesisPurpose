"""
Full Experiment Pipeline for Paper-Aligned D3QN with .cfg Configuration Support.
Runs:
1. Workload Patterns Chart Generation
2. Stationary D3QN Training (Baseline)
3. Non-Stationary D3QN Training (Pattern from .cfg)
4. Benchmark Evaluation (All-Local, Random, Greedy, DRL)
5. Generation of all 6 Publication Figures and Final Markdown Report
"""

import os
import sys
import argparse
import json
import time

# Ensure local packages (including torch) in .venv_pkgs are found
if sys.version_info >= (3, 8):
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv_pkgs')
    if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

# Ensure matplotlib temporary directory is writable
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

from mec_env import MECEnvironment
from workload import WorkloadGenerator
from train import train_d3qn, evaluate_policy, evaluate_baseline
from config_parser import load_config
from generate_report import (
    plot_learning_convergence,
    plot_delay_evolution,
    plot_drop_rates,
    plot_offload_dynamics,
    plot_baseline_comparison,
    build_markdown_report
)


def run_paper_aligned_experiment(config_dict=None, config_file=None):
    if config_dict is None:
        config_dict = load_config(config_file)

    env_cfg = config_dict['ENVIRONMENT']
    wl_cfg = config_dict['WORKLOAD']
    exp_cfg = config_dict['EXPERIMENT']

    num_devices = env_cfg['num_devices']
    num_edge_nodes = env_cfg['num_edge_nodes']
    episodes = exp_cfg['episodes']
    nonstat_pattern = wl_cfg['pattern']
    output_dir = exp_cfg.get('output_dir', 'paper_aligned_code/reports')

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("==========================================================================")
    print("      D3QN MEC EXPERIMENT PIPELINE (STRICTLY PAPER CONFORMING)            ")
    print("==========================================================================")
    print(f"Loaded Config: {config_file if config_file else 'Default / In-Memory'}")
    print(f"Episodes per run: {episodes} | Devices (M): {num_devices} | Edge Nodes (N): {num_edge_nodes}")
    print(f"Non-Stationary Workload Pattern: {nonstat_pattern.upper()}")
    print("==========================================================================\n")

    # Step 1: Workload patterns chart
    print("[1/5] Generating Workload Patterns Comparison Chart...")
    WorkloadGenerator.plot_comparison(num_time_slots=env_cfg['num_time_slots'],
                                     save_path=os.path.join(figures_dir, 'workload_comparison.png'))

    # Step 2: Train on Stationary Workload
    print("\n[2/5] Training D3QN on STATIONARY Workload (Paper Table 1 Baseline)...")
    t0 = time.time()
    cfg_stat = json.loads(json.dumps(config_dict))
    cfg_stat['WORKLOAD']['pattern'] = 'stationary'
    metrics_stat, _, _ = train_d3qn(config_dict=cfg_stat, verbose=True)
    print(f"Stationary training completed in {time.time() - t0:.1f}s")

    # Step 3: Train on Non-Stationary Workload
    print(f"\n[3/5] Training D3QN on NON-STATIONARY ({nonstat_pattern.upper()}) Workload...")
    t0 = time.time()
    metrics_nonstat, trained_agents, env_instance = train_d3qn(config_dict=config_dict, verbose=True)
    print(f"Non-Stationary training completed in {time.time() - t0:.1f}s")

    # Save metrics JSON
    with open(os.path.join(output_dir, 'metrics_stationary.json'), 'w') as f:
        json.dump(metrics_stat, f, indent=2)
    with open(os.path.join(output_dir, f'metrics_{nonstat_pattern}.json'), 'w') as f:
        json.dump(metrics_nonstat, f, indent=2)

    # Step 4: Benchmark Baseline Heuristics
    print("\n[4/5] Evaluating Baseline Policies under Non-Stationary Workload...")
    workload_eval = WorkloadGenerator(
        pattern=nonstat_pattern,
        num_devices=num_devices,
        base_arrival_prob=wl_cfg.get('base_arrival_prob', 0.3),
        seed=123
    )
    baseline_results = {}

    eval_eps = exp_cfg.get('eval_episodes', 15)
    print("  Evaluating All-Local execution baseline...")
    baseline_results['All-Local'] = evaluate_baseline(env_instance, workload_eval, mode='local', num_episodes=eval_eps)
    print(f"    Delay: {baseline_results['All-Local']['avg_delay']:.2f} slots, Drop: {baseline_results['All-Local']['avg_drop_ratio']*100:.1f}%")

    print("  Evaluating Random offloading baseline...")
    baseline_results['Random'] = evaluate_baseline(env_instance, workload_eval, mode='random', num_episodes=eval_eps)
    print(f"    Delay: {baseline_results['Random']['avg_delay']:.2f} slots, Drop: {baseline_results['Random']['avg_drop_ratio']*100:.1f}%")

    print("  Evaluating Greedy queue-aware baseline...")
    baseline_results['Greedy'] = evaluate_baseline(env_instance, workload_eval, mode='greedy', num_episodes=eval_eps)
    print(f"    Delay: {baseline_results['Greedy']['avg_delay']:.2f} slots, Drop: {baseline_results['Greedy']['avg_drop_ratio']*100:.1f}%")

    print("  Evaluating Trained D3QN (Proposed)...")
    baseline_results['D3QN (Proposed)'] = evaluate_policy(env_instance, trained_agents, workload_eval, num_eval_episodes=eval_eps)
    print(f"    Delay: {baseline_results['D3QN (Proposed)']['avg_delay']:.2f} slots, Drop: {baseline_results['D3QN (Proposed)']['avg_drop_ratio']*100:.1f}%")

    with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    # Step 5: Visualizations & Report
    print("\n[5/5] Generating Visual Figures & Final Markdown Report...")
    plot_learning_convergence(metrics_stat, metrics_nonstat, os.path.join(figures_dir, 'learning_convergence.png'))
    plot_delay_evolution(metrics_stat, metrics_nonstat, os.path.join(figures_dir, 'delay_evolution.png'))
    plot_drop_rates(metrics_stat, metrics_nonstat, os.path.join(figures_dir, 'drop_rate_breakdown.png'))
    plot_offload_dynamics(metrics_nonstat, os.path.join(figures_dir, 'offload_dynamics.png'))
    plot_baseline_comparison(baseline_results, os.path.join(figures_dir, 'baseline_comparison.png'))

    report_path = os.path.join(output_dir, 'FINAL_REPORT_PAPER_ALIGNED.md')
    build_markdown_report(metrics_stat, metrics_nonstat, baseline_results, report_path=report_path)

    print("\n==========================================================================")
    print("                     EXPERIMENT SUCCESSFULLY COMPLETED!                   ")
    print("==========================================================================")
    print(f"Report:  {report_path}")
    print(f"Figures: {figures_dir}")
    print("==========================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Paper-Aligned Experiment with .cfg Support")
    parser.add_argument("--config", type=str, default="paper_aligned_code/config.cfg",
                        help="Path to .cfg configuration file (default: paper_aligned_code/config.cfg)")
    parser.add_argument("--episodes", type=int, default=None, help="Override number of episodes")
    parser.add_argument("--devices", type=int, default=None, help="Override number of mobile devices")
    parser.add_argument("--edge_nodes", type=int, default=None, help="Override number of edge nodes")
    parser.add_argument("--pattern", type=str, default=None, help="Override non-stationary pattern")
    args = parser.parse_args()

    # Load base config from file
    cfg = load_config(args.config)

    # CLI overrides if provided
    if args.episodes is not None: cfg['EXPERIMENT']['episodes'] = args.episodes
    if args.devices is not None: cfg['ENVIRONMENT']['num_devices'] = args.devices
    if args.edge_nodes is not None: cfg['ENVIRONMENT']['num_edge_nodes'] = args.edge_nodes
    if args.pattern is not None: cfg['WORKLOAD']['pattern'] = args.pattern

    run_paper_aligned_experiment(config_dict=cfg, config_file=args.config)

    run_paper_aligned_experiment(config_dict=cfg, config_file=args.config)

