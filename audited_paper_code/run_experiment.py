"""
Audited Experiment Runner for MEC D3QN Task Offloading Research.
Executes the 3-Way Controlled Experiment and Paired Benchmark Evaluation:
1. Stationary Baseline (p = 0.30)
2. Stationary Control with Matched Offered Load (p = 0.38)
3. Non-Stationary Sudden Shift (p = 0.15 -> 0.65 -> 0.30, avg p = 0.38)
4. Paired Evaluation of All-Local, Random, Greedy, and D3QN on Identical Traces
5. Publication-Ready 300 DPI Figures and Comprehensive Final Markdown Report
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Ensure local package path support
if sys.version_info >= (3, 8):
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv_pkgs')
    if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

from train import train_d3qn, evaluate_policy_on_traces, evaluate_baseline_on_traces
from workload import AuditedWorkloadGenerator
from mec_env import AuditedMECEnvironment
from generate_report import (
    plot_learning_convergence,
    plot_delay_evolution,
    plot_drop_rates,
    plot_offload_dynamics,
    plot_baseline_comparison,
    plot_workload_comparison,
    plot_3way_controlled_comparison,
    plot_transmission_queue_diagnostics,
    generate_audited_markdown_report
)


def run_full_audit_experiment(episodes=30, eval_episodes=15, output_dir='audited_paper_code/reports'):
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("================================================================================")
    print("           AUDITED D3QN MEC EXPERIMENT RUNNER (STRICT CONFORMING)              ")
    print("================================================================================")
    print(f"Output Directory: {output_dir}")
    print(f"Training Episodes: {episodes} per condition | Paired Eval Episodes: {eval_episodes}")

    # --------------------------------------------------------------------------
    # Step 1: Condition 1 - Stationary Baseline (p = 0.30)
    # --------------------------------------------------------------------------
    print("\n[1/4] Training Condition 1: Stationary Baseline (p = 0.30, Paper Table 1)...")
    metrics_base, _, _ = train_d3qn(
        num_episodes=episodes,
        num_devices=20,
        workload_pattern='stationary',
        base_arrival_prob=0.30,
        verbose=True
    )
    with open(os.path.join(output_dir, 'metrics_stationary_baseline.json'), 'w') as f:
        json.dump(metrics_base, f, indent=2)

    # --------------------------------------------------------------------------
    # Step 2: Condition 2 - Stationary Control (Matched Load p = 0.38)
    # --------------------------------------------------------------------------
    print("\n[2/4] Training Condition 2: Stationary Control (Matched Offered Load p = 0.38)...")
    metrics_ctrl, _, _ = train_d3qn(
        num_episodes=episodes,
        num_devices=20,
        workload_pattern='stationary_control',
        base_arrival_prob=0.38,
        verbose=True
    )
    with open(os.path.join(output_dir, 'metrics_stationary_control.json'), 'w') as f:
        json.dump(metrics_ctrl, f, indent=2)

    # --------------------------------------------------------------------------
    # Step 3: Condition 3 - Non-Stationary Sudden Shift (avg p = 0.38)
    # --------------------------------------------------------------------------
    print("\n[3/4] Training Condition 3: Non-Stationary Sudden Shift (p: 0.15 -> 0.65 -> 0.30)...")
    metrics_shift, trained_shift_agents, env_shift = train_d3qn(
        num_episodes=episodes,
        num_devices=20,
        workload_pattern='sudden_shift',
        base_arrival_prob=0.38,
        verbose=True
    )
    with open(os.path.join(output_dir, 'metrics_sudden_shift.json'), 'w') as f:
        json.dump(metrics_shift, f, indent=2)

    # --------------------------------------------------------------------------
    # Step 4: Paired Evaluation on Identical Pre-Generated Traces
    # --------------------------------------------------------------------------
    print("\n[4/4] Conducting Paired Multi-Policy Evaluation on Identical Traces...")
    workload_eval = AuditedWorkloadGenerator(
        pattern='sudden_shift',
        num_devices=20,
        base_arrival_prob=0.38,
        seed=999
    )
    eval_traces = workload_eval.generate_paired_traces(
        num_episodes=eval_episodes,
        num_time_slots=100,
        max_deadline=10,
        base_seed=999
    )

    eval_env = AuditedMECEnvironment(num_devices=20, num_time_slots=100, max_deadline=10)

    baseline_results = {}
    print("  Evaluating All-Local policy...")
    baseline_results['All-Local'] = evaluate_baseline_on_traces(eval_env, eval_traces, mode='local')

    print("  Evaluating Random Offloading policy...")
    baseline_results['Random'] = evaluate_baseline_on_traces(eval_env, eval_traces, mode='random', random_seed=42)

    print("  Evaluating Greedy Queue-Aware policy...")
    baseline_results['Greedy Queue'] = evaluate_baseline_on_traces(eval_env, eval_traces, mode='greedy')

    print("  Evaluating Proposed D3QN agent...")
    baseline_results['D3QN (Proposed)'] = evaluate_policy_on_traces(eval_env, trained_shift_agents, eval_traces)

    with open(os.path.join(output_dir, 'baseline_results_paired.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    print("\nBenchmark Summary (Mean +/- Std):")
    for pol, res in baseline_results.items():
        print(f"  {pol:18s} | Delay: {res['mean_delay']:4.2f} +/- {res['std_delay']:4.2f}s | "
              f"Drop: {res['mean_drop_ratio']*100:5.1f}% +/- {res['std_drop_ratio']*100:4.1f}% | "
              f"Cost: {res['mean_cost']:5.2f} +/- {res['std_cost']:4.2f}")

    # --------------------------------------------------------------------------
    # Step 5: Visual Figures & Report Generation (The Full Suite Matching Pre-Audit)
    # --------------------------------------------------------------------------
    print("\nGenerating 300 DPI Visual Figures (Full Suite Matching Pre-Audit)...")
    
    # 1. workload_comparison.png
    fig_workload = os.path.join(figures_dir, 'workload_comparison.png')
    plot_workload_comparison(fig_workload)
    print(f"  Saved: {fig_workload}")

    # 2. learning_convergence.png
    fig_conv = os.path.join(figures_dir, 'learning_convergence.png')
    plot_learning_convergence(metrics_ctrl, metrics_shift, fig_conv)
    print(f"  Saved: {fig_conv}")

    # 3. delay_evolution.png
    fig_delay = os.path.join(figures_dir, 'delay_evolution.png')
    plot_delay_evolution(metrics_ctrl, metrics_shift, fig_delay)
    print(f"  Saved: {fig_delay}")

    # 4. drop_rate_breakdown.png
    fig_drops = os.path.join(figures_dir, 'drop_rate_breakdown.png')
    plot_drop_rates(metrics_ctrl, metrics_shift, fig_drops)
    print(f"  Saved: {fig_drops}")

    # 5. offload_dynamics.png
    fig_offload = os.path.join(figures_dir, 'offload_dynamics.png')
    plot_offload_dynamics(metrics_shift, fig_offload)
    print(f"  Saved: {fig_offload}")

    # 6. baseline_comparison.png
    fig_base = os.path.join(figures_dir, 'baseline_comparison.png')
    plot_baseline_comparison(baseline_results, fig_base)
    print(f"  Saved: {fig_base}")

    # Diagnostic additions
    fig_3way = os.path.join(figures_dir, 'workload_controlled_comparison.png')
    plot_3way_controlled_comparison(metrics_base, metrics_ctrl, metrics_shift, fig_3way)
    print(f"  Saved: {fig_3way}")

    fig_tran = os.path.join(figures_dir, 'transmission_diagnostics.png')
    plot_transmission_queue_diagnostics(metrics_shift['slot_diagnostics_last_ep'], fig_tran)
    print(f"  Saved: {fig_tran}")

    report_path = os.path.join(output_dir, 'FINAL_AUDIT_REPORT.md')
    generate_audited_markdown_report(metrics_base, metrics_ctrl, metrics_shift, baseline_results, report_path)

    print("\n================================================================================")
    print("                     ALL EXPERIMENTS COMPLETED SUCCESSFULLY                     ")
    print("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audited D3QN MEC Experiment Runner")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per training condition")
    parser.add_argument("--eval_episodes", type=int, default=15, help="Paired evaluation episodes")
    parser.add_argument("--output_dir", type=str, default="audited_paper_code/reports", help="Output directory")
    args = parser.parse_args()

    run_full_audit_experiment(episodes=args.episodes, eval_episodes=args.eval_episodes, output_dir=args.output_dir)

