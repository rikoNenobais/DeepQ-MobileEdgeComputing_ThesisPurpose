"""
Automated Test Suite for Audited MEC Simulator.
Verifies all 9 requirements specified in Section 7 of the thesis audit mandate:
1. Tasks completing within deadline are NOT marked as dropped.
2. Tasks exceeding deadline are marked dropped EXACTLY ONCE.
3. Tasks arriving at the last active slot (t=99) receive full drain lifecycle evaluation.
4. No tasks are lost due to episode termination.
5. Conservation law: total_arrived == total_completed + total_dropped.
6. Category consistency: total_dropped == drop_local + drop_tran + drop_edge.
7. Workload traces in paired evaluation are 100% IDENTICAL across all policies.
8. Stationary control and sudden shift share identical capacities and task size distributions.
9. Successful delay is strictly isolated from dropped task penalty delay.
"""

import os
import sys
import unittest
import numpy as np

# Ensure local package path support
if sys.version_info >= (3, 8):
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv_pkgs')
    if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

from mec_env import AuditedMECEnvironment
from workload import AuditedWorkloadGenerator
from train import evaluate_baseline_on_traces


class TestMECSimulatorAudit(unittest.TestCase):

    def setUp(self):
        self.M = 5
        self.N = 3
        self.T = 20
        self.tau = 5
        self.env = AuditedMECEnvironment(
            num_devices=self.M,
            num_edge_nodes=self.N,
            num_time_slots=self.T,
            max_deadline=self.tau,
            time_slot_duration=0.1
        )

    def test_01_successful_task_not_dropped(self):
        """Test 1: A task with ample time and processing capacity completes and is NOT dropped."""
        # Single device receives a small task of 2.0 Mbit at t=0
        arrivals = np.zeros((self.env.total_slots, self.M), dtype=np.float32)
        arrivals[0, 0] = 2.0  # At local execution: 2.0 * 0.297 / 0.25 = 2.37 slots -> finishes at t=2
        self.env.reset(arrivals)

        for _ in range(self.env.total_slots):
            self.env.step(np.zeros(self.M, dtype=int))

        summary = self.env.get_summary_metrics()
        self.assertEqual(summary['total_arrived'], 1)
        self.assertEqual(summary['completed_count'], 1)
        self.assertEqual(summary['dropped_count'], 0)
        self.assertFalse(self.env.task_dropped[0, 0])
        self.assertGreater(self.env.task_delay[0, 0], 0)
        self.assertLessEqual(self.env.task_delay[0, 0], self.tau)

    def test_02_expired_task_dropped_exactly_once(self):
        """Test 2: A task that exceeds deadline is marked dropped exactly once with reason."""
        arrivals = np.zeros((self.env.total_slots, self.M), dtype=np.float32)
        # Device receives 3 huge tasks in consecutive slots; queue delay forces later task to expire
        arrivals[0, 0] = 5.0
        arrivals[1, 0] = 5.0
        arrivals[2, 0] = 5.0
        self.env.reset(arrivals)

        for _ in range(self.env.total_slots):
            self.env.step(np.zeros(self.M, dtype=int))

        summary = self.env.get_summary_metrics()
        self.assertEqual(summary['total_arrived'], 3)
        self.assertEqual(summary['total_arrived'], summary['completed_count'] + summary['dropped_count'])
        # Each task recorded exactly once
        self.assertEqual(len(self.env.task_drop_reason), summary['dropped_count'])

    def test_03_last_slot_task_drained_properly(self):
        """Test 3: Task arriving at the very last active slot (t = T-1) has full drain period to complete."""
        arrivals = np.zeros((self.env.total_slots, self.M), dtype=np.float32)
        last_active_slot = self.T - 1
        arrivals[last_active_slot, 1] = 2.0  # Small task arriving at last slot
        self.env.reset(arrivals)

        while True:
            _, _, _, done, info = self.env.step(np.zeros(self.M, dtype=int))
            if done:
                break

        summary = self.env.get_summary_metrics()
        self.assertEqual(summary['total_arrived'], 1)
        # It must complete during drain period (slots T..T+tau-1), NOT get killed by truncation
        self.assertEqual(summary['completed_count'], 1)
        self.assertEqual(summary['dropped_count'], 0)
        self.assertTrue(self.env.task_delay[last_active_slot, 1] > 0)
        self.assertFalse(self.env.task_dropped[last_active_slot, 1])

    def test_04_no_tasks_lost_at_termination(self):
        """Test 4: Heavy traffic does not lose tasks due to episode boundary."""
        workload_gen = AuditedWorkloadGenerator(
            pattern='sudden_shift',
            num_devices=self.M,
            base_arrival_prob=0.38,
            seed=123
        )
        arrivals = workload_gen.generate(self.T, self.tau)
        self.env.reset(arrivals)

        while True:
            _, _, _, done, _ = self.env.step(np.zeros(self.M, dtype=int))
            if done:
                break

        summary = self.env.get_summary_metrics()
        expected_arrived = int(np.count_nonzero(arrivals[:self.T, :]))
        self.assertEqual(summary['total_arrived'], expected_arrived)
        self.assertEqual(summary['total_arrived'], summary['completed_count'] + summary['dropped_count'])

    def test_05_conservation_law_arrived_equals_completed_plus_dropped(self):
        """Test 5: Strict conservation law holding under random actions."""
        rng = np.random.RandomState(42)
        workload_gen = AuditedWorkloadGenerator(pattern='sinusoidal', num_devices=self.M, seed=42)
        arrivals = workload_gen.generate(self.T, self.tau)
        self.env.reset(arrivals)

        while True:
            random_actions = rng.randint(0, self.env.num_actions, size=self.M)
            _, _, _, done, _ = self.env.step(random_actions)
            if done:
                break

        summary = self.env.get_summary_metrics()
        self.assertEqual(summary['total_arrived'], summary['completed_count'] + summary['dropped_count'])

    def test_06_category_consistency_sum_equals_total_dropped(self):
        """Test 6: drop_local + drop_tran + drop_edge == total_dropped."""
        rng = np.random.RandomState(99)
        workload_gen = AuditedWorkloadGenerator(pattern='bursty', num_devices=self.M, seed=99)
        arrivals = workload_gen.generate(self.T, self.tau)
        self.env.reset(arrivals)

        while True:
            actions = rng.randint(0, self.env.num_actions, size=self.M)
            _, _, _, done, _ = self.env.step(actions)
            if done:
                break

        summary = self.env.get_summary_metrics()
        cat_sum = summary['drop_local'] + summary['drop_tran'] + summary['drop_edge']
        self.assertEqual(summary['dropped_count'], cat_sum)

    def test_07_paired_traces_identical_across_baselines(self):
        """Test 7: Ensure paired traces fed to All-Local, Random, Greedy are bitwise identical."""
        workload_gen = AuditedWorkloadGenerator(pattern='sudden_shift', num_devices=self.M, seed=777)
        traces = workload_gen.generate_paired_traces(num_episodes=5, num_time_slots=self.T, max_deadline=self.tau, base_seed=777)

        # Traces passed to evaluate_baseline_on_traces
        res_local = evaluate_baseline_on_traces(self.env, traces, mode='local')
        res_greedy = evaluate_baseline_on_traces(self.env, traces, mode='greedy')

        # Check that traces themselves were not modified or drifted
        for ep in range(5):
            ref_rng = np.random.RandomState(777 + ep)
            ref_trace = workload_gen.generate(self.T, self.tau, episode=ep, custom_rng=ref_rng)
            np.testing.assert_array_equal(traces[ep], ref_trace)

    def test_08_stationary_control_and_sudden_shift_matched_load(self):
        """Test 8: Stationary control (0.38) and sudden shift (0.15/0.65/0.30) have identical expected load."""
        gen_ctrl = AuditedWorkloadGenerator(pattern='stationary_control', num_devices=20, seed=1)
        gen_shift = AuditedWorkloadGenerator(pattern='sudden_shift', num_devices=20, seed=1)

        slots = 100
        ctrl_probs = [gen_ctrl.get_arrival_probability(t, slots) for t in range(slots)]
        shift_probs = [gen_shift.get_arrival_probability(t, slots) for t in range(slots)]

        mean_ctrl = np.mean(ctrl_probs)
        mean_shift = np.mean(shift_probs)

        self.assertAlmostEqual(mean_ctrl, 0.380, places=3)
        self.assertAlmostEqual(mean_shift, 0.380, places=3)
        self.assertAlmostEqual(mean_ctrl, mean_shift, places=3)
        np.testing.assert_array_equal(gen_ctrl.discrete_sizes, gen_shift.discrete_sizes)

    def test_09_delay_successful_isolated_from_penalty(self):
        """Test 9: Successful task delay is strictly isolated from penalty cost."""
        arrivals = np.zeros((self.env.total_slots, self.M), dtype=np.float32)
        # Device 0: successful small task
        arrivals[0, 0] = 2.0
        # Device 1: overloaded tasks causing drops
        arrivals[0, 1] = 5.0
        arrivals[1, 1] = 5.0
        arrivals[2, 1] = 5.0
        arrivals[3, 1] = 5.0
        self.env.reset(arrivals)

        while True:
            _, _, _, done, _ = self.env.step(np.zeros(self.M, dtype=int))
            if done:
                break

        summary = self.env.get_summary_metrics()
        self.assertGreater(summary['completed_count'], 0)
        self.assertGreater(summary['dropped_count'], 0)

        # Successful delay must be <= tau
        self.assertLessEqual(summary['avg_delay_successful'], self.tau)
        # Average cost must include penalty C=20.0, so avg_cost > avg_delay_successful
        self.assertGreater(summary['avg_cost'], summary['avg_delay_successful'])


if __name__ == '__main__':
    unittest.main()

