"""
Audited Workload Generator for Mobile Edge Computing (MEC) Task Offloading.
Conforms strictly to Tang & Wong (Table 1) with discrete task sizes Lambda in {2.0..5.0} Mbits,
drain period zero-padding, matched stationary controls, and deterministic trace pairing.
"""

import numpy as np
import os
import matplotlib.pyplot as plt


class AuditedWorkloadGenerator:
    """
    Workload generator supporting:
    1. 'stationary': Baseline from Table 1 of paper (p = 0.30).
    2. 'stationary_control': Matched-load stationary control (p = 0.38) with identical expected load to sudden shift.
    3. 'sudden_shift': Multi-phase regime shift (0.15 -> 0.65 -> 0.30) with weighted average p = 0.38.
    4. 'sinusoidal': Continuous diurnal oscillations.
    5. 'bursty': Flash crowd burst arrivals.
    """

    def __init__(self,
                 pattern='stationary',
                 num_devices=20,
                 base_arrival_prob=0.30,
                 min_size=2.0,
                 max_size=5.0,
                 size_step=0.1,
                 period=50,
                 seed=None):

        self.pattern = pattern.lower()
        self.M = num_devices
        self.base_prob = base_arrival_prob
        self.min_size = min_size
        self.max_size = max_size
        self.size_step = size_step
        self.period = period
        self.seed = seed

        # Discrete set Lambda = {2.0, 2.1, ..., 5.0} Mbits (Table 1)
        self.discrete_sizes = np.round(np.arange(min_size, max_size + size_step / 2, size_step), 1)
        self.rng = np.random.RandomState(seed)

    def get_arrival_probability(self, time_slot, total_slots=100, episode=0):
        t = time_slot

        if self.pattern == 'stationary':
            # Paper baseline Table 1: constant p = 0.30
            return float(self.base_prob)

        elif self.pattern == 'stationary_control':
            # Matched-load stationary control: constant p = 0.38
            return 0.38

        elif self.pattern == 'sudden_shift':
            # Multi-phase regime shift:
            # Phase 1 [0..40%): low load (0.15)
            # Phase 2 [40..80%): heavy rush hour (0.65)
            # Phase 3 [80..100%): moderate load (0.30)
            # Weighted expected probability = 0.40*0.15 + 0.40*0.65 + 0.20*0.30 = 0.380
            ratio = t / max(1, total_slots)
            if ratio < 0.40:
                return 0.15
            elif ratio < 0.80:
                return 0.65
            else:
                return 0.30

        elif self.pattern == 'sinusoidal':
            # Diurnal sinusoidal variation
            sine_wave = np.sin(2.0 * np.pi * t / self.period)
            prob = self.base_prob + 0.25 * sine_wave
            return float(np.clip(prob, 0.05, 0.95))

        elif self.pattern == 'bursty':
            ratio = t / max(1, total_slots)
            if (0.25 <= ratio <= 0.35) or (0.65 <= ratio <= 0.75):
                return 0.85
            return 0.20

        elif self.pattern == 'markovian':
            states = [0.15, 0.35, 0.65]
            return states[self.rng.choice([0, 1, 2], p=[0.2, 0.6, 0.2])]

        return float(self.base_prob)

    @classmethod
    def plot_comparison(cls, num_time_slots=100, save_path='audited_paper_code/reports/figures/workload_comparison.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        patterns = ['stationary', 'sinusoidal', 'sudden_shift', 'bursty', 'markovian']
        titles = [
            '1. Stationary (Tang & Wong Baseline)',
            '2. Sinusoidal Diurnal (Rush Hour Waves)',
            '3. Sudden Shift (Multi-Phase Regime Shift)',
            '4. Bursty / Flash Crowd (Spikes)',
            '5. Markov-Modulated (MMAP Stochastic)'
        ]
        colors = ['#2980B9', '#C0392B', '#E67E22', '#8E44AD', '#16A085']

        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (pat, title, col) in enumerate(zip(patterns, titles, colors)):
            gen = cls(pattern=pat, seed=42)
            probs = [gen.get_arrival_probability(t, num_time_slots) for t in range(num_time_slots)]
            ax = axes[idx]
            ax.plot(range(num_time_slots), probs, color=col, linewidth=2.2, label=r'$\lambda(t)$ Arrival Prob')
            ax.fill_between(range(num_time_slots), 0, probs, color=col, alpha=0.18)
            ax.axhline(0.30, color='gray', linestyle='--', alpha=0.6, label='Paper Baseline (0.30)')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel(r'Arrival Prob $\lambda(t)$', fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8)

        # Blank out last subplot
        axes[5].axis('off')
        axes[5].text(0.1, 0.4, "All tasks are strictly sampled\nfrom discrete set:\nLambda = {2.0, 2.1, ..., 5.0} Mbits\nas defined in Table 1.",
                     fontsize=11, style='italic', color='#555555')

        axes[2].set_xlabel('Time Slot $t$', fontsize=10)
        axes[3].set_xlabel('Time Slot $t$', fontsize=10)
        axes[4].set_xlabel('Time Slot $t$', fontsize=10)

        plt.suptitle('Non-Stationary Workload Formulations in MEC Systems', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Workload comparison chart saved: {save_path}")

    def generate(self, num_time_slots=100, max_deadline=10, episode=0, custom_rng=None):
        """
        Generate bit_arrive matrix of shape [total_slots, M] where total_slots = num_time_slots + max_deadline.
        Rows 0..num_time_slots-1: active arrival phase.
        Rows num_time_slots..total_slots-1: drain phase (strictly 0.0).
        """
        rng = custom_rng if custom_rng is not None else self.rng
        total_slots = num_time_slots + max_deadline
        bit_arrive = np.zeros((total_slots, self.M), dtype=np.float32)

        for t in range(num_time_slots):
            prob_t = self.get_arrival_probability(t, num_time_slots, episode)
            arrived = rng.uniform(0.0, 1.0, size=self.M) < prob_t
            sampled_sizes = rng.choice(self.discrete_sizes, size=self.M)
            bit_arrive[t, :] = sampled_sizes * arrived

        # Last tau slots are guaranteed zero arrivals for complete queue draining
        bit_arrive[num_time_slots:, :] = 0.0
        return bit_arrive

    def generate_paired_traces(self, num_episodes=15, num_time_slots=100, max_deadline=10, base_seed=1000):
        """
        Generate a fixed, deterministic set of identical traces for multi-policy benchmarking.
        Each episode gets a dedicated deterministic seed (base_seed + ep).
        """
        traces = []
        for ep in range(num_episodes):
            ep_rng = np.random.RandomState(base_seed + ep)
            trace = self.generate(num_time_slots, max_deadline, episode=ep, custom_rng=ep_rng)
            traces.append(trace)
        return traces

