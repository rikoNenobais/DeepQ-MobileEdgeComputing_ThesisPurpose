"""
Workload Generator for MEC Task Offloading Experiments.
Generates tasks strictly conforming to Tang & Wong (Table 1):
- Discrete Task Size Set: Lambda = {2.0, 2.1, ..., 5.0} Mbits
- Supports both Stationary (Baseline) and Controlled Non-Stationary Patterns
"""

import os
import numpy as np

# Ensure matplotlib temporary directory is writable
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class WorkloadGenerator:
    """
    Generates task arrival matrix [total_time_slots, num_devices] with discrete task sizes Lambda.
    """
    def __init__(self,
                 pattern='stationary',
                 num_devices=50,
                 base_arrival_prob=0.3,
                 min_size=2.0,
                 max_size=5.0,
                 size_step=0.1,
                 period=50,
                 seed=None):

        self.pattern = pattern.lower()
        self.M = num_devices
        self.base_prob = base_arrival_prob
        # Discrete set Lambda = {2.0, 2.1, ..., 5.0} Mbits
        self.discrete_sizes = np.round(np.arange(min_size, max_size + size_step / 2, size_step), 1)
        self.period = period
        self.rng = np.random.RandomState(seed)

    def get_arrival_probability(self, time_slot, total_slots, episode=0):
        t = time_slot

        if self.pattern == 'stationary':
            # Stationary baseline (Table 1): p = 0.30 constant
            return self.base_prob

        elif self.pattern == 'sinusoidal':
            # Diurnal rush-hour oscillations: swings between 0.10 and 0.60
            sine_wave = np.sin(2.0 * np.pi * t / self.period)
            prob = self.base_prob + 0.25 * sine_wave
            return float(np.clip(prob, 0.05, 0.95))

        elif self.pattern == 'sudden_shift':
            # Controlled step shift for thesis recovery study:
            # t in [0, 40): low load (0.15)
            # t in [40, 80): heavy rush hour (0.65)
            # t in [80, 100): moderate load (0.30)
            ratio = t / max(1, total_slots)
            if ratio < 0.40:
                return 0.15
            elif ratio < 0.80:
                return 0.65
            else:
                return 0.30

        elif self.pattern == 'bursty':
            # Flash crowd bursts at slots [25-35] and [65-75]
            ratio = t / max(1, total_slots)
            if (0.25 <= ratio <= 0.35) or (0.65 <= ratio <= 0.75):
                return 0.85
            return 0.20

        elif self.pattern == 'markovian':
            # 3-State Markovian switching
            states = [0.15, 0.35, 0.65]
            return states[self.rng.choice([0, 1, 2], p=[0.2, 0.6, 0.2])]

        return self.base_prob

    def generate(self, num_time_slots, max_deadline, episode=0):
        """
        Generate bit_arrive matrix of shape [T + tau, M].
        Each arriving task is strictly sampled from discrete set Lambda.
        """
        total_slots = num_time_slots + max_deadline
        bit_arrive = np.zeros((total_slots, self.M), dtype=np.float32)

        for t in range(num_time_slots):
            prob_t = self.get_arrival_probability(t, num_time_slots, episode)

            # Bernoulli arrival indicator for each mobile device
            arrived = self.rng.uniform(0.0, 1.0, size=self.M) < prob_t

            # Discrete task size sampled from Lambda
            sampled_sizes = self.rng.choice(self.discrete_sizes, size=self.M)

            bit_arrive[t, :] = sampled_sizes * arrived

        # Last tau slots have zero arrivals to drain remaining tasks
        bit_arrive[num_time_slots:, :] = 0.0
        return bit_arrive

    @classmethod
    def plot_comparison(cls, num_time_slots=100, save_path='paper_aligned_code/reports/figures/workload_comparison.png'):
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
