"""
Non-Stationary Workload Generator for Mobile Edge Computing Environment.

This module provides non-stationary workload models for MEC task offloading:
1. Stationary (Baseline benchmark)
2. Sinusoidal / Diurnal (Rush hour & quiet cycles)
3. Piecewise / Multi-phase (Regime shifts / concept drift)
4. Bursty / Flash-Crowd (Sudden event-driven load spikes)
5. Markov-Modulated (Stochastic state-switching arrivals)
6. Episode-Drifting (Long-term environmental drift across training episodes)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Ensure matplotlib temporary directory is writable
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')


class NonStationaryWorkload:
    """
    Generates non-stationary task arrival matrices (bitarrive) of shape [n_time, n_iot].
    """

    SUPPORTED_PATTERNS = [
        'stationary',
        'sinusoidal',
        'piecewise',
        'bursty',
        'markovian',
        'episode_drift'
    ]

    def __init__(self,
                 pattern='sinusoidal',
                 min_bit_arrive=2.0,
                 max_bit_arrive=5.0,
                 base_task_prob=0.3,
                 amplitude=0.25,
                 period=50,
                 burst_prob=0.15,
                 burst_multiplier=2.5,
                 seed=None):
        """
        Args:
            pattern: One of ['stationary', 'sinusoidal', 'piecewise', 'bursty', 'markovian', 'episode_drift']
            min_bit_arrive: Minimum task size in Mbits.
            max_bit_arrive: Maximum task size in Mbits.
            base_task_prob: Base task arrival probability (default 0.3).
            amplitude: Fluctuation amplitude for sinusoidal/diurnal pattern.
            period: Cycle period in time slots for sinusoidal pattern.
            burst_prob: Probability of triggering a burst window in bursty mode.
            burst_multiplier: Load increase factor during a burst.
            seed: Random seed for reproducibility.
        """
        self.pattern = pattern.lower()
        if self.pattern not in self.SUPPORTED_PATTERNS:
            raise ValueError(f"Unknown pattern '{pattern}'. Supported: {self.SUPPORTED_PATTERNS}")

        self.min_bit_arrive = min_bit_arrive
        self.max_bit_arrive = max_bit_arrive
        self.base_task_prob = base_task_prob
        self.amplitude = amplitude
        self.period = period
        self.burst_prob = burst_prob
        self.burst_multiplier = burst_multiplier
        self.rng = np.random.RandomState(seed)

        # Markovian transition matrix [Low, Medium, High]
        self.markov_states = [0.15, 0.35, 0.65]  # arrival probs
        self.markov_trans = np.array([
            [0.85, 0.12, 0.03],  # from Low
            [0.10, 0.80, 0.10],  # from Medium
            [0.05, 0.15, 0.80],  # from High
        ])
        self.markov_current_state = 1

    def compute_arrival_prob(self, time_slot, n_time_base, episode=0):
        """
        Compute the expected task arrival probability at a given time slot and episode.
        """
        t = time_slot

        if self.pattern == 'stationary':
            return self.base_task_prob

        elif self.pattern == 'sinusoidal':
            # Sinusoidal diurnal wave: oscillates between (base - A) and (base + A)
            sine_val = np.sin(2.0 * np.pi * t / self.period)
            prob = self.base_task_prob + self.amplitude * sine_val
            return float(np.clip(prob, 0.05, 0.95))

        elif self.pattern == 'piecewise':
            # 3-Phase regime shift:
            # Phase 1: Off-peak (0 to 30%)
            # Phase 2: Peak rush (30% to 70%)
            # Phase 3: Moderate recovery (70% to 100%)
            ratio = t / max(1, n_time_base)
            if ratio < 0.30:
                return 0.15
            elif ratio < 0.70:
                return 0.65
            else:
                return 0.35

        elif self.pattern == 'bursty':
            # Baseline with deterministic or pseudo-stochastic bursts
            # Default bursts at [0.25-0.35] and [0.65-0.75]
            ratio = t / max(1, n_time_base)
            is_burst = (0.25 <= ratio <= 0.35) or (0.65 <= ratio <= 0.75)
            if is_burst:
                return min(0.85, self.base_task_prob * self.burst_multiplier)
            return max(0.10, self.base_task_prob * 0.7)

        elif self.pattern == 'markovian':
            # Markov chain state transition
            curr = self.markov_current_state
            next_state = self.rng.choice([0, 1, 2], p=self.markov_trans[curr])
            self.markov_current_state = next_state
            return self.markov_states[next_state]

        elif self.pattern == 'episode_drift':
            # Drift across training episodes: cyclic seasonal shift per 100 episodes
            # plus within-episode time variation
            ep_phase = np.sin(2.0 * np.pi * (episode % 200) / 200.0)
            time_phase = np.sin(2.0 * np.pi * t / self.period)
            prob = self.base_task_prob + 0.15 * ep_phase + 0.15 * time_phase
            return float(np.clip(prob, 0.08, 0.90))

        return self.base_task_prob

    def generate(self, n_time, n_iot, max_delay, episode=0):
        """
        Generate bitarrive matrix of shape [n_time, n_iot].
        The last max_delay slots have zero arrival (standard MEC buffer flushing).
        """
        n_time_base = max(1, n_time - max_delay)
        bitarrive = np.zeros((n_time, n_iot), dtype=np.float32)

        # Reset Markov state at start of episode if markovian
        if self.pattern == 'markovian':
            self.markov_current_state = self.rng.choice([0, 1, 2])

        # Track arrival probs for record
        arrival_probs = np.zeros(n_time_base, dtype=np.float32)

        for t in range(n_time_base):
            prob_t = self.compute_arrival_prob(t, n_time_base, episode)
            arrival_probs[t] = prob_t

            # Dynamic task size scaling under high load (peak traffic has larger requests)
            load_factor = prob_t / max(0.01, self.base_task_prob)
            min_size = self.min_bit_arrive * (0.85 + 0.15 * min(load_factor, 1.5))
            max_size = self.max_bit_arrive * (0.85 + 0.25 * min(load_factor, 1.5))

            # Task size uniformly sampled in [min_size, max_size]
            task_sizes = self.rng.uniform(min_size, max_size, size=n_iot)

            # Bernoulli task arrivals for each IoT device with probability prob_t
            task_arrived = (self.rng.uniform(0.0, 1.0, size=n_iot) < prob_t).astype(np.float32)

            bitarrive[t, :] = task_sizes * task_arrived

        # Last max_delay slots have 0 arrivals
        bitarrive[-max_delay:, :] = 0.0
        return bitarrive

    @classmethod
    def plot_comparison(cls, n_time=110, max_delay=10, save_path='reports/figures/workload_comparison.png'):
        """
        Generate and save a multi-panel visual comparison of all workload patterns.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        patterns = ['stationary', 'sinusoidal', 'piecewise', 'bursty', 'markovian', 'episode_drift']
        titles = [
            '1. Stationary (Baseline)',
            '2. Sinusoidal / Diurnal (Rush Hour Cycles)',
            '3. Piecewise / Multi-Phase (Regime Shifts)',
            '4. Bursty / Flash-Crowd (Traffic Spikes)',
            '5. Markov-Modulated (MMAP Stochastic)',
            '6. Episode-Drifting (Cross-Episode Drift)'
        ]
        colors = ['#4A90E2', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

        n_time_base = n_time - max_delay
        fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (pat, title, col) in enumerate(zip(patterns, titles, colors)):
            gen = cls(pattern=pat, seed=42)
            probs = [gen.compute_arrival_prob(t, n_time_base, episode=50) for t in range(n_time_base)]

            ax = axes[idx]
            ax.plot(range(n_time_base), probs, color=col, linewidth=2.2, label=r'$\lambda(t)$ (Arrival Prob)')
            ax.fill_between(range(n_time_base), 0, probs, color=col, alpha=0.18)
            ax.axhline(0.3, color='gray', linestyle='--', alpha=0.6, label='Stationary Baseline (0.30)')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
            ax.set_ylabel(r'Arrival Prob $\lambda(t)$', fontsize=10)
            ax.set_ylim(0, 1.0)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8)

        axes[4].set_xlabel('Time Slot $t$', fontsize=11)
        axes[5].set_xlabel('Time Slot $t$', fontsize=11)

        plt.suptitle('Non-Stationary Workload Patterns in Mobile Edge Computing',
                     fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Workload comparison saved to: {save_path}")
