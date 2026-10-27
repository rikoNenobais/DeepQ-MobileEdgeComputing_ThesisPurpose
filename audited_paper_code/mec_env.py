"""
Audited MEC Environment Simulator conforming to Tang & Wong (IEEE TMC 2020 / 2022).
Implements exact mathematical models with strict task lifecycle, drain period,
exact deadline accounting, and no premature terminal truncations.
"""

import numpy as np
import queue
import math


class AuditedMECEnvironment:
    """
    Audited Mobile Edge Computing (MEC) System with M mobile devices and N edge nodes.
    Features:
    1. Two-phase horizon: Active arrival phase (0..T-1) and Drain period (T..T+tau-1).
    2. Strict Task Lifecycle: Every task is tracked by a unique ID and resolves to
       EITHER completed OR dropped with an exact physical reason.
    3. Mathematical conservation: total_arrived == total_completed + total_dropped.
    4. Exact drop category consistency: total_dropped == drop_local + drop_tran + drop_edge.
    5. Clean transmission-to-edge handover: transmission finishing at slot t enters
       the edge node queue at the beginning of slot t+1.
    6. Comprehensive slot-level diagnostic logging for queue and transmission timeouts.
    """

    def __init__(self,
                 num_devices=20,          # M
                 num_edge_nodes=5,        # N
                 num_time_slots=100,      # T (Active arrival phase)
                 max_deadline=10,         # tau_m (in slots, e.g. 1.0 second)
                 time_slot_duration=0.1,  # Delta = 0.1 second
                 f_device=2.5,            # 2.5 GHz
                 f_edge=41.8,             # 41.8 GHz
                 r_tran=14.0,             # 14 Mbps
                 comp_density=0.297,      # 0.297 gigacycles per Mbit
                 task_arrival_prob=0.3,   # Default arrival prob
                 t_step=10,               # T_step for historical load H(t)
                 penalty_cost=20.0):      # Constant penalty C for dropped tasks

        self.M = num_devices
        self.N = num_edge_nodes
        self.T = num_time_slots          # Active arrival slots (0..T-1)
        self.tau = max_deadline          # Deadline duration in slots
        self.total_slots = self.T + self.tau  # Total episode slots including drain phase
        self.Delta = time_slot_duration
        self.T_step = t_step
        self.C = penalty_cost

        # Computational & Transmission Capacities (per time slot)
        # 1 Gigacycle = 10^9 cycles. Capacity per time slot = f * Delta
        self.f_device = f_device * self.Delta  # Gigacycles per slot (0.25)
        self.f_edge = f_edge * self.Delta      # Gigacycles per slot (4.18)
        self.r_tran = r_tran * self.Delta      # Mbits per slot (1.4)
        self.rho = comp_density                # Gigacycles per Mbit (0.297)

        # Discrete task size set Lambda = {2.0, 2.1, ..., 5.0} Mbits (Table 1)
        self.task_size_set = np.round(np.arange(2.0, 5.05, 0.1), 1)
        self.task_arrival_prob = task_arrival_prob

        # Action space: 0 = Local, 1 = Edge 1, ..., N = Edge N (Total 1 + N choices)
        self.num_actions = 1 + self.N

        # State dimensions:
        # local_features: [lambda_m(t), delta_comp_m(t), delta_tran_m(t), q_edge_m_1..N(t-1)]
        self.num_local_features = 3 + self.N
        self.lstm_seq_len = self.T_step
        self.lstm_feature_dim = self.N

        self.time_slot = 0
        self.reset()

    def reset(self, bit_arrive_matrix=None):
        """
        Reset the environment for a new episode.
        bit_arrive_matrix: matrix of shape [total_slots, M] specifying task arrivals.
        """
        self.time_slot = 0

        if bit_arrive_matrix is not None:
            self.bit_arrive = bit_arrive_matrix.copy()
            # Ensure drain phase (T..total_slots) has 0 arrivals
            if len(self.bit_arrive) > self.T:
                self.bit_arrive[self.T:, :] = 0.0
        else:
            self.bit_arrive = np.zeros((self.total_slots, self.M), dtype=np.float32)
            for t in range(self.T):
                arrival_mask = np.random.uniform(0.0, 1.0, size=self.M) < self.task_arrival_prob
                random_sizes = np.random.choice(self.task_size_set, size=self.M)
                self.bit_arrive[t, :] = random_sizes * arrival_mask
            self.bit_arrive[self.T:, :] = 0.0

        # FIFO Queue structures
        self.queue_device_comp = [queue.Queue() for _ in range(self.M)]
        self.queue_device_tran = [queue.Queue() for _ in range(self.M)]
        self.queue_edge_comp = [[queue.Queue() for _ in range(self.N)] for _ in range(self.M)]

        # Intermediate buffer for tasks completing transmission at slot t, arriving at edge at t+1
        self.edge_transit_buffer = []

        # Tracking end-of-queue completion time slots for state estimation (Eq. 2 & 5)
        self.l_comp_latest = -np.ones(self.M, dtype=int)
        self.l_tran_latest = -np.ones(self.M, dtype=int)

        # Edge queue length vector q_edge_m(t-1) for device m at edge nodes: shape [M, N] (Eq. 8)
        self.q_edge = np.zeros((self.M, self.N), dtype=np.float32)

        # Active processing slots
        self.current_proc_local = [{'size': 0.0, 'time': -1, 'remain': np.nan, 'task_id': None} for _ in range(self.M)]
        self.current_tran_local = [{'size': 0.0, 'time': -1, 'edge': -1, 'remain': np.nan, 'task_id': None, 'tran_time': 0} for _ in range(self.M)]
        self.current_proc_edge = [[{'size': 0.0, 'time': -1, 'remain': np.nan, 'task_id': None} for _ in range(self.N)] for _ in range(self.M)]

        # Edge node load broadcast history: shape [T_step, N]
        self.load_history = np.zeros((self.T_step, self.N), dtype=np.float32)
        self.B_n = np.zeros(self.N, dtype=np.float32)

        # Performance and lifecycle record keepers
        self.task_cost = np.zeros((self.total_slots, self.M), dtype=np.float32)
        self.task_delay = np.zeros((self.total_slots, self.M), dtype=np.float32)
        self.task_dropped = np.zeros((self.total_slots, self.M), dtype=bool)
        self.task_drop_reason = {}  # task_id -> reason string

        # Exact accounting counters
        self.total_arrived_count = int(np.count_nonzero(self.bit_arrive[:self.T, :]))
        self.total_completed_count = 0
        self.total_dropped_count = 0
        self.drop_local_count = 0
        self.drop_tran_count = 0
        self.drop_edge_count = 0

        # Detailed per-slot diagnostic log
        self.diagnostic_logs = []

        return self._get_all_states()

    def _get_waiting_time_comp(self, m, t):
        """Eq. (2): delta_comp_m(t) waiting time for computation queue."""
        if self.l_comp_latest[m] < 0:
            return 0
        return max(0, self.l_comp_latest[m] - t + 1)

    def _get_waiting_time_tran(self, m, t):
        """Eq. (5): delta_tran_m(t) waiting time for transmission queue."""
        if self.l_tran_latest[m] < 0:
            return 0
        return max(0, self.l_tran_latest[m] - t + 1)

    def _get_all_states(self):
        """
        Build state s_m(t) according to Eq. (13):
        sm(t) = (lambda_m(t), delta_comp_m(t), delta_tran_m(t), q_edge_m(t-1), H(t))
        """
        local_states = np.zeros((self.M, self.num_local_features), dtype=np.float32)
        H_matrices = np.zeros((self.M, self.T_step, self.N), dtype=np.float32)

        t = self.time_slot
        for m in range(self.M):
            task_bits = self.bit_arrive[t, m] if t < self.T else 0.0

            if task_bits > 0.0:
                delta_comp = self._get_waiting_time_comp(m, t)
                delta_tran = self._get_waiting_time_tran(m, t)
                q_edge_prev = self.q_edge[m, :]
                local_states[m, :] = np.hstack([task_bits, delta_comp, delta_tran, q_edge_prev])

            H_matrices[m, :, :] = self.load_history

        return local_states, H_matrices

    def step(self, actions):
        """
        Execute one time slot.
        actions: list or array of length M with values in {0, 1, ..., N}.
        Returns:
            next_local_states, next_H_matrices, finished_experiences, done, info
        """
        t = self.time_slot
        finished_experiences = [[] for _ in range(self.M)]

        # Slot diagnostic counters
        slot_arrived = 0
        slot_comp_drops = 0
        slot_tran_drops = 0
        slot_edge_drops = 0
        slot_completions = 0

        # ----------------------------------------------------------------------
        # 0. DELIVER TRANSIT BUFFER TO EDGE QUEUES (Transmissions finished at t-1)
        # ----------------------------------------------------------------------
        for transit_item in self.edge_transit_buffer:
            m = transit_item['m']
            n = transit_item['edge']
            self.queue_edge_comp[m][n].put(transit_item['task'])
            self.q_edge[m, n] += transit_item['task']['size']
        self.edge_transit_buffer = []

        # ----------------------------------------------------------------------
        # 1. ENQUEUE NEW ARRIVING TASKS (Only during active arrival phase t < T)
        # ----------------------------------------------------------------------
        if t < self.T:
            for m in range(self.M):
                task_bits = self.bit_arrive[t, m]
                if task_bits > 0.0:
                    slot_arrived += 1
                    act = int(actions[m]) if actions is not None else 0
                    task_item = {
                        'task_id': (t, m),
                        'size': task_bits,
                        'time': t,
                        'edge': -1,
                        'wait_slots': 0
                    }

                    if act == 0:
                        # Local Computation Queue (Section 2.1.3)
                        self.queue_device_comp[m].put(task_item)
                        delta_comp = self._get_waiting_time_comp(m, t)
                        service_slots = math.ceil(task_bits * self.rho / self.f_device)
                        predicted_finish = t + delta_comp + service_slots - 1
                        deadline_slot = t + self.tau - 1
                        self.l_comp_latest[m] = min(predicted_finish, deadline_slot)
                    else:
                        # Transmission Queue for Offloading (Section 2.1.4)
                        edge_idx = act - 1
                        task_item['edge'] = edge_idx
                        self.queue_device_tran[m].put(task_item)
                        delta_tran = self._get_waiting_time_tran(m, t)
                        tran_slots = math.ceil(task_bits / self.r_tran)
                        predicted_finish = t + delta_tran + tran_slots - 1
                        deadline_slot = t + self.tau - 1
                        self.l_tran_latest[m] = min(predicted_finish, deadline_slot)

        # ----------------------------------------------------------------------
        # 2. LOCAL COMPUTATION QUEUE PROCESSING (Section 2.1.3)
        # ----------------------------------------------------------------------
        for m in range(self.M):
            # If processor idle, fetch next non-expired task
            if math.isnan(self.current_proc_local[m]['remain']) and not self.queue_device_comp[m].empty():
                while not self.queue_device_comp[m].empty():
                    candidate = self.queue_device_comp[m].get()
                    # Age check: elapsed slots including current slot
                    elapsed = t - candidate['time'] + 1
                    if elapsed <= self.tau:
                        self.current_proc_local[m] = {
                            'task_id': candidate['task_id'],
                            'size': candidate['size'],
                            'time': candidate['time'],
                            'remain': candidate['size']
                        }
                        break
                    else:
                        # Expired waiting in queue
                        self._record_drop(m, candidate['time'], candidate['task_id'], reason='comp_timeout')
                        finished_experiences[m].append(candidate['time'])
                        slot_comp_drops += 1

            # Process active local task
            if self.current_proc_local[m]['remain'] > 0:
                self.current_proc_local[m]['remain'] -= (self.f_device / self.rho)
                elapsed = t - self.current_proc_local[m]['time'] + 1

                if self.current_proc_local[m]['remain'] <= 0:
                    # Successfully completed
                    delay = elapsed
                    self._record_finished(m, self.current_proc_local[m]['time'], self.current_proc_local[m]['task_id'], delay)
                    finished_experiences[m].append(self.current_proc_local[m]['time'])
                    self.current_proc_local[m]['remain'] = np.nan
                    slot_completions += 1
                elif elapsed >= self.tau:
                    # Deadline expired during computation
                    self._record_drop(m, self.current_proc_local[m]['time'], self.current_proc_local[m]['task_id'], reason='comp_timeout')
                    finished_experiences[m].append(self.current_proc_local[m]['time'])
                    self.current_proc_local[m]['remain'] = np.nan
                    slot_comp_drops += 1

        # ----------------------------------------------------------------------
        # 3. TRANSMISSION QUEUE PROCESSING (Section 2.1.4)
        # ----------------------------------------------------------------------
        for m in range(self.M):
            # If transmitter idle, fetch next non-expired task
            if math.isnan(self.current_tran_local[m]['remain']) and not self.queue_device_tran[m].empty():
                while not self.queue_device_tran[m].empty():
                    candidate = self.queue_device_tran[m].get()
                    elapsed = t - candidate['time'] + 1
                    if elapsed <= self.tau:
                        self.current_tran_local[m] = {
                            'task_id': candidate['task_id'],
                            'size': candidate['size'],
                            'time': candidate['time'],
                            'edge': candidate['edge'],
                            'remain': candidate['size'],
                            'tran_time': 0
                        }
                        break
                    else:
                        # Expired waiting in transmission queue
                        self._record_drop(m, candidate['time'], candidate['task_id'], reason='tran_timeout')
                        finished_experiences[m].append(candidate['time'])
                        slot_tran_drops += 1

            # Transmit active task (Dedicated bandwidth: 1.4 Mbit/slot per device)
            if self.current_tran_local[m]['remain'] > 0:
                self.current_tran_local[m]['remain'] -= self.r_tran
                self.current_tran_local[m]['tran_time'] += 1
                elapsed = t - self.current_tran_local[m]['time'] + 1
                edge_idx = self.current_tran_local[m]['edge']

                if self.current_tran_local[m]['remain'] <= 0:
                    # Transmission completed successfully at end of slot t
                    # Place in transit buffer; arrives at edge queue at beginning of slot t+1
                    self.edge_transit_buffer.append({
                        'm': m,
                        'edge': edge_idx,
                        'task': {
                            'task_id': self.current_tran_local[m]['task_id'],
                            'size': self.current_tran_local[m]['size'],
                            'time': self.current_tran_local[m]['time']
                        }
                    })
                    self.current_tran_local[m]['remain'] = np.nan
                elif elapsed >= self.tau:
                    # Expired during transmission
                    self._record_drop(m, self.current_tran_local[m]['time'], self.current_tran_local[m]['task_id'], reason='tran_timeout')
                    finished_experiences[m].append(self.current_tran_local[m]['time'])
                    self.current_tran_local[m]['remain'] = np.nan
                    slot_tran_drops += 1

        # ----------------------------------------------------------------------
        # 4. EDGE NODE PROCESSING WITH GPS SHARING (Section 2.2.1)
        # ----------------------------------------------------------------------
        # Determine active queues B_n(t) across all devices for each edge node
        B_n_current = np.zeros(self.N, dtype=np.float32)
        for n in range(self.N):
            active_count = 0
            for m in range(self.M):
                if not self.queue_edge_comp[m][n].empty() or self.current_proc_edge[m][n]['remain'] > 0:
                    active_count += 1
            B_n_current[n] = max(1.0, float(active_count))

        self.B_n = B_n_current

        for n in range(self.N):
            allocated_capacity = self.f_edge / (self.rho * self.B_n[n])

            for m in range(self.M):
                # If edge processor idle, fetch next task
                if math.isnan(self.current_proc_edge[m][n]['remain']) and not self.queue_edge_comp[m][n].empty():
                    while not self.queue_edge_comp[m][n].empty():
                        candidate = self.queue_edge_comp[m][n].get()
                        elapsed = t - candidate['time'] + 1
                        if elapsed <= self.tau:
                            self.current_proc_edge[m][n] = {
                                'task_id': candidate['task_id'],
                                'size': candidate['size'],
                                'time': candidate['time'],
                                'remain': candidate['size']
                            }
                            break
                        else:
                            # Expired in edge queue
                            self._record_drop(m, candidate['time'], candidate['task_id'], reason='edge_timeout')
                            finished_experiences[m].append(candidate['time'])
                            self.q_edge[m, n] = max(0.0, self.q_edge[m, n] - candidate['size'])
                            slot_edge_drops += 1

                # Execute edge task with GPS sharing
                if self.current_proc_edge[m][n]['remain'] > 0:
                    self.current_proc_edge[m][n]['remain'] -= allocated_capacity
                    self.q_edge[m, n] = max(0.0, self.q_edge[m, n] - allocated_capacity)
                    elapsed = t - self.current_proc_edge[m][n]['time'] + 1

                    if self.current_proc_edge[m][n]['remain'] <= 0:
                        delay = elapsed
                        self._record_finished(m, self.current_proc_edge[m][n]['time'], self.current_proc_edge[m][n]['task_id'], delay)
                        finished_experiences[m].append(self.current_proc_edge[m][n]['time'])
                        self.current_proc_edge[m][n]['remain'] = np.nan
                        slot_completions += 1
                    elif elapsed >= self.tau:
                        self._record_drop(m, self.current_proc_edge[m][n]['time'], self.current_proc_edge[m][n]['task_id'], reason='edge_timeout')
                        finished_experiences[m].append(self.current_proc_edge[m][n]['time'])
                        self.current_proc_edge[m][n]['remain'] = np.nan
                        slot_edge_drops += 1

        # ----------------------------------------------------------------------
        # 5. BROADCAST ACTIVE QUEUES & UPDATE LOAD HISTORY H(t) (Eq. 12)
        # ----------------------------------------------------------------------
        self.load_history = np.vstack([self.load_history[1:], self.B_n])

        # Diagnostic slot snapshot
        tran_q_lens = [self.queue_device_tran[m].qsize() for m in range(self.M)]
        comp_q_lens = [self.queue_device_comp[m].qsize() for m in range(self.M)]
        self.diagnostic_logs.append({
            'slot': t,
            'is_drain_phase': (t >= self.T),
            'arrivals': slot_arrived,
            'completions': slot_completions,
            'drops_comp': slot_comp_drops,
            'drops_tran': slot_tran_drops,
            'drops_edge': slot_edge_drops,
            'tran_queue_total': int(sum(tran_q_lens)),
            'tran_queue_max': int(max(tran_q_lens) if tran_q_lens else 0),
            'comp_queue_total': int(sum(comp_q_lens)),
            'edge_loads': self.B_n.tolist()
        })

        # Advance slot time
        self.time_slot += 1

        # Episode terminates strictly after the full drain phase finishes (slot total_slots)
        done = (self.time_slot >= self.total_slots)

        next_local_states, next_H_matrices = self._get_all_states()
        info = {
            'is_drain_phase': (self.time_slot >= self.T),
            'active_slot': self.time_slot < self.T
        }

        return next_local_states, next_H_matrices, finished_experiences, done, info

    def _record_finished(self, device_m, arrive_time, task_id, delay):
        """Eq. (17): Cost cm = Delaym."""
        self.task_delay[arrive_time, device_m] = delay
        self.task_cost[arrive_time, device_m] = delay
        self.task_dropped[arrive_time, device_m] = False
        self.total_completed_count += 1

    def _record_drop(self, device_m, arrive_time, task_id, reason=''):
        """Eq. (18): Cost cm = C (penalty)."""
        self.task_delay[arrive_time, device_m] = self.tau
        self.task_cost[arrive_time, device_m] = self.C
        self.task_dropped[arrive_time, device_m] = True
        self.task_drop_reason[task_id] = reason
        self.total_dropped_count += 1

        if reason == 'comp_timeout':
            self.drop_local_count += 1
        elif reason == 'tran_timeout':
            self.drop_tran_count += 1
        elif reason == 'edge_timeout':
            self.drop_edge_count += 1
        else:
            raise ValueError(f"Unknown drop reason: {reason}")

    def get_summary_metrics(self):
        """
        Compute audited, strictly consistent metrics for the entire episode.
        Only tasks that arrived during active phase (0..T-1) constitute the population.
        """
        active_arrivals = self.bit_arrive[:self.T, :]
        total_arrived = int(np.count_nonzero(active_arrivals))

        # Active arrivals mask
        arrived_mask = active_arrivals > 0.0

        # Finished and dropped status for active arrivals
        active_delays = self.task_delay[:self.T, :]
        active_costs = self.task_cost[:self.T, :]
        active_dropped = self.task_dropped[:self.T, :]

        completed_mask = arrived_mask & (~active_dropped)
        dropped_mask = arrived_mask & active_dropped

        completed_count = int(np.sum(completed_mask))
        dropped_count = int(np.sum(dropped_mask))

        # Check conservation law: arrived == completed + dropped
        assert total_arrived == completed_count + dropped_count, \
            f"Accounting mismatch: Arrived({total_arrived}) != Completed({completed_count}) + Dropped({dropped_count})"

        # Drop ratio (Eq. in audit requirement)
        drop_ratio = float(dropped_count / max(1, total_arrived))

        # Successful task delay (excluding penalty/dropped tasks)
        if completed_count > 0:
            avg_delay_successful = float(np.mean(active_delays[completed_mask]))
        else:
            avg_delay_successful = float(self.tau)

        # Average cost (including penalty C=20 for dropped tasks)
        avg_cost = float(np.mean(active_costs[arrived_mask])) if total_arrived > 0 else 0.0

        # Delay with penalty included
        avg_delay_with_penalty = float(np.mean(active_delays[arrived_mask])) if total_arrived > 0 else float(self.tau)

        # Category check
        cat_sum = self.drop_local_count + self.drop_tran_count + self.drop_edge_count
        assert dropped_count == cat_sum, \
            f"Category mismatch: Dropped({dropped_count}) != Local({self.drop_local_count}) + Tran({self.drop_tran_count}) + Edge({self.drop_edge_count})"

        return {
            'total_arrived': total_arrived,
            'completed_count': completed_count,
            'dropped_count': dropped_count,
            'drop_ratio': drop_ratio,
            'avg_delay_successful': avg_delay_successful,
            'avg_delay_with_penalty': avg_delay_with_penalty,
            'avg_cost': avg_cost,
            'drop_local': self.drop_local_count,
            'drop_tran': self.drop_tran_count,
            'drop_edge': self.drop_edge_count,
            'diagnostic_logs': self.diagnostic_logs
        }

