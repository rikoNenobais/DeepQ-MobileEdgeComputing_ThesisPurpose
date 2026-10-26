"""
MEC Environment Simulator strictly conforming to Tang & Wong (IEEE TMC 2020 / 2022).
Implements the exact mathematical models from Section 2 and Section 3 of the paper:
- Section 2.1: Mobile Device Model (Task Model, Offloading Decision, Computation Queue, Transmission Queue)
- Section 2.2: Edge Node Model (Queues with Generalized Processor Sharing - GPS)
- Section 3: Task Offloading Problem Formulation (State sm(t), Action am(t), Cost cm(t))
"""

import numpy as np
import queue
import math


class MECEnvironment:
    """
    Mobile Edge Computing (MEC) System with M mobile devices and N edge nodes.
    Parameters match Table 1 of Tang & Wong (2020).
    """
    def __init__(self,
                 num_devices=50,       # M
                 num_edge_nodes=5,     # N
                 num_time_slots=100,   # T
                 max_deadline=10,      # tau_m (in time slots, i.e., 1.0 second)
                 time_slot_duration=0.1,  # Delta = 0.1 second
                 f_device=2.5,         # 2.5 GHz
                 f_edge=41.8,          # 41.8 GHz
                 r_tran=14.0,          # 14 Mbps
                 comp_density=0.297,   # 0.297 gigacycles per Mbit
                 task_arrival_prob=0.3, # Task arrival probability
                 t_step=10,            # T_step for LSTM historical load matrix H(t)
                 penalty_cost=20.0):   # Constant penalty C for dropped tasks

        # System dimensions
        self.M = num_devices
        self.N = num_edge_nodes
        self.T = num_time_slots
        self.tau = max_deadline
        self.Delta = time_slot_duration
        self.T_step = t_step
        self.C = penalty_cost

        # Computational & Transmission Capacities (per time slot)
        # 1 Gigacycle = 10^9 cycles. Capacity per time slot = f * Delta
        self.f_device = f_device * self.Delta  # Gigacycles per slot
        self.f_edge = f_edge * self.Delta      # Gigacycles per slot
        self.r_tran = r_tran * self.Delta      # Mbits per slot
        self.rho = comp_density                # Gigacycles per Mbit

        # Discrete task size set Lambda = {2.0, 2.1, ..., 5.0} Mbits (Table 1)
        self.task_size_set = np.round(np.arange(2.0, 5.05, 0.1), 1)
        self.task_arrival_prob = task_arrival_prob

        # Action space: 0 = Local, 1 = Edge 1, ..., N = Edge N (Total 1 + N choices)
        self.num_actions = 1 + self.N

        # State dimensions:
        # local_features: [lambda_m(t), delta_comp_m(t), delta_tran_m(t), q_edge_m_1..N(t-1)] -> 1 + 1 + 1 + N
        self.num_local_features = 3 + self.N
        # historical load matrix H(t): size T_step x N
        self.lstm_seq_len = self.T_step
        self.lstm_feature_dim = self.N

        # Queues and states initialized in reset()
        self.time_slot = 0
        self.reset()

    def reset(self, bit_arrive_matrix=None):
        """
        Reset the environment for a new episode.
        bit_arrive_matrix: optional matrix of shape [T + tau, M] specifying task arrivals.
        """
        self.time_slot = 0
        total_time_slots = self.T + self.tau

        # Task arrivals: if not provided, sample from discrete set Lambda with arrival probability
        if bit_arrive_matrix is not None:
            self.bit_arrive = bit_arrive_matrix.copy()
        else:
            self.bit_arrive = np.zeros((total_time_slots, self.M), dtype=np.float32)
            for t in range(self.T):
                arrival_mask = np.random.uniform(0.0, 1.0, size=self.M) < self.task_arrival_prob
                random_sizes = np.random.choice(self.task_size_set, size=self.M)
                self.bit_arrive[t, :] = random_sizes * arrival_mask
            # Final tau slots have zero arrivals to drain remaining tasks
            self.bit_arrive[self.T:, :] = 0.0

        # FIFO Queue structures
        self.queue_device_comp = [queue.Queue() for _ in range(self.M)]
        self.queue_device_tran = [queue.Queue() for _ in range(self.M)]
        self.queue_edge_comp = [[queue.Queue() for _ in range(self.N)] for _ in range(self.M)]

        # Tracking end-of-queue completion time slots
        self.l_comp_latest = -np.ones(self.M, dtype=int)
        self.l_tran_latest = -np.ones(self.M, dtype=int)

        # Edge queue length vector q_edge_m(t-1) for device m at edge nodes: shape [M, N] (Eq. 8)
        self.q_edge = np.zeros((self.M, self.N), dtype=np.float32)

        # Active tasks currently in processing/transmission
        self.current_proc_local = [{'size': 0.0, 'time': -1, 'remain': np.nan} for _ in range(self.M)]
        self.current_tran_local = [{'size': 0.0, 'time': -1, 'edge': -1, 'remain': np.nan} for _ in range(self.M)]
        self.current_proc_edge = [[{'size': 0.0, 'time': -1, 'remain': np.nan} for _ in range(self.N)] for _ in range(self.M)]

        # Edge node load broadcast history: deque / array of shape [T_step, N]
        self.load_history = np.zeros((self.T_step, self.N), dtype=np.float32)

        # Active queue counts B_n(t) for current time slot
        self.B_n = np.zeros(self.N, dtype=np.float32)

        # Performance trackers: cost, delay, drops
        self.task_cost = np.zeros((total_time_slots, self.M), dtype=np.float32)
        self.task_delay = np.zeros((total_time_slots, self.M), dtype=np.float32)
        self.task_dropped = np.zeros((total_time_slots, self.M), dtype=bool)

        self.drop_local_count = 0
        self.drop_tran_count = 0
        self.drop_edge_count = 0

        # Return initial states
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
        Build state s_m(t) for each device m according to Eq. (13):
        sm(t) = (lambda_m(t), delta_comp_m(t), delta_tran_m(t), q_edge_m(t-1), H(t))
        Returns:
            local_states: [M, num_local_features]
            H_matrix: [M, T_step, N]
        """
        local_states = np.zeros((self.M, self.num_local_features), dtype=np.float32)
        H_matrices = np.zeros((self.M, self.T_step, self.N), dtype=np.float32)

        t = self.time_slot
        for m in range(self.M):
            task_bits = self.bit_arrive[t, m] if t < len(self.bit_arrive) else 0.0

            if task_bits > 0.0:
                delta_comp = self._get_waiting_time_comp(m, t)
                delta_tran = self._get_waiting_time_tran(m, t)
                q_edge_prev = self.q_edge[m, :]

                # [lambda_m(t), delta_comp, delta_tran, q_edge_1, ..., q_edge_N]
                local_states[m, :] = np.hstack([task_bits, delta_comp, delta_tran, q_edge_prev])

            H_matrices[m, :, :] = self.load_history

        return local_states, H_matrices

    def step(self, actions):
        """
        Execute one time slot with actions am(t) from all mobile devices.
        actions: list or 1D array of length M with values in {0, 1, ..., N}.
                 0 = Local execution
                 1..N = Offload to Edge Node (action - 1)
        """
        t = self.time_slot
        finished_experiences = [[] for _ in range(self.M)]

        # ----------------------------------------------------------------------
        # 1. ENQUEUE NEW ARRIVING TASKS (Section 2.1.2)
        # ----------------------------------------------------------------------
        for m in range(self.M):
            task_bits = self.bit_arrive[t, m] if t < len(self.bit_arrive) else 0.0
            act = int(actions[m])

            if task_bits > 0.0:
                task_item = {'size': task_bits, 'time': t, 'edge': -1}

                if act == 0:
                    # x_m(t) = 1: Local Computation Queue
                    self.queue_device_comp[m].put(task_item)

                    # Update l_comp_latest tracking (Eq. 3)
                    delta_comp = self._get_waiting_time_comp(m, t)
                    service_slots = math.ceil(task_bits * self.rho / self.f_device)
                    predicted_finish = t + delta_comp + service_slots - 1
                    deadline_slot = t + self.tau - 1
                    self.l_comp_latest[m] = min(predicted_finish, deadline_slot)

                else:
                    # x_m(t) = 0: Transmission Queue for offloading to Edge (act - 1)
                    edge_idx = act - 1
                    task_item['edge'] = edge_idx
                    self.queue_device_tran[m].put(task_item)

                    # Update l_tran_latest tracking (Eq. 6)
                    delta_tran = self._get_waiting_time_tran(m, t)
                    tran_slots = math.ceil(task_bits / self.r_tran)
                    predicted_finish = t + delta_tran + tran_slots - 1
                    deadline_slot = t + self.tau - 1
                    self.l_tran_latest[m] = min(predicted_finish, deadline_slot)

        # ----------------------------------------------------------------------
        # 2. LOCAL COMPUTATION QUEUE PROCESSING (Section 2.1.3)
        # ----------------------------------------------------------------------
        for m in range(self.M):
            # Fetch next task if processor is idle
            if math.isnan(self.current_proc_local[m]['remain']) and not self.queue_device_comp[m].empty():
                while not self.queue_device_comp[m].empty():
                    task = self.queue_device_comp[m].get()
                    if t - task['time'] + 1 <= self.tau:
                        self.current_proc_local[m] = {'size': task['size'], 'time': task['time'], 'remain': task['size']}
                        break
                    else:
                        # Task expired before starting
                        self._record_drop(m, task['time'], reason='comp_timeout')
                        finished_experiences[m].append(task['time'])

            # Execute computation
            if self.current_proc_local[m]['remain'] > 0:
                self.current_proc_local[m]['remain'] -= (self.f_device / self.rho)

                # Completed
                if self.current_proc_local[m]['remain'] <= 0:
                    delay = t - self.current_proc_local[m]['time'] + 1
                    self._record_finished(m, self.current_proc_local[m]['time'], delay)
                    finished_experiences[m].append(self.current_proc_local[m]['time'])
                    self.current_proc_local[m]['remain'] = np.nan
                # Deadline expired during processing
                elif t - self.current_proc_local[m]['time'] + 1 >= self.tau:
                    self._record_drop(m, self.current_proc_local[m]['time'], reason='comp_timeout')
                    finished_experiences[m].append(self.current_proc_local[m]['time'])
                    self.current_proc_local[m]['remain'] = np.nan

        # ----------------------------------------------------------------------
        # 3. TRANSMISSION QUEUE PROCESSING (Section 2.1.4)
        # ----------------------------------------------------------------------
        for m in range(self.M):
            if math.isnan(self.current_tran_local[m]['remain']) and not self.queue_device_tran[m].empty():
                while not self.queue_device_tran[m].empty():
                    task = self.queue_device_tran[m].get()
                    if t - task['time'] + 1 <= self.tau:
                        self.current_tran_local[m] = {
                            'size': task['size'], 'time': task['time'], 'edge': task['edge'], 'remain': task['size']
                        }
                        break
                    else:
                        self._record_drop(m, task['time'], reason='tran_timeout')
                        finished_experiences[m].append(task['time'])

            if self.current_tran_local[m]['remain'] > 0:
                self.current_tran_local[m]['remain'] -= self.r_tran
                edge_idx = self.current_tran_local[m]['edge']

                # Transmission completed: arrive at Edge Node queue at beginning of next slot
                if self.current_tran_local[m]['remain'] <= 0:
                    self.queue_edge_comp[m][edge_idx].put({
                        'size': self.current_tran_local[m]['size'],
                        'time': self.current_tran_local[m]['time']
                    })
                    self.q_edge[m, edge_idx] += self.current_tran_local[m]['size']
                    self.current_tran_local[m]['remain'] = np.nan
                elif t - self.current_tran_local[m]['time'] + 1 >= self.tau:
                    self._record_drop(m, self.current_tran_local[m]['time'], reason='tran_timeout')
                    finished_experiences[m].append(self.current_tran_local[m]['time'])
                    self.current_tran_local[m]['remain'] = np.nan

        # ----------------------------------------------------------------------
        # 4. EDGE NODE PROCESSING WITH GPS EQUAL SHARING (Section 2.2.1)
        # ----------------------------------------------------------------------
        # Determine active queues Bn(t) at edge node n (Eq. 7)
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
                # Fetch next edge task if server queue is ready
                if math.isnan(self.current_proc_edge[m][n]['remain']) and not self.queue_edge_comp[m][n].empty():
                    while not self.queue_edge_comp[m][n].empty():
                        task = self.queue_edge_comp[m][n].get()
                        if t - task['time'] + 1 <= self.tau:
                            self.current_proc_edge[m][n] = {
                                'size': task['size'], 'time': task['time'], 'remain': task['size']
                            }
                            break
                        else:
                            self._record_drop(m, task['time'], reason='edge_timeout')
                            finished_experiences[m].append(task['time'])

                # Process edge task with GPS sharing
                if self.current_proc_edge[m][n]['remain'] > 0:
                    self.current_proc_edge[m][n]['remain'] -= allocated_capacity
                    # Update local edge queue size tracking (Eq. 8)
                    self.q_edge[m, n] = max(0.0, self.q_edge[m, n] - allocated_capacity)

                    if self.current_proc_edge[m][n]['remain'] <= 0:
                        delay = t - self.current_proc_edge[m][n]['time'] + 1
                        self._record_finished(m, self.current_proc_edge[m][n]['time'], delay)
                        finished_experiences[m].append(self.current_proc_edge[m][n]['time'])
                        self.current_proc_edge[m][n]['remain'] = np.nan
                    elif t - self.current_proc_edge[m][n]['time'] + 1 >= self.tau:
                        self._record_drop(m, self.current_proc_edge[m][n]['time'], reason='edge_timeout')
                        finished_experiences[m].append(self.current_proc_edge[m][n]['time'])
                        self.current_proc_edge[m][n]['remain'] = np.nan

        # ----------------------------------------------------------------------
        # 5. BROADCAST ACTIVE QUEUES & UPDATE LOAD HISTORY H(t) (Eq. 12)
        # ----------------------------------------------------------------------
        # Roll historical matrix and insert current active queue broadcast B_n(t)
        self.load_history = np.vstack([self.load_history[1:], self.B_n])

        # Step time
        self.time_slot += 1
        done = (self.time_slot >= self.T)

        # Terminal cleanup for in-flight tasks if episode finished
        if done:
            for time_idx in range(self.time_slot):
                for m in range(self.M):
                    if self.bit_arrive[time_idx, m] > 0 and self.task_delay[time_idx, m] == 0:
                        self._record_drop(m, time_idx, reason='terminal_drop')
                        finished_experiences[m].append(time_idx)

        next_local_states, next_H_matrices = self._get_all_states()
        return next_local_states, next_H_matrices, finished_experiences, done

    def _record_finished(self, device_m, arrive_time, delay):
        """Eq. (17): Cost cm = Delaym."""
        self.task_delay[arrive_time, device_m] = delay
        self.task_cost[arrive_time, device_m] = delay
        self.task_dropped[arrive_time, device_m] = False

    def _record_drop(self, device_m, arrive_time, reason=''):
        """Eq. (18): Cost cm = C (penalty)."""
        self.task_delay[arrive_time, device_m] = self.tau
        self.task_cost[arrive_time, device_m] = self.C
        self.task_dropped[arrive_time, device_m] = True
        if reason == 'comp_timeout':
            self.drop_local_count += 1
        elif reason == 'tran_timeout':
            self.drop_tran_count += 1
        elif reason == 'edge_timeout':
            self.drop_edge_count += 1

