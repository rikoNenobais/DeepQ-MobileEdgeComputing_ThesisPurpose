"""
D3QN Agent strictly conforming to Tang & Wong (IEEE TMC 2020 / 2022).
Implements the exact algorithms:
- Algorithm 1: Device-side Action Selection with argmin Cost Minimization (Eq. 22)
- Algorithm 2: Edge-side Double Dueling DQN Training with Target Network (Eq. 24, 25, 26)
- Exploration Probability Schedule: epsilon decaying from 1.0 down to 0.01 (Section 5)
- Non-Stationary Adaptation Mechanism: Reset epsilon to 1.0 upon environment shift (Section 5)
"""

import os
import sys
import numpy as np

# Ensure local packages (including torch) in .venv_pkgs are found
if sys.version_info >= (3, 8):
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv_pkgs')
    if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from d3qn_network import D3QNPaperNet


class PaperReplayMemory:
    """
    Replay Memory Dm storing complete experience tuples (sm(t), am(t), cm(t), sm(t+1))
    where state sm(t) includes local_features and complete load matrix H(t).
    """
    def __init__(self, capacity=500, num_local_features=8, t_step=10, num_edge_nodes=5):
        self.capacity = capacity
        self.num_local_features = num_local_features
        self.t_step = t_step
        self.num_edge_nodes = num_edge_nodes

        # Pre-allocate buffer arrays
        self.mem_local_s = np.zeros((capacity, num_local_features), dtype=np.float32)
        self.mem_H_s = np.zeros((capacity, t_step, num_edge_nodes), dtype=np.float32)
        self.mem_action = np.zeros(capacity, dtype=np.int64)
        self.mem_cost = np.zeros(capacity, dtype=np.float32)
        self.mem_local_s_next = np.zeros((capacity, num_local_features), dtype=np.float32)
        self.mem_H_s_next = np.zeros((capacity, t_step, num_edge_nodes), dtype=np.float32)

        self.pointer = 0
        self.size = 0

    def store(self, local_s, H_s, action, cost, local_s_next, H_s_next):
        idx = self.pointer % self.capacity
        self.mem_local_s[idx] = local_s
        self.mem_H_s[idx] = H_s
        self.mem_action[idx] = action
        self.mem_cost[idx] = cost
        self.mem_local_s_next[idx] = local_s_next
        self.mem_H_s_next[idx] = H_s_next

        self.pointer += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.mem_local_s[indices],
            self.mem_H_s[indices],
            self.mem_action[indices],
            self.mem_cost[indices],
            self.mem_local_s_next[indices],
            self.mem_H_s_next[indices]
        )


class D3QNPaperAgent:
    """
    Device & Edge Agent implementing D3QN with:
    - Cost Minimization (argmin)
    - Double DQN (decoupled selection and evaluation)
    - Dueling DQN (A&V streams)
    - LSTM historical load feature extraction
    - Paper hyperparameters: batch_size=16, lr=0.001, gamma=0.9, epsilon=1.0 -> 0.01
    """
    def __init__(self,
                 num_local_features=8,
                 num_edge_nodes=5,
                 num_actions=6,
                 learning_rate=0.001,      # Table 1 & Section 5: lr = 0.001
                 discount_factor=0.9,      # Table 1 & Section 5: gamma = 0.9
                 batch_size=16,            # Table 1 & Section 5: batch_size = 16
                 replace_threshold=200,    # Replace_Threshold for Target Net update
                 memory_capacity=500,
                 epsilon_init=1.0,         # Probability of random exploration: 1.0 -> 0.01
                 epsilon_min=0.01,
                 epsilon_decay_steps=4000,
                 device=None):

        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.replace_threshold = replace_threshold

        # Epsilon exploration probability (Section 5: 1.0 down to 0.01)
        self.epsilon = float(epsilon_init)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay_step = (epsilon_init - epsilon_min) / max(1, epsilon_decay_steps)

        self.train_step_count = 0
        self.device = torch.device('cpu') if device is None else torch.device(device)

        # 1. Evaluation Network (Net_m) & Target Network (Target_Net_m)
        self.eval_net = D3QNPaperNet(
            num_local_features=num_local_features,
            num_edge_nodes=num_edge_nodes,
            num_actions=num_actions,
            n_lstm=20,
            n_l1=20
        ).to(self.device)

        self.target_net = D3QNPaperNet(
            num_local_features=num_local_features,
            num_edge_nodes=num_edge_nodes,
            num_actions=num_actions,
            n_lstm=20,
            n_l1=20
        ).to(self.device)

        # Initialize Target Net with Evaluation Net weights (Alg. 2, Step 3)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        # Optimizer: RMSProp (Section 5, Page 9)
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.latest_loss = 0.0

        # Replay Memory Dm
        self.memory = PaperReplayMemory(
            capacity=memory_capacity,
            num_local_features=num_local_features,
            t_step=10,
            num_edge_nodes=num_edge_nodes
        )

    def select_action(self, local_s, H_s):
        """
        Algorithm 1, Eq. (22):
        am(t) = random action with probability epsilon
                argmin_a Qm(sm(t), a; theta_m) with probability 1 - epsilon
        """
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            # Random exploration
            return int(np.random.randint(0, self.num_actions))

        # Greedy choice: argmin Q-value (Cost Minimization)
        self.eval_net.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(local_s[np.newaxis, :]).float().to(self.device)
            H_t = torch.from_numpy(H_s[np.newaxis, :, :]).float().to(self.device)
            q_values = self.eval_net(s_t, H_t)
            # argmin because we are minimizing cost/delay
            best_action = int(torch.argmin(q_values, dim=1).item())

        return best_action

    def store_experience(self, local_s, H_s, action, cost, local_s_next, H_s_next):
        """Algorithm 1 Step 12 & Algorithm 2 Step 9: Store experience in Dm."""
        self.memory.store(local_s, H_s, action, cost, local_s_next, H_s_next)

    def train_step(self):
        """
        Algorithm 2 (Steps 10-20):
        1. Sample mini-batch of |I| = 16 from Dm
        2. Double DQN target calculation with argmin (Eq. 25 & 26)
        3. Backpropagation minimizing loss (Eq. 24)
        4. Target Net sync every Replace_Threshold steps
        """
        if self.memory.size < self.batch_size:
            return

        # 1. Sample mini-batch
        b_s, b_H, b_a, b_c, b_s_next, b_H_next = self.memory.sample(self.batch_size)

        s_tensor = torch.from_numpy(b_s).float().to(self.device)
        H_tensor = torch.from_numpy(b_H).float().to(self.device)
        a_tensor = torch.from_numpy(b_a).long().to(self.device)
        c_tensor = torch.from_numpy(b_c).float().to(self.device)
        s_next_tensor = torch.from_numpy(b_s_next).float().to(self.device)
        H_next_tensor = torch.from_numpy(b_H_next).float().to(self.device)

        # 2. Evaluate current Q-values: Qm(sm(i), am(i); theta_m)
        self.eval_net.train()
        q_eval = self.eval_net(s_tensor, H_tensor)
        q_eval_action = q_eval.gather(1, a_tensor.unsqueeze(1)).squeeze(1)

        # 3. Double DQN Target Calculation (Eq. 25 & 26):
        with torch.no_grad():
            # Action selection with Evaluation Network: a_next = argmin_a Q(s', a; theta_m)
            q_eval_next = self.eval_net(s_next_tensor, H_next_tensor)
            a_next = torch.argmin(q_eval_next, dim=1, keepdim=True)

            # Action evaluation with Target Network: Q(s', a_next; theta_m^-)
            q_target_next_all = self.target_net(s_next_tensor, H_next_tensor)
            q_target_next_selected = q_target_next_all.gather(1, a_next).squeeze(1)

            # Eq. (26): Q_Target = cost + gamma * Q(s', a_next; theta_m^-)
            q_target = c_tensor + self.gamma * q_target_next_selected

        # 4. Minimize loss (Eq. 24)
        loss = self.loss_fn(q_eval_action, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.latest_loss = float(loss.item())

        # 5. Decay exploration probability epsilon down to epsilon_min
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay_step)

        # 6. Periodic Target Network update (Alg. 2 Step 18-20)
        self.train_step_count += 1
        if self.train_step_count % self.replace_threshold == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def decay_epsilon_by_episode(self, episode, total_episodes):
        """Linearly decay epsilon from 1.0 to epsilon_min over 75% of training episodes."""
        decay_episodes = max(1, int(0.75 * total_episodes))
        if episode < decay_episodes:
            self.epsilon = float(1.0 - (episode / decay_episodes) * (1.0 - self.epsilon_min))
        else:
            self.epsilon = float(self.epsilon_min)

    def reset_exploration(self, new_epsilon=1.0):
        """
        Explicit adaptation mechanism described in Section 5 (Page 9) of Tang & Wong:
        'Under a non-stationary environment, if the environment has changed,
         then the proposed algorithm can adapt to it by resetting the probability
         of random exploration to be one so as to enable the random exploration again.'
        """
        self.epsilon = float(new_epsilon)



# Backward-compatibility alias
D3QNAgent = D3QNPaperAgent
