"""
D3QN Network (Dueling Double DQN with LSTM) strictly conforming to Tang & Wong (IEEE TMC 2020 / 2022).
Implements the exact neural network architecture from Section 4.1 (Fig. 2 & Fig. 3) and Eq. (20).
"""

import os
import sys

# Ensure local packages (including torch) in .venv_pkgs are found
if sys.version_info >= (3, 8):
    pkg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv_pkgs')
    if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

import torch
import torch.nn as nn


class D3QNPaperNet(nn.Module):
    """
    Neural Network mapping state sm(t) = (lambda, delta_comp, delta_tran, q_edge, H(t))
    to Q-values of actions a in A under Dueling architecture (Eq. 20).
    """
    def __init__(self,
                 num_local_features=8,   # 1 + 1 + 1 + N (for N=5)
                 num_edge_nodes=5,       # N
                 num_actions=6,          # 1 + N
                 n_lstm=20,              # LSTM hidden units
                 n_l1=20):               # FC hidden units
        super(D3QNPaperNet, self).__init__()

        self.num_actions = num_actions
        self.num_local_features = num_local_features
        self.num_edge_nodes = num_edge_nodes

        # 1. LSTM Layer for Historical Load Matrix H(t) of shape [batch, T_step, N] (Section 4.1.2, Fig. 3)
        self.lstm = nn.LSTM(input_size=num_edge_nodes, hidden_size=n_lstm, batch_first=True)

        # 2. FC Layer 1: Takes [LSTM_output, local_features] (Section 4.1.3)
        # Total input dimension = n_lstm + num_local_features = 20 + 8 = 28
        self.fc1 = nn.Linear(n_lstm + num_local_features, n_l1)
        self.relu1 = nn.ReLU()

        # 3. FC Layer 2 (Section 4.1.3)
        self.fc2 = nn.Linear(n_l1, n_l1)
        self.relu2 = nn.ReLU()

        # 4. A&V Layer (Dueling DQN, Section 4.1.4, Eq. 20)
        # Network V: State-value V(sm(t); theta_m) -> scalar [batch, 1]
        self.V_stream = nn.Linear(n_l1, 1)

        # Network A: Action-advantage Am(sm(t), a; theta_m) -> [batch, num_actions]
        self.A_stream = nn.Linear(n_l1, num_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.3)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, local_features, H_matrix):
        """
        Forward pass.
        local_features: [batch_size, num_local_features]
        H_matrix: [batch_size, T_step, num_edge_nodes]
        """
        # LSTM processes sequential load levels
        lstm_out, _ = self.lstm(H_matrix)
        # Extract the last LSTM unit output: [batch_size, n_lstm]
        lstm_reduced = lstm_out[:, -1, :]

        # Concatenate temporal representation with remaining state features
        x = torch.cat([lstm_reduced, local_features], dim=1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))

        # Dueling decomposition
        V = self.V_stream(x)   # [batch_size, 1]
        A = self.A_stream(x)   # [batch_size, num_actions]

        # Equation (20): Qm = Vm + (Am - (1 / |A|) * sum(Am))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

