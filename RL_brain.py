"""
Deep Q-Network with Double Q-learning, Dueling Architecture, and LSTM.
Supports DUAL BACKENDS:
1. PyTorch (Default for modern Python 3.12 / host execution)
2. TensorFlow 1.x (For legacy Python 3.7 / Docker dqn-legacy execution)
"""

import os
import sys
from collections import deque
import numpy as np

# Only add .venv_pkgs on modern Python (>=3.8) to prevent SyntaxError in Python 3.7
if sys.version_info >= (3, 8):
    pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.venv_pkgs')
    if os.path.exists(pkg_dir) and pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

# Check available backends
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HAS_TF = False
try:
    import tensorflow as tf
    if hasattr(tf, 'contrib') and hasattr(tf, 'Session'):
        HAS_TF = True
except Exception:
    HAS_TF = False


# ==============================================================================
# 1. PyTorch Implementation (Modern, High Performance)
# ==============================================================================
if HAS_TORCH:
    class DuelingDQNNet(nn.Module):
        def __init__(self, n_features, n_lstm_state, n_actions, n_lstm=20, n_l1=20, dueling=True):
            super(DuelingDQNNet, self).__init__()
            self.dueling = dueling
            self.n_actions = n_actions

            self.lstm = nn.LSTM(input_size=n_lstm_state, hidden_size=n_lstm, batch_first=True)
            self.fc1 = nn.Linear(n_lstm + n_features, n_l1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(n_l1, n_l1)
            self.relu2 = nn.ReLU()

            if self.dueling:
                self.val = nn.Linear(n_l1, 1)
                self.adv = nn.Linear(n_l1, n_actions)
            else:
                self.q = nn.Linear(n_l1, n_actions)

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.3)
                    nn.init.constant_(m.bias, 0.1)

        def forward(self, s, lstm_s):
            lstm_out, _ = self.lstm(lstm_s)
            lstm_reduced = lstm_out[:, -1, :]

            x = torch.cat([lstm_reduced, s], dim=1)
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))

            if self.dueling:
                val = self.val(x)
                adv = self.adv(x)
                q = val + (adv - adv.mean(dim=1, keepdim=True))
            else:
                q = self.q(x)
            return q

    class DeepQNetworkTorch:
        def __init__(self,
                     n_actions,
                     n_features,
                     n_lstm_features,
                     n_time,
                     learning_rate=0.01,
                     reward_decay=0.9,
                     e_greedy=0.99,
                     replace_target_iter=200,
                     memory_size=500,
                     batch_size=32,
                     e_greedy_increment=0.00025,
                     n_lstm_step=10,
                     dueling=True,
                     double_q=True,
                     N_L1=20,
                     N_lstm=20,
                     device=None):

            self.n_actions = n_actions
            self.n_features = n_features
            self.n_time = n_time
            self.lr = learning_rate
            self.gamma = reward_decay
            self.epsilon_max = e_greedy
            self.replace_target_iter = replace_target_iter
            self.memory_size = memory_size
            self.batch_size = batch_size
            self.epsilon_increment = e_greedy_increment
            self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
            self.dueling = dueling
            self.double_q = double_q
            self.learn_step_counter = 0
            self.N_L1 = N_L1

            self.N_lstm = N_lstm
            self.n_lstm_step = n_lstm_step
            self.n_lstm_state = n_lstm_features
            self.device = torch.device('cpu') if device is None else torch.device(device)

            total_dim = self.n_features + 2 + self.n_features + 2 * self.n_lstm_state
            self.memory = np.zeros((self.memory_size, total_dim), dtype=np.float32)
            self.memory_counter = 0

            self.eval_net = DuelingDQNNet(
                n_features=self.n_features,
                n_lstm_state=self.n_lstm_state,
                n_actions=self.n_actions,
                n_lstm=self.N_lstm,
                n_l1=self.N_L1,
                dueling=self.dueling
            ).to(self.device)

            self.target_net = DuelingDQNNet(
                n_features=self.n_features,
                n_lstm_state=self.n_lstm_state,
                n_actions=self.n_actions,
                n_lstm=self.N_lstm,
                n_l1=self.N_L1,
                dueling=self.dueling
            ).to(self.device)

            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
            self.loss_fn = nn.MSELoss()
            self.cost = 0.0

            self.reward_store = list()
            self.action_store = list()
            self.delay_store = list()

            self.lstm_history = deque(maxlen=self.n_lstm_step)
            for _ in range(self.n_lstm_step):
                self.lstm_history.append(np.zeros(self.n_lstm_state, dtype=np.float32))

            self.store_q_value = list()

        def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
            transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_)).astype(np.float32)
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

        def update_lstm(self, lstm_s):
            self.lstm_history.append(np.array(lstm_s, dtype=np.float32))

        def choose_action(self, observation):
            observation = np.asarray(observation, dtype=np.float32)

            if np.random.uniform() < self.epsilon:
                lstm_observation = np.array(self.lstm_history, dtype=np.float32)
                lstm_input = lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)

                s_tensor = torch.from_numpy(observation[np.newaxis, :]).to(self.device)
                lstm_tensor = torch.from_numpy(lstm_input).to(self.device)

                self.eval_net.eval()
                with torch.no_grad():
                    q_vals = self.eval_net(s_tensor, lstm_tensor)
                    actions_value = q_vals.cpu().numpy()

                self.store_q_value.append({'observation': observation[np.newaxis, :], 'q_value': actions_value})
                action = int(np.argmax(actions_value))
            else:
                action = int(np.random.randint(0, self.n_actions))

            return action

        def learn(self):
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())

            max_idx = min(self.memory_counter, self.memory_size)
            if max_idx <= self.n_lstm_step:
                return

            sample_index = np.random.choice(max_idx - self.n_lstm_step, size=self.batch_size)

            n_feat = self.n_features
            batch_memory = self.memory[sample_index, :n_feat + 2 + n_feat]

            s = torch.from_numpy(batch_memory[:, :n_feat]).to(self.device)
            eval_act_index = torch.from_numpy(batch_memory[:, n_feat].astype(np.int64)).to(self.device)
            reward = torch.from_numpy(batch_memory[:, n_feat + 1]).to(self.device)
            s_ = torch.from_numpy(batch_memory[:, -n_feat:]).to(self.device)

            lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2], dtype=np.float32)
            lstm_start = n_feat + 2 + n_feat
            for ii in range(len(sample_index)):
                for jj in range(self.n_lstm_step):
                    lstm_batch_memory[ii, jj, :] = self.memory[sample_index[ii] + jj, lstm_start:]

            lstm_s = torch.from_numpy(lstm_batch_memory[:, :, :self.n_lstm_state]).to(self.device)
            lstm_s_ = torch.from_numpy(lstm_batch_memory[:, :, self.n_lstm_state:]).to(self.device)

            self.eval_net.train()
            q_eval = self.eval_net(s, lstm_s)
            q_eval_selected = q_eval.gather(1, eval_act_index.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.double_q:
                    q_eval4next = self.eval_net(s_, lstm_s_)
                    max_act4next = torch.argmax(q_eval4next, dim=1, keepdim=True)
                    q_next_all = self.target_net(s_, lstm_s_)
                    selected_q_next = q_next_all.gather(1, max_act4next).squeeze(1)
                else:
                    q_next_all = self.target_net(s_, lstm_s_)
                    selected_q_next = torch.max(q_next_all, dim=1)[0]

                q_target = reward + self.gamma * selected_q_next

            loss = self.loss_fn(q_eval_selected, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.cost = float(loss.item())

            if self.epsilon < self.epsilon_max:
                self.epsilon += self.epsilon_increment
            else:
                self.epsilon = self.epsilon_max

            self.learn_step_counter += 1

        def do_store_reward(self, episode, time, reward):
            while episode >= len(self.reward_store):
                self.reward_store.append(np.zeros([self.n_time], dtype=np.float32))
            self.reward_store[episode][time] = reward

        def do_store_action(self, episode, time, action):
            while episode >= len(self.action_store):
                self.action_store.append(-np.ones([self.n_time], dtype=np.float32))
            self.action_store[episode][time] = action

        def do_store_delay(self, episode, time, delay):
            while episode >= len(self.delay_store):
                self.delay_store.append(np.zeros([self.n_time], dtype=np.float32))
            self.delay_store[episode][time] = delay


# ==============================================================================
# 2. TensorFlow 1.x Implementation (Legacy Docker dqn-legacy environment)
# ==============================================================================
if HAS_TF:
    class DeepQNetworkTF:
        def __init__(self,
                     n_actions,
                     n_features,
                     n_lstm_features,
                     n_time,
                     learning_rate=0.01,
                     reward_decay=0.9,
                     e_greedy=0.99,
                     replace_target_iter=200,
                     memory_size=500,
                     batch_size=32,
                     e_greedy_increment=0.00025,
                     n_lstm_step=10,
                     dueling=True,
                     double_q=True,
                     N_L1=20,
                     N_lstm=20):

            self.n_actions = n_actions
            self.n_features = n_features
            self.n_time = n_time
            self.lr = learning_rate
            self.gamma = reward_decay
            self.epsilon_max = e_greedy
            self.replace_target_iter = replace_target_iter
            self.memory_size = memory_size
            self.batch_size = batch_size
            self.epsilon_increment = e_greedy_increment
            self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
            self.dueling = dueling
            self.double_q = double_q
            self.learn_step_counter = 0
            self.N_L1 = N_L1

            self.N_lstm = N_lstm
            self.n_lstm_step = n_lstm_step
            self.n_lstm_state = n_lstm_features

            self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                    + self.n_features + self.n_lstm_state + self.n_lstm_state))
            self.memory_counter = 0

            self._build_net()

            t_params = tf.get_collection(f'target_net_params_{id(self)}')
            e_params = tf.get_collection(f'eval_net_params_{id(self)}')
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.reward_store = list()
            self.action_store = list()
            self.delay_store = list()

            self.lstm_history = deque(maxlen=self.n_lstm_step)
            for _ in range(self.n_lstm_step):
                self.lstm_history.append(np.zeros([self.n_lstm_state]))

            self.store_q_value = list()

        def _build_net(self):
            uid = id(self)

            def build_layers(s, lstm_s, c_names, n_l1, n_lstm, w_init, b_init):
                with tf.variable_scope(f'l0_{uid}'):
                    lstm_dnn = tf.contrib.rnn.BasicLSTMCell(n_lstm)
                    lstm_output, _ = tf.nn.dynamic_rnn(lstm_dnn, lstm_s, dtype=tf.float32)
                    lstm_reduced = tf.reshape(lstm_output[:, -1, :], shape=[-1, n_lstm])

                with tf.variable_scope(f'l1_{uid}'):
                    w1 = tf.get_variable('w1', [n_lstm + self.n_features, n_l1], initializer=w_init, collections=c_names)
                    b1 = tf.get_variable('b1', [1, n_l1], initializer=b_init, collections=c_names)
                    l1 = tf.nn.relu(tf.matmul(tf.concat([lstm_reduced, s], 1), w1) + b1)

                with tf.variable_scope(f'l12_{uid}'):
                    w12 = tf.get_variable('w12', [n_l1, n_l1], initializer=w_init, collections=c_names)
                    b12 = tf.get_variable('b12', [1, n_l1], initializer=b_init, collections=c_names)
                    l12 = tf.nn.relu(tf.matmul(l1, w12) + b12)

                if self.dueling:
                    with tf.variable_scope(f'Value_{uid}'):
                        w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_init, collections=c_names)
                        b2 = tf.get_variable('b2', [1, 1], initializer=b_init, collections=c_names)
                        V = tf.matmul(l12, w2) + b2
                    with tf.variable_scope(f'Advantage_{uid}'):
                        w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_init, collections=c_names)
                        b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init, collections=c_names)
                        A = tf.matmul(l12, w2) + b2
                    with tf.variable_scope(f'Q_{uid}'):
                        out = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))
                else:
                    with tf.variable_scope(f'Q_{uid}'):
                        w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_init, collections=c_names)
                        b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_init, collections=c_names)
                        out = tf.matmul(l1, w2) + b2
                return out

            self.s = tf.placeholder(tf.float32, [None, self.n_features], name=f's_{uid}')
            self.lstm_s = tf.placeholder(tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name=f'lstm1_s_{uid}')
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name=f'Q_target_{uid}')
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name=f's_next_{uid}')
            self.lstm_s_ = tf.placeholder(tf.float32, [None, self.n_lstm_step, self.n_lstm_state], name=f'lstm1_s_next_{uid}')

            c_eval = [f'eval_net_params_{uid}', tf.GraphKeys.GLOBAL_VARIABLES]
            w_init = tf.random_normal_initializer(0., 0.3)
            b_init = tf.constant_initializer(0.1)

            with tf.variable_scope(f'eval_net_{uid}'):
                self.q_eval = build_layers(self.s, self.lstm_s, c_eval, self.N_L1, self.N_lstm, w_init, b_init)

            c_target = [f'target_net_params_{uid}', tf.GraphKeys.GLOBAL_VARIABLES]
            with tf.variable_scope(f'target_net_{uid}'):
                self.q_next = build_layers(self.s_, self.lstm_s_, c_target, self.N_L1, self.N_lstm, w_init, b_init)

            with tf.variable_scope(f'loss_{uid}'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope(f'train_{uid}'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
            transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

        def update_lstm(self, lstm_s):
            self.lstm_history.append(lstm_s)

        def choose_action(self, observation):
            observation = observation[np.newaxis, :]
            if np.random.uniform() < self.epsilon:
                lstm_observation = np.array(self.lstm_history)
                actions_value = self.sess.run(self.q_eval, feed_dict={
                    self.s: observation,
                    self.lstm_s: lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)
                })
                self.store_q_value.append({'observation': observation, 'q_value': actions_value})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
            return action

        def learn(self):
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.replace_target_op)

            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

            batch_memory = self.memory[sample_index, :self.n_features + 2 + self.n_features]
            lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
            for ii in range(len(sample_index)):
                for jj in range(self.n_lstm_step):
                    lstm_batch_memory[ii, jj, :] = self.memory[sample_index[ii] + jj, self.n_features + 2 + self.n_features:]

            q_next, q_eval4next = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={
                    self.s_: batch_memory[:, -self.n_features:],
                    self.lstm_s_: lstm_batch_memory[:, :, self.n_lstm_state:],
                    self.s: batch_memory[:, -self.n_features:],
                    self.lstm_s: lstm_batch_memory[:, :, self.n_lstm_state:]
                }
            )
            q_eval = self.sess.run(self.q_eval, {
                self.s: batch_memory[:, :self.n_features],
                self.lstm_s: lstm_batch_memory[:, :, :self.n_lstm_state]
            })
            q_target = q_eval.copy()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            if self.double_q:
                max_act4next = np.argmax(q_eval4next, axis=1)
                selected_q_next = q_next[batch_index, max_act4next]
            else:
                selected_q_next = np.max(q_next, axis=1)

            q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

            _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.lstm_s: lstm_batch_memory[:, :, :self.n_lstm_state],
                self.q_target: q_target
            })

            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            self.learn_step_counter += 1

        def do_store_reward(self, episode, time, reward):
            while episode >= len(self.reward_store):
                self.reward_store.append(np.zeros([self.n_time]))
            self.reward_store[episode][time] = reward

        def do_store_action(self, episode, time, action):
            while episode >= len(self.action_store):
                self.action_store.append(-np.ones([self.n_time]))
            self.action_store[episode][time] = action

        def do_store_delay(self, episode, time, delay):
            while episode >= len(self.delay_store):
                self.delay_store.append(np.zeros([self.n_time]))
            self.delay_store[episode][time] = delay


# ==============================================================================
# 3. Factory / Wrapper Class: DeepQNetwork
# ==============================================================================
def DeepQNetwork(*args, backend='auto', **kwargs):
    """
    Instantiate DeepQNetwork automatically picking the best available backend:
    - 'torch': PyTorch implementation (default if available)
    - 'tf': TensorFlow 1.x implementation (for legacy docker)
    - 'auto': Automatically select PyTorch if installed, otherwise TensorFlow.
    """
    if backend == 'torch' or (backend == 'auto' and HAS_TORCH):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available. Install torch or use backend='tf'")
        return DeepQNetworkTorch(*args, **kwargs)
    elif backend == 'tf' or (backend == 'auto' and HAS_TF):
        if not HAS_TF:
            raise RuntimeError("TensorFlow 1.x is not available. Use backend='torch'")
        return DeepQNetworkTF(*args, **kwargs)
    else:
        raise RuntimeError("No suitable Deep Learning backend found. Please install PyTorch or TensorFlow 1.x")
