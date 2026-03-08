# -*- coding: UTF-8 -*-
# @Author:zhuxiao ASUS
import pandas as pd
import numpy as np
import random
from geopy.distance import geodesic
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tracemalloc
np.random.seed(52)
random.seed(52)
torch.manual_seed(52)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DQN(nn.Module):
    """基于PyTorch的DQN神经网络模型"""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PlatformDQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(PlatformDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals


class WorkerEnvironment:
    def __init__(self, data_path):
        self.task_location = (31.23, 121.48)
        self.electricity_price = 0.5
        self.power_consumption = 0.2
        self.c_n = 0.1
        self.c_b = 15
        self.b_j = self.c_b
        self.lambda_param = 50

        self.data = pd.read_csv(data_path)
        self.data = self._preprocess_data(self.data)

        self.state_size = 5
        self.action_size = 2
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001

        self.model = DQN(self.state_size, self.action_size).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.total_episodes = 0
        self.total_accepts = 0
        self.total_rejects = 0

    def _preprocess_data(self, data):
        data = data.drop_duplicates(subset=['id', 'latitude', 'longitude'])
        unique_ids = data['id'].unique()
        if len(unique_ids) > 1000:
            selected_ids = np.random.choice(unique_ids, 1000, replace=False)
            data = data[data['id'].isin(selected_ids)]

        data = data[data['quantity'] != 0]
        data = data[data['speed'] > 0]

        data['q_i'] = 0.8 * data['quantity'] / 100 + 0.2 * data['volume'] / 100
        # data['q_i'] = data['quantity'] / 100
        # data['q_i'] = data['volume'] / 100

        data['distance_to_task'] = data.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), self.task_location).kilometers, axis=1)

        return data[['id', 'latitude', 'longitude', 'q_i', 'speed', 'distance_to_task']]

    def get_state(self, worker):
        return torch.FloatTensor([
            worker['latitude'],
            worker['longitude'],
            worker['q_i'],
            worker['speed'],
            worker['distance_to_task']
        ]).to(device)

    def get_worker_basics(self, worker):
        t_i = worker['distance_to_task'] / worker['speed']
        c_d = worker['distance_to_task'] * self.power_consumption * self.electricity_price
        return t_i, c_d

    def step(self, worker, action):
        if action == 0:
            self.total_rejects += 1
            reward = 0.0
            new_state = self.get_state(worker)
            done = True
            return new_state, reward, done, {
                'platform_profit': 0.0,
                'action': 'reject',
                'worker_accepted': False,
                'reward_share': 0.0,
                'sum_t_k': 0.0,
                't_i': 0.0,
                'c_d': 0.0,
                'q_i': worker['q_i']
            }

        self.total_accepts += 1
        t_i, c_d = self.get_worker_basics(worker)

        temp_reward = -c_d - self.c_n

        new_worker = worker.copy()
        new_worker['latitude'] = self.task_location[0]
        new_worker['longitude'] = self.task_location[1]
        new_worker['distance_to_task'] = 0
        new_state = self.get_state(new_worker)

        done = True

        return new_state, temp_reward, done, {
            'platform_profit': 0.0,
            'action': 'accept',
            'worker_accepted': True,
            'reward_share': 0.0,
            'sum_t_k': t_i,
            't_i': t_i,
            'c_d': c_d,
            'q_i': worker['q_i'],
            'worker_id': worker['id']
        }

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = state.unsqueeze(0)
            act_values = self.model(state)
        return act_values.max(1)[1].item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states)
        dones = torch.BoolTensor(dones).to(device)

        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PlatformEnvironment:

    def __init__(self, worker_env, n_workers_pool=100, n_workers_needed=5):
        self.worker_env = worker_env
        self.n_workers_pool = n_workers_pool
        self.n_workers_needed = n_workers_needed
        self.worker_pool = None
        self.accepted_workers = []
        self.sum_t_k = 0.0
        self.total_platform_profit = 0.0
        self.selected_workers_data = []
        self.total_payment_to_workers = 0.0
        self.worker_experiences = []

        self.state_size = n_workers_pool * 5
        self.action_size = n_workers_pool
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

        self.model = DuelingDQN(self.state_size, self.action_size).to(device)
        self.target_model = DuelingDQN(self.state_size, self.action_size).to(device)

        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.total_episodes = 0
        self.total_profit = 0
        self.avg_profit = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def reset(self):
        self.accepted_workers = []
        self.sum_t_k = 0.0
        self.total_platform_profit = 0.0
        self.selected_workers_data = []
        self.total_payment_to_workers = 0.0
        self.worker_experiences = []

        data_sorted = self.worker_env.data.sort_values('q_i', ascending=False)
        unique_workers = data_sorted.drop_duplicates(subset='id')
        self.worker_pool = unique_workers.head(self.n_workers_pool)

        platform_state = []
        for _, worker in self.worker_pool.iterrows():
            platform_state.extend([
                worker['latitude'],
                worker['longitude'],
                worker['q_i'],
                worker['speed'],
                worker['distance_to_task']
            ])

        return torch.FloatTensor(platform_state).to(device)

    def step(self, action):
        if action < 0 or action >= len(self.worker_pool):
            reward = -1.0
            next_state = self.reset()
            done = True
            info = {
                'platform_profit': 0,
                'worker_profit': 0,
                'worker_selected': None,
                'worker_accepted': False,
                'reward_share': 0.0,
                'sum_t_k': self.sum_t_k,
                'n_accepted': len(self.accepted_workers),
                'total_payment': self.total_payment_to_workers
            }
            return next_state, reward, done, info

        selected_worker = self.worker_pool.iloc[action]

        worker_state = self.worker_env.get_state(selected_worker)
        worker_action = self.worker_env.act(worker_state)

        next_worker_state, worker_temp_reward, worker_done, worker_info = self.worker_env.step(
            selected_worker, worker_action
        )

        self.worker_experiences.append((
            worker_state, worker_action, worker_temp_reward, next_worker_state, worker_done
        ))

        if worker_action == 1:
            worker_data = worker_info.copy()
            worker_data['worker'] = selected_worker

            self.accepted_workers.append(worker_data)
            self.sum_t_k += worker_info['t_i']

            self.selected_workers_data.append(worker_data)

            self.worker_pool = self.worker_pool.drop(self.worker_pool.index[action])

            platform_state = []
            for _, worker in self.worker_pool.iterrows():
                platform_state.extend([
                    worker['latitude'],
                    worker['longitude'],
                    worker['q_i'],
                    worker['speed'],
                    worker['distance_to_task']
                ])
            padding = [0] * (self.state_size - len(platform_state))
            platform_state.extend(padding)
            next_state = torch.FloatTensor(platform_state).to(device)

            done = len(self.accepted_workers) >= self.n_workers_needed or len(self.worker_pool) == 0

            info = {
                'platform_profit': self.total_platform_profit,
                'worker_profit': worker_temp_reward,
                'worker_selected': selected_worker['id'],
                'worker_accepted': True,
                'reward_share': worker_info['reward_share'],
                'total_payment': self.total_payment_to_workers,
                'sum_t_k': self.sum_t_k,
                'n_accepted': len(self.accepted_workers)
            }

            if done:
                self._calculate_final_rewards()
                info['platform_profit'] = self.total_platform_profit
                info['total_payment'] = self.total_payment_to_workers
                self._update_worker_experiences()

            reward = 0.0
            return next_state, reward, done, info

        else:
            reward = -0.1
            self.worker_pool = self.worker_pool.drop(self.worker_pool.index[action])

            if len(self.worker_pool) > 0:
                platform_state = []
                for _, worker in self.worker_pool.iterrows():
                    platform_state.extend([
                        worker['latitude'],
                        worker['longitude'],
                        worker['q_i'],
                        worker['speed'],
                        worker['distance_to_task']
                    ])
                padding = [0] * (self.state_size - len(platform_state))
                platform_state.extend(padding)
                next_state = torch.FloatTensor(platform_state).to(device)
                done = False
            else:
                next_state = torch.zeros(self.state_size, device=device)
                done = True
                if self.accepted_workers:
                    self._calculate_final_rewards()
                    self._update_worker_experiences()

            info = {
                'platform_profit': self.total_platform_profit,
                'worker_profit': 0,
                'worker_selected': None,
                'worker_accepted': False,
                'reward_share': 0.0,
                'sum_t_k': self.sum_t_k,
                'n_accepted': len(self.accepted_workers),
                'total_payment': self.total_payment_to_workers
            }

            return next_state, reward, done, info

    def _calculate_final_rewards(self):
        if not self.selected_workers_data:
            return

        sum_t_k = self.sum_t_k

        total_reward = 0.0
        for worker_data in self.selected_workers_data:
            # print(f"t_i: {worker_data['t_i']}, distance: {worker_data['worker']['distance_to_task']}, speed: {worker_data['worker']['speed']}")  # 检查t_i、距离、速度
            # print(f"q_i: {worker_data['q_i']}")
            if sum_t_k == 0:
                reward_share = 0.0
            else:
                reward_share = (
                    worker_data['q_i'].clip(max=0.8) *
                    self.worker_env.b_j *
                    (worker_data['t_i'] / sum_t_k)
                )
            worker_data['reward_share'] = reward_share
            total_reward += reward_share
            worker_data['U_i'] = reward_share - worker_data['c_d'] - self.worker_env.c_n

        terms = [1 + w['t_i'] * w['q_i'] for w in self.selected_workers_data]
        log_terms = np.log(terms)
        sum_log_terms = np.sum(log_terms)
        self.total_platform_profit = self.worker_env.lambda_param * np.log(1 + sum_log_terms) - total_reward
        self.total_payment_to_workers = total_reward

    def _update_worker_experiences(self):
        for i, exp in enumerate(self.worker_experiences):
            worker_state, worker_action, _, next_worker_state, worker_done = exp
            if i < len(self.selected_workers_data) and self.selected_workers_data[i]['worker_accepted']:
                final_reward = self.selected_workers_data[i]['U_i']
            else:
                final_reward = 0.0
            self.worker_env.remember(worker_state, worker_action, final_reward, next_worker_state, worker_done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, next_state, reward, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            act_values = self.model(state.unsqueeze(0))
        return act_values.max(1)[1].item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes=1000, batch_size=32):
    tracemalloc.start()

    worker_env = WorkerEnvironment('sh_csv/merged_shanghai_taxi_gps.csv')
    platform_env = PlatformEnvironment(worker_env, n_workers_pool=100, n_workers_needed=5)

    all_episode_rewards = []
    all_platform_total_profits = []
    all_platform_profits = []
    all_worker_profits = []
    all_accept_rates = []
    all_sum_t_k = []
    all_platform_payments = []

    for e in range(episodes):
        platform_state = platform_env.reset()
        total_platform_reward = 0
        total_platform_profit = 0
        accepts = 0
        total_workers = 0
        current_info = None

        while True:
            platform_action = platform_env.act(platform_state)

            next_platform_state, platform_reward, done, info = platform_env.step(platform_action)
            current_info = info

            platform_env.remember(
                platform_state,
                torch.tensor([platform_action], device=device),
                torch.tensor([platform_reward], device=device),
                next_platform_state,
                done
            )

            platform_state = next_platform_state
            total_platform_reward += platform_reward

            worker_accepted = info.get('worker_accepted', False)
            if worker_accepted:
                accepts += 1
                total_platform_profit += info['platform_profit']

            total_workers += 1

            if done:
                total_platform_reward = platform_env.total_platform_profit
                all_sum_t_k.append(info['sum_t_k'])
                all_platform_payments.append(info['total_payment'])
                all_platform_total_profits.append(platform_env.total_platform_profit)
                if platform_env.selected_workers_data:
                    total_worker_profit = sum(w['U_i'] for w in platform_env.selected_workers_data)
                    avg_worker_profit = total_worker_profit / len(platform_env.selected_workers_data)
                else:
                    avg_worker_profit = 0.0
                all_worker_profits.append(avg_worker_profit)
                break

        platform_env.replay()

        worker_env.replay(batch_size)

        if e % 100 == 0:
            platform_env.update_target_model()

        platform_env.total_episodes += 1
        worker_env.total_episodes += 1

        accept_rate = accepts / total_workers if total_workers > 0 else 0
        avg_platform_profit = total_platform_profit / accepts if accepts > 0 else 0

        all_episode_rewards.append(total_platform_reward)
        all_platform_profits.append(avg_platform_profit)
        all_accept_rates.append(accept_rate)

        if e % 50 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"\nEpisode: {e}/{episodes}")
            print(f"Platform Reward: {total_platform_reward:.2f}, Platform Payment to Workers: {current_info['total_payment']:.2f}")
            print(f"Avg Worker Profit: {avg_worker_profit:.2f}, Accept Rate: {accept_rate:.2%}")
            print(f"Epsilon: {platform_env.epsilon:.3f}, Worker Epsilon: {worker_env.epsilon:.3f}")
            print(f"Memory Usage: {current / 10 ** 6:.2f} MB (Peak: {peak / 10 ** 6:.2f} MB)")
            print(f"工人接受/拒绝统计: {worker_env.total_accepts}/{worker_env.total_rejects}")
            print(f"本轮任务完成总时间: {current_info['sum_t_k']:.2f}")

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("\n内存使用统计:")
    for stat in top_stats[:5]:
        print(stat)

    tracemalloc.stop()

    print("\n===== 训练结果 =====")
    print(f"最终平台支付给工人的总奖励: {all_platform_payments[-1]:.4f}")
    print(f"最终平台总利润: {all_platform_total_profits[-1]:.4f}")
    print(f"最终工人平均利润: {all_worker_profits[-1]:.4f}")
    print(f"最终接受率: {all_accept_rates[-1]:.2%}")
    print(f"平均任务完成总时间: {np.mean(all_sum_t_k):.2f}")
    print(f"总接受任务数: {worker_env.total_accepts}")
    print(f"总拒绝任务数: {worker_env.total_rejects}")

    return {
        'rewards': all_episode_rewards,
        'platform_total_profits': all_platform_total_profits,
        'platform_payments': all_platform_payments,
        'worker_profits': all_worker_profits,
        'accept_rates': all_accept_rates,
        'sum_t_k': all_sum_t_k
    }


if __name__ == "__main__":
    results = train_agent(episodes=1000, batch_size=32)