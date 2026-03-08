# -*- coding: UTF-8 -*-
# @Author:zhuxiao ASUS
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

np.random.seed(54)
random.seed(54)
torch.manual_seed(54)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class Task:
    def __init__(self, task_id, location, b_j=21, n_workers_needed=5):
        self.task_id = task_id
        self.location = location
        self.b_j = b_j
        self.n_workers_needed = n_workers_needed
        self.assigned_workers = []
        self.sum_t_k = 0.0
        self.total_profit = 0.0
        self.total_payment = 0.0
        self.completed = False


class TaskGenerator:
    def __init__(self, n_tasks_per_episode=20):
        self.n_tasks_per_episode = n_tasks_per_episode
        self.base_location = (31.23, 121.48)

    def generate_tasks(self):
        tasks = []
        for i in range(self.n_tasks_per_episode):
            lat = self.base_location[0] + random.uniform(-0.15, 0.15)
            lon = self.base_location[1] + random.uniform(-0.15, 0.15)
            tasks.append(
                Task(
                    task_id=i,
                    location=(lat, lon),
                    n_workers_needed=5
                )
            )
        return tasks


class DQN(nn.Module):

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


class WorkerEnvironment:
    def __init__(self, data_path):
        self.electricity_price = 0.5
        self.power_consumption = 0.2
        self.c_n = 0.1
        self.lambda_param = 50

        self.data = pd.read_csv(data_path)
        self.data = self._preprocess_data(self.data)
        self.original_states = {}
        for _, worker in self.data.iterrows():
            self.original_states[worker['id']] = {
                'latitude': worker['latitude'],
                'longitude': worker['longitude']
            }

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

        self.total_accepts = 0
        self.total_rejects = 0

    def _preprocess_data(self, data):
        data = data.drop_duplicates(subset=['id', 'latitude', 'longitude'])
        unique_ids = data['id'].unique()
        if len(unique_ids) > 1000:
            selected_ids = np.random.choice(unique_ids, 1000, replace=False)
            data = data[data['id'].isin(selected_ids)]
        data = data[(data['quantity'] != 0) & (data['speed'] > 0)]

        # data['q_i'] = np.random.uniform(low=0.0, high=1.0, size=len(data))
        data['q_i'] = 0.5 * data['quantity'] / 100 + 0.5 * data['volume'] / 100
        # data['q_i'] = data['quantity'] / 100
        # data['q_i'] = data['volume'] / 100

        data['distance_to_task'] = 0.0
        return data[['id', 'latitude', 'longitude', 'q_i', 'speed', 'distance_to_task']]

    def reset_worker(self, worker_id):
        if worker_id in self.original_states:
            orig = self.original_states[worker_id]
            self.data.loc[self.data['id'] == worker_id, 'latitude'] = orig['latitude']
            self.data.loc[self.data['id'] == worker_id, 'longitude'] = orig['longitude']

    def get_worker_state(self, worker):
        return torch.FloatTensor([
            worker['latitude'],
            worker['longitude'],
            worker['q_i'],
            worker['speed'],
            worker['distance_to_task']
        ]).to(device)

    def calculate_worker_basics(self, worker):
        t_i = worker['distance_to_task'] / worker['speed']
        c_d = worker['distance_to_task'] * self.power_consumption * self.electricity_price
        return t_i, c_d

    def step(self, worker, action):
        if action == 0:
            self.total_rejects += 1
            return self.get_worker_state(worker), 0.0, True, {
                'accepted': False,
                't_i': 0.0,
                'c_d': 0.0,
                'q_i': worker['q_i']
            }

        self.total_accepts += 1
        t_i, c_d = self.calculate_worker_basics(worker)
        temp_reward = -c_d - self.c_n

        worker_copy = worker.copy()
        worker_copy['distance_to_task'] = 0.0
        return self.get_worker_state(worker_copy), temp_reward, True, {
            'accepted': True,
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
            return self.model(state.unsqueeze(0)).max(1)[1].item()

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
            next_q = self.model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PlatformEnvironment:
    def __init__(self, worker_env, n_workers_pool=60):
        self.worker_env = worker_env
        self.n_workers_pool = n_workers_pool
        self.available_workers = None
        self.current_task = None

        self.state_size = self.n_workers_pool * 5
        self.action_size = self.n_workers_pool
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def reset_for_task(self, task, all_workers):
        self.current_task = task
        self.available_workers = all_workers.copy()
        self.available_workers['distance_to_task'] = self.available_workers.apply(
            lambda row: geodesic(
                (row['latitude'], row['longitude']),
                task.location
            ).kilometers, axis=1
        )

    def get_platform_state(self):
        state = []
        for _, worker in self.available_workers.iterrows():
            state.extend([
                worker['latitude'],
                worker['longitude'],
                worker['q_i'],
                worker['speed'],
                worker['distance_to_task']
            ])
        padding = [0] * (self.state_size - len(state))
        return torch.FloatTensor(state + padding).to(device)

    def assign_worker(self, action):
        if action < 0 or action >= len(self.available_workers):
            return None, -1.0, False

        selected_worker = self.available_workers.iloc[action]
        worker_state = self.worker_env.get_worker_state(selected_worker)
        worker_action = self.worker_env.act(worker_state)
        _, _, _, worker_info = self.worker_env.step(selected_worker, worker_action)

        if worker_action == 1:
            self.current_task.assigned_workers.append({
                'worker_id': selected_worker['id'],
                't_i': worker_info['t_i'],
                'c_d': worker_info['c_d'],
                'q_i': worker_info['q_i']
            })
            self.current_task.sum_t_k += worker_info['t_i']

            self.available_workers = self.available_workers.drop(
                self.available_workers.index[action]
            )
            return selected_worker, 0.0, True
        else:
            self.available_workers = self.available_workers.drop(
                self.available_workers.index[action]
            )
            return None, -0.1, False

    def calculate_task_rewards(self):
        task = self.current_task
        if not task.assigned_workers:
            task.total_profit = 0.0
            return

        n_workers = len(task.assigned_workers)
        total_reward = 0.0
        for worker in task.assigned_workers:
            if task.sum_t_k == 0:
                share = 0.0
            else:
                share = (
                    min(worker['q_i'], 0.8) *
                    task.b_j *
                    (worker['t_i'] / task.sum_t_k)
                )
            worker['reward_share'] = share
            worker['final_profit'] = share - worker['c_d'] - self.worker_env.c_n
            total_reward += share

        log_terms = np.log([1 + w['t_i'] * w['q_i'] for w in task.assigned_workers])
        sum_log = np.sum(log_terms)
        task.total_profit = self.worker_env.lambda_param * np.log(1 + sum_log) - total_reward
        task.total_payment = total_reward
        task.completed = True

    def release_workers(self):
        for worker in self.current_task.assigned_workers:
            self.worker_env.reset_worker(worker['worker_id'])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, next_state, reward, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            return self.model(state.unsqueeze(0)).max(1)[1].item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=device, dtype=torch.bool)
        non_final_next = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_vals = self.model(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_vals = torch.zeros(self.batch_size, device=device)
        next_state_vals[non_final_mask] = self.target_model(non_final_next).max(1)[0].detach()
        expected_vals = (next_state_vals * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_vals, expected_vals.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_multitask_agent(episodes=300, batch_size=32, n_tasks_per_episode=20, n_workers_pool=60):
    tracemalloc.start()

    worker_env = WorkerEnvironment('sh_csv/merged_shanghai_taxi_gps.csv')
    platform_env = PlatformEnvironment(worker_env, n_workers_pool=n_workers_pool)
    task_generator = TaskGenerator(n_tasks_per_episode=n_tasks_per_episode)

    all_task_profits = []
    all_worker_profits = []
    all_task_times = []
    all_accept_rates = []

    for episode in range(episodes):
        all_workers = worker_env.data.drop_duplicates(subset='id').head(n_workers_pool)
        tasks = task_generator.generate_tasks()
        episode_profit = 0.0
        episode_worker_profit = []

        for task in tasks:
            platform_env.reset_for_task(task, all_workers)
            state = platform_env.get_platform_state()
            task_done = False
            step = 0

            while not task_done:
                action = platform_env.act(state)
                selected_worker, reward, accepted = platform_env.assign_worker(action)

                next_state = platform_env.get_platform_state()
                platform_env.remember(
                    state,
                    torch.tensor([action], device=device),
                    torch.tensor([reward], device=device),
                    next_state,
                    False
                )

                state = next_state
                step += 1

                if (len(task.assigned_workers) >= task.n_workers_needed
                    or len(platform_env.available_workers) == 0):
                    task_done = True
                    platform_env.calculate_task_rewards()
                    final_reward = task.total_profit
                    platform_env.remember(
                        state,
                        torch.tensor([action], device=device),
                        torch.tensor([final_reward], device=device),
                        next_state,
                        True
                    )
                    episode_profit += final_reward
                    all_task_profits.append(task.total_profit)
                    all_task_times.append(task.sum_t_k)

                    if task.assigned_workers:
                        avg_worker_profit = np.mean([w['final_profit'] for w in task.assigned_workers])
                        episode_worker_profit.append(avg_worker_profit)
                        all_worker_profits.append(avg_worker_profit)

                    platform_env.release_workers()

        platform_env.replay()
        worker_env.replay(batch_size)

        if episode % 100 == 0:
            platform_env.update_target_model()

        total_workers = worker_env.total_accepts + worker_env.total_rejects
        accept_rate = worker_env.total_accepts / total_workers if total_workers > 0 else 0
        all_accept_rates.append(accept_rate)

        if episode % 50 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"\nEpisode: {episode}/{episodes}")
            print(f"回合总利润: {episode_profit:.2f}, 平均单任务利润: {np.mean(all_task_profits[-n_tasks_per_episode:]):.2f}")
            print(f"平均工人利润: {np.mean(episode_worker_profit):.2f}, 接受率: {accept_rate:.2%}")
            print(f"平台ε: {platform_env.epsilon:.3f}, 工人ε: {worker_env.epsilon:.3f}")
            print(f"内存使用: {current/1e6:.2f}MB (峰值: {peak/1e6:.2f}MB)")
            print(f"工人接受/拒绝: {worker_env.total_accepts}/{worker_env.total_rejects}")

    print("\n===== 训练结果 =====")
    print(f"平均平台利润: {np.mean(all_task_profits):.2f}")
    print(f"平均工人利润: {np.mean(all_worker_profits):.2f}")
    print(f"平均任务时间: {np.mean(all_task_times):.2f}")
    print(f"最终接受率: {all_accept_rates[-1]:.2%}")
    print(f"总接受任务数: {worker_env.total_accepts}")

    tracemalloc.stop()
    return {
        'task_profits': all_task_profits,
        'worker_profits': all_worker_profits,
        'task_times': all_task_times,
    }


if __name__ == "__main__":
    results = train_multitask_agent(episodes=300, batch_size=32, n_tasks_per_episode=20)