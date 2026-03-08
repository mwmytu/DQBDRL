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
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

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
    def __init__(self, n_tasks_per_episode=100):
        self.n_tasks_per_episode = n_tasks_per_episode
        self.base_location = (31.23, 121.48)

    def generate_tasks(self):
        tasks = []
        for i in range(self.n_tasks_per_episode):
            lat = self.base_location[0] + random.uniform(-0.1, 0.1)
            lon = self.base_location[1] + random.uniform(-0.1, 0.1)
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
        self.electricity_price = 0.5
        self.power_consumption = 0.2
        self.c_n = 0.1
        self.lambda_param = 50

        try:
            self.data = pd.read_csv(data_path)
        except:
            print("警告：未找到数据文件，使用模拟数据训练")
            self.data = self._generate_sample_data(1000)

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
        self.loss_history = []
        self.episode_loss = []

    def _generate_sample_data(self, n_workers=1000):
        data = pd.DataFrame()
        data['id'] = range(n_workers)
        data['latitude'] = 31.23 + np.random.uniform(-0.1, 0.1, n_workers)
        data['longitude'] = 121.48 + np.random.uniform(-0.1, 0.1, n_workers)
        data['quantity'] = np.random.uniform(0, 100, n_workers)
        data['volume'] = np.random.uniform(0, 100, n_workers)
        data['speed'] = np.random.uniform(20, 60, n_workers)
        return data

    def _preprocess_data(self, data):
        data = data.drop_duplicates(subset=['id', 'latitude', 'longitude'])
        unique_ids = data['id'].unique()
        if len(unique_ids) > 1000:
            selected_ids = np.random.choice(unique_ids, 1000, replace=False)
            data = data[data['id'].isin(selected_ids)]
        data = data[(data['quantity'] != 0) & (data['speed'] > 0)]

        data['q_i'] = 0.5 * data['quantity'] / 100 + 0.5 * data['volume'] / 100
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
        state = self.get_worker_state(worker)

        if action == 0:
            self.total_rejects += 1
            next_state = state
            reward = 0.0
            done = True
            worker_info = {
                'accepted': False,
                't_i': 0.0,
                'c_d': 0.0,
                'q_i': worker['q_i']
            }
        else:
            self.total_accepts += 1
            t_i, c_d = self.calculate_worker_basics(worker)
            reward = -c_d - self.c_n

            worker_copy = worker.copy()
            worker_copy['distance_to_task'] = 0.0
            next_state = self.get_worker_state(worker_copy)
            done = True
            worker_info = {
                'accepted': True,
                't_i': t_i,
                'c_d': c_d,
                'q_i': worker['q_i'],
                'worker_id': worker['id']
            }

        self.remember(state, action, reward, next_state, done)

        return next_state, reward, done, worker_info

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            return self.model(state.unsqueeze(0)).max(1)[1].item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            if len(self.loss_history) == 0:
                self.loss_history.append(0.0)
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

        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def get_avg_loss(self):
        if len(self.loss_history) == 0:
            avg_loss = 0.0
        else:
            avg_loss = np.mean(self.loss_history)

        self.episode_loss.append(avg_loss)
        self.loss_history = []
        return avg_loss


class PlatformEnvironment:
    def __init__(self, worker_env, n_workers_pool=20):
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

        self.model = DuelingDQN(self.state_size, self.action_size).to(device)
        self.target_model = DuelingDQN(self.state_size, self.action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss_history = []
        self.episode_loss = []

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
        _, reward, _, worker_info = self.worker_env.step(selected_worker, worker_action)

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
            if len(self.loss_history) == 0:
                self.loss_history.append(0.0)
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
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def get_avg_loss(self):
        if len(self.loss_history) == 0:
            avg_loss = 0.0
        else:
            avg_loss = np.mean(self.loss_history)

        self.episode_loss.append(avg_loss)
        self.loss_history = []
        return avg_loss


def plot_loss_curves(worker_loss, platform_loss, smooth_window=10):
    if len(worker_loss) == 0:
        worker_loss = [0.0] * len(platform_loss)
    if len(platform_loss) == 0:
        platform_loss = [0.0] * len(worker_loss)

    min_len = max(1, min(len(worker_loss), len(platform_loss)))
    worker_loss = worker_loss[:min_len]
    platform_loss = platform_loss[:min_len]
    episodes = range(1, min_len + 1)

    smooth_window = min(smooth_window, min_len)
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_window = max(3, smooth_window)

    try:
        if smooth_window <= min_len:
            worker_loss_smooth = savgol_filter(worker_loss, smooth_window, 1)
            platform_loss_smooth = savgol_filter(platform_loss, smooth_window, 1)
        else:
            worker_loss_smooth = worker_loss
            platform_loss_smooth = platform_loss
    except:
        worker_loss_smooth = worker_loss
        platform_loss_smooth = platform_loss

    plt.figure(figsize=(12, 6))

    plt.plot(episodes, worker_loss, alpha=0.3, color='blue', label='Original loss of worker network')
    plt.plot(episodes, platform_loss, alpha=0.3, color='red', label='Original loss of platform network')
    plt.plot(episodes, worker_loss_smooth, color='blue', linewidth=2,
             label=f'Loss of worker network')
    plt.plot(episodes, platform_loss_smooth, color='red', linewidth=2,
             label=f'Loss of platform network')

    plt.xlabel('Episode', fontsize=20, fontfamily='Times New Roman')
    plt.ylabel('Loss', fontsize=20, fontfamily='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman', 'size': 22})
    plt.xticks(fontsize=20, fontfamily='Times New Roman')
    plt.yticks(fontsize=20, fontfamily='Times New Roman')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n===== 损失曲线统计 =====")
    if len(worker_loss_smooth) > 0:
        print(f"工人网络最终损失: {worker_loss_smooth[-1]:.4f} (第{min_len}回合)")
        if len(worker_loss_smooth) > 1:
            loss_change = (worker_loss_smooth[0] - worker_loss_smooth[-1])
            if worker_loss_smooth[0] > 0:
                decline_rate = loss_change / worker_loss_smooth[0] * 100
                print(f"工人网络损失下降率: {decline_rate:.2f}%")
            else:
                print(f"工人网络损失变化: {loss_change:.4f}")
    else:
        print("工人网络无损失数据")

    if len(platform_loss_smooth) > 0:
        print(f"平台网络最终损失: {platform_loss_smooth[-1]:.4f} (第{min_len}回合)")
        if len(platform_loss_smooth) > 1:
            loss_change = (platform_loss_smooth[0] - platform_loss_smooth[-1])
            if platform_loss_smooth[0] > 0:
                decline_rate = loss_change / platform_loss_smooth[0] * 100
                print(f"平台网络损失下降率: {decline_rate:.2f}%")
            else:
                print(f"平台网络损失变化: {loss_change:.4f}")
    else:
        print("平台网络无损失数据")


def train_multitask_agent(episodes=1000, batch_size=32, n_tasks_per_episode=100, n_workers_pool=20):
    tracemalloc.start()

    worker_env = WorkerEnvironment('sh_csv/merged_shanghai_taxi_gps.csv')
    platform_env = PlatformEnvironment(worker_env, n_workers_pool=n_workers_pool)
    task_generator = TaskGenerator(n_tasks_per_episode=n_tasks_per_episode)

    # 统计信息
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

        worker_avg_loss = worker_env.get_avg_loss()
        platform_avg_loss = platform_env.get_avg_loss()

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
            print(f"工人网络损失: {worker_avg_loss:.4f}, 平台网络损失: {platform_avg_loss:.4f}")
            print(f"内存使用: {current / 1e6:.2f}MB (峰值: {peak / 1e6:.2f}MB)")
            print(f"工人接受/拒绝: {worker_env.total_accepts}/{worker_env.total_rejects}")

    plot_loss_curves(worker_env.episode_loss, platform_env.episode_loss, smooth_window=10)

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
        'worker_loss': worker_env.episode_loss,
        'platform_loss': platform_env.episode_loss
    }


if __name__ == "__main__":
    results = train_multitask_agent(episodes=1000, batch_size=32, n_tasks_per_episode=50)