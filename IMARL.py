# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import random
import time
import math
from geopy.distance import geodesic
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import tracemalloc

np.random.seed(54)
random.seed(54)
torch.manual_seed(54)
torch.cuda.manual_seed(54)
torch.cuda.manual_seed_all(54)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


def calculate_statistics(data):
    if len(data) == 0:
        return {
            'mean': 0.0, 'var': 0.0, 'std': 0.0,
            'ci95_lower': 0.0, 'ci95_upper': 0.0,
            'cv': 0.0
        }

    mean_val = np.mean(data)
    var_val = np.var(data, ddof=1)
    std_val = np.std(data, ddof=1)
    n = len(data)

    if n >= 30:
        t_val = 1.96
    else:
        from scipy import stats
        t_val = stats.t.ppf(0.975, n - 1)

    ci_error = t_val * (std_val / math.sqrt(n))
    ci95_lower = mean_val - ci_error
    ci95_upper = mean_val + ci_error
    cv = std_val / mean_val if mean_val != 0 else 0.0

    return {
        'mean': round(mean_val, 4),
        'var': round(var_val, 4),
        'std': round(std_val, 4),
        'ci95_lower': round(ci95_lower, 4),
        'ci95_upper': round(ci95_upper, 4),
        'cv': round(cv, 4)
    }


def print_statistics(title, stats):
    print(f"\n=== {title} ===")
    print(f"均值: {stats['mean']}")
    print(f"方差: {stats['var']} | 标准差: {stats['std']}")
    print(f"95%置信区间: [{stats['ci95_lower']}, {stats['ci95_upper']}]")
    print(f"波动系数(CV): {stats['cv']} (越小越稳定)")


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
            tasks.append(Task(i, (lat, lon), n_workers_needed=5))
        return tasks


class UnstableVehicleAgent(nn.Module):

    def __init__(self, state_size, action_size):
        super(UnstableVehicleAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 32)
        self.dynamic_fc2 = None
        self.actor = nn.Linear(16, action_size)
        self.critic = nn.Linear(16, 1)
        self.gradient_freeze_prob = 0.1
        self.redundant_forward_times = 2

    def _update_dynamic_layer(self):
        hidden_size = random.choice([8, 16, 24])
        self.dynamic_fc2 = nn.Linear(32, hidden_size).to(device)
        nn.init.normal_(self.dynamic_fc2.weight, mean=0, std=0.2)
        nn.init.normal_(self.dynamic_fc2.bias, mean=0, std=0.2)

    def forward(self, x):
        for _ in range(self.redundant_forward_times):
            x1 = F.relu(self.fc1(x))
            self._update_dynamic_layer()
            x2 = F.relu(self.dynamic_fc2(x1))
            if x2.shape[-1] != 16:
                if x2.shape[-1] < 16:
                    x2 = F.pad(x2, (0, 16 - x2.shape[-1]))
                else:
                    x2 = x2[:, :16]

        action_logits = self.actor(x2)
        state_value = self.critic(x2)
        return F.softmax(action_logits, dim=-1), state_value


class MultiAgentEnvironment:
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
        self.memory = deque(maxlen=1000)
        self.gamma = 0.90
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.005

        self.model = UnstableVehicleAgent(self.state_size, self.action_size).to(device)
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
        noise = np.random.normal(0, 0.1)
        temp_reward = -c_d - self.c_n + noise

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
            probs, _ = self.model(state.unsqueeze(0))
            probs = probs + torch.randn_like(probs) * 0.1
            probs = F.softmax(probs, dim=-1)
            if np.random.rand() < 0.5:
                action = Categorical(probs).sample().item()
            else:
                action = np.random.choice([0, 1], p=probs.cpu().numpy()[0])
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        dynamic_batch_size = batch_size + random.randint(-5, 5)
        dynamic_batch_size = max(16, min(dynamic_batch_size, 64))
        minibatch = random.sample(self.memory, dynamic_batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states)
        dones = torch.BoolTensor(dones).to(device)

        probs, state_values = self.model(states)
        _, next_state_values = self.model(next_states)

        target_q = rewards + self.gamma * next_state_values.squeeze() * (~dones)
        current_q = state_values.squeeze()
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        if np.random.rand() < self.model.gradient_freeze_prob:
            return

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class MultiAgentPlatform:
    def __init__(self, agent_env, n_workers_pool=20):
        self.agent_env = agent_env
        self.n_workers_pool = n_workers_pool
        self.available_workers = None
        self.current_task = None

        self.gamma = 0.90
        self.epsilon = 0.95
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.005

    def reset_for_task(self, task, all_workers):
        self.current_task = task
        self.available_workers = all_workers.copy()
        self.available_workers['distance_to_task'] = self.available_workers.apply(
            lambda row: geodesic((row['latitude'], row['longitude']), task.location).kilometers, axis=1
        )

    def assign_worker(self, action):
        if action < 0 or action >= len(self.available_workers):
            return None, -1.0, False

        selected_worker = self.available_workers.iloc[action]
        worker_state = self.agent_env.get_worker_state(selected_worker)
        worker_action = self.agent_env.act(worker_state)
        _, _, _, worker_info = self.agent_env.step(selected_worker, worker_action)

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
            worker['final_profit'] = share - worker['c_d'] - self.agent_env.c_n
            total_reward += share

        log_terms = np.log([1 + w['t_i'] * w['q_i'] for w in task.assigned_workers])
        sum_log = np.sum(log_terms)
        task.total_profit = self.agent_env.lambda_param * np.log(1 + sum_log) - total_reward
        task.total_payment = total_reward
        task.completed = True

    def release_workers(self):
        for worker in self.current_task.assigned_workers:
            self.agent_env.reset_worker(worker['worker_id'])


def train_contrast_experiment(episodes=200, batch_size=32, n_tasks_per_episode=100, n_workers_pool=20):
    tracemalloc.start()

    worker_env = MultiAgentEnvironment('sh_csv/merged_shanghai_taxi_gps.csv')
    platform_env = MultiAgentPlatform(worker_env, n_workers_pool=n_workers_pool)
    task_generator = TaskGenerator(n_tasks_per_episode=n_tasks_per_episode)

    all_task_profits = []
    all_worker_profits = []
    all_task_times = []
    all_accept_rates = []
    all_step_times = []
    episode_times = []

    for episode in range(episodes):
        episode_start = time.time()

        all_workers = worker_env.data.drop_duplicates(subset='id').head(n_workers_pool)
        tasks = task_generator.generate_tasks()
        episode_profit = 0.0
        episode_worker_profit = []

        for task in tasks:
            platform_env.reset_for_task(task, all_workers)
            state = worker_env.get_worker_state(all_workers.iloc[0])
            task_done = False
            step = 0

            while not task_done:
                step_start = time.time()

                action = random.randrange(n_workers_pool) if np.random.rand() < 0.1 else step % len(
                    platform_env.available_workers)
                selected_worker, reward, accepted = platform_env.assign_worker(action)

                next_state = state
                worker_env.remember(state, action, reward, next_state, False)

                state = next_state
                step += 1

                step_time = time.time() - step_start
                all_step_times.append(step_time)

                if (len(task.assigned_workers) >= task.n_workers_needed or len(platform_env.available_workers) == 0):
                    task_done = True
                    platform_env.calculate_task_rewards()
                    final_reward = task.total_profit
                    worker_env.remember(state, action, final_reward, next_state, True)
                    episode_profit += final_reward
                    all_task_profits.append(task.total_profit)
                    all_task_times.append(task.sum_t_k)

                    if task.assigned_workers:
                        avg_worker_profit = np.mean([w['final_profit'] for w in task.assigned_workers])
                        episode_worker_profit.append(avg_worker_profit)
                        all_worker_profits.append(avg_worker_profit)

                    platform_env.release_workers()

        episode_time = time.time() - episode_start
        episode_times.append(episode_time)

        worker_env.replay(batch_size // 2)

        total_workers = worker_env.total_accepts + worker_env.total_rejects
        accept_rate = worker_env.total_accepts / total_workers if total_workers > 0 else 0
        all_accept_rates.append(accept_rate)

        if episode % 50 == 0:
            current, peak = tracemalloc.get_traced_memory()
            recent_profits = all_task_profits[-n_tasks_per_episode:] if all_task_profits else []
            recent_step_times = all_step_times[-1000:] if all_step_times else []

            profit_stats = calculate_statistics(recent_profits)
            time_stats = calculate_statistics(recent_step_times)

            print(f"\n===== Episode: {episode}/{episodes} =====")
            print(f"回合总利润: {episode_profit:.2f}, 近期单任务利润: 均值={profit_stats['mean']}, 方差={profit_stats['var']}")
            print(
                f"平均单步耗时: 均值={time_stats['mean']:.6f}s, 方差={time_stats['var']:.8f}, 95%CI=[{time_stats['ci95_lower']:.6f}, {time_stats['ci95_upper']:.6f}]")
            print(f"平均工人利润: {np.mean(episode_worker_profit):.2f}, 接受率: {accept_rate:.2%}")
            print(f"ε: {worker_env.epsilon:.3f}")
            print(f"内存使用: {current / 1e6:.2f}MB (峰值: {peak / 1e6:.2f}MB)")
            print(f"工人接受/拒绝: {worker_env.total_accepts}/{worker_env.total_rejects}")

    print("\n" + "=" * 50)
    print("训练结果 - 完整统计分析（含方差、置信区间）")
    print("=" * 50)

    profit_stats = calculate_statistics(all_task_profits)
    worker_profit_stats = calculate_statistics(all_worker_profits)
    task_time_stats = calculate_statistics(all_task_times)
    step_time_stats = calculate_statistics(all_step_times)
    accept_rate_stats = calculate_statistics(all_accept_rates)

    print_statistics("平台单任务利润（元）", profit_stats)
    print_statistics("工人平均利润（元）", worker_profit_stats)
    print_statistics("任务完成时间（小时）", task_time_stats)
    print_statistics("单步分配耗时（秒）- 计算复杂度指标", step_time_stats)
    print_statistics("工人任务接受率", accept_rate_stats)

    print("\n=== 训练稳定性总结 ===")
    stability_metrics = {
        "平台利润": profit_stats['cv'],
        "工人利润": worker_profit_stats['cv'],
        "任务时间": task_time_stats['cv'],
        "计算耗时": step_time_stats['cv']
    }
    for metric, cv in stability_metrics.items():
        if cv < 0.1:
            stability = "高度稳定"
        elif cv < 0.3:
            stability = "稳定"
        elif cv < 0.5:
            stability = "基本稳定"
        else:
            stability = "波动较大"
        print(f"{metric} 波动系数: {cv} - {stability}")

    print(f"\n=== 基础统计 ===")
    print(f"总接受任务数: {worker_env.total_accepts}")
    print(f"总训练回合数: {episodes}")
    print(f"总处理任务数: {len(all_task_profits)}")
    print(f"平均每回合耗时: {np.mean(episode_times):.2f}s")

    tracemalloc.stop()

    stats_df = pd.DataFrame({
        '指标': ['平台利润', '工人利润', '任务时间', '单步耗时', '接受率'],
        '均值': [
            profit_stats['mean'],
            worker_profit_stats['mean'],
            task_time_stats['mean'],
            step_time_stats['mean'],
            accept_rate_stats['mean']
        ],
        '方差': [
            profit_stats['var'],
            worker_profit_stats['var'],
            task_time_stats['var'],
            step_time_stats['var'],
            accept_rate_stats['var']
        ],
        '95%CI下限': [
            profit_stats['ci95_lower'],
            worker_profit_stats['ci95_lower'],
            task_time_stats['ci95_lower'],
            step_time_stats['ci95_lower'],
            accept_rate_stats['ci95_lower']
        ],
        '95%CI上限': [
            profit_stats['ci95_upper'],
            worker_profit_stats['ci95_upper'],
            task_time_stats['ci95_upper'],
            step_time_stats['ci95_upper'],
            accept_rate_stats['ci95_upper']
        ],
        '波动系数': [
            profit_stats['cv'],
            worker_profit_stats['cv'],
            task_time_stats['cv'],
            step_time_stats['cv'],
            accept_rate_stats['cv']
        ]
    })
    stats_df.to_csv('statistics/contrast_imarl_two.csv', index=False, encoding='utf-8-sig')
    print("\n对比试验统计结果已保存到 contrast_imarl_ten.csv")

    return {
        'task_profits': all_task_profits,
        'worker_profits': all_worker_profits,
        'task_times': all_task_times,
        'step_times': all_step_times,
        'accept_rates': all_accept_rates,
        'profit_stats': profit_stats,
        'worker_profit_stats': worker_profit_stats,
        'task_time_stats': task_time_stats,
        'step_time_stats': step_time_stats,
        'accept_rate_stats': accept_rate_stats
    }


if __name__ == "__main__":
    results = train_contrast_experiment(episodes=200, batch_size=32, n_tasks_per_episode=100)