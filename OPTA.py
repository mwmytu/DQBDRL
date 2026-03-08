# -*- coding: UTF-8 -*-
# @Author:zhuxiao ASUS
import pandas as pd
import numpy as np
import random
import math
from collections import defaultdict, deque
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np.random.seed(54)
random.seed(54)
torch.manual_seed(54)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
        self.extra_reward = 0


class TaskGenerator:

    def __init__(self, n_tasks_per_episode=80):
        self.n_tasks_per_episode = n_tasks_per_episode
        self.base_location = (22.54, 114.05)  # 31.23, 121.48 22.54, 114.05

    def generate_tasks(self):
        tasks = []
        for i in range(self.n_tasks_per_episode):
            lat = self.base_location[0] + random.uniform(-0.1, 0.1)
            lon = self.base_location[1] + random.uniform(-0.1, 0.1)
            tasks.append(Task(
                task_id=i,
                location=(lat, lon),
                n_workers_needed=5
            ))
        return tasks


class Worker:

    def __init__(self, worker_id, latitude, longitude, q_i, speed):
        self.id = worker_id
        self.latitude = latitude
        self.longitude = longitude
        self.q_i = q_i
        self.speed = speed
        self.current_location = (latitude, longitude)
        self.available = True
        self.current_time = 0
        self.remaining_time = 8 * 3600
        self.original_location = (latitude, longitude)


class WorkerEnvironment:

    def __init__(self, data_path, max_workers=1000, n_workers_pool=100):
        self.max_workers = max_workers
        self.n_workers_pool = n_workers_pool
        self.electricity_price = 0.5
        self.power_consumption = 0.2
        self.c_n = 0.1
        self.lambda_param = 50

        self.data = pd.read_csv(data_path)
        self.data = self._preprocess_data(self.data)

        self.all_workers = []
        worker_count = 0
        for _, row in self.data.iterrows():
            if worker_count >= self.max_workers:
                break
            self.all_workers.append(Worker(
                worker_id=row['id'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                q_i=row['q_i'],
                speed=row['speed']
            ))
            worker_count += 1

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

        self.episode_accepts = 0
        self.episode_rejects = 0

    def _preprocess_data(self, data):
        data = data.drop_duplicates(subset=['id', 'latitude', 'longitude'])
        unique_ids = data['id'].unique()
        if len(unique_ids) > self.max_workers:
            selected_ids = np.random.choice(unique_ids, self.max_workers, replace=False)
            data = data[data['id'].isin(selected_ids)]
        data = data[(data['quantity'] != 0) & (data['speed'] > 0)]
        data['q_i'] = 0.5 * data['quantity'] / 100 + 0.5 * data['volume'] / 100
        return data[['id', 'latitude', 'longitude', 'q_i', 'speed']]

    def reset_episode_stats(self):
        self.episode_accepts = 0
        self.episode_rejects = 0

    def get_workers_for_episode(self, n_workers=None):
        if n_workers is None:
            n_workers = self.n_workers_pool

        if len(self.all_workers) > n_workers:
            selected_workers = random.sample(self.all_workers, n_workers)
        else:
            selected_workers = self.all_workers.copy()

        for worker in selected_workers:
            worker.available = True
            worker.current_location = worker.original_location
            worker.remaining_time = 8 * 3600
            worker.current_time = 0

        return selected_workers

    def calculate_distance(self, loc1, loc2):
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111

    def get_worker_state(self, worker, task_location):
        distance = self.calculate_distance(worker.current_location, task_location)
        return torch.FloatTensor([
            worker.latitude,
            worker.longitude,
            worker.q_i,
            worker.speed,
            distance
        ]).to(device)

    def calculate_worker_basics(self, worker, task_location):
        distance = self.calculate_distance(worker.current_location, task_location)
        t_i = distance / max(worker.speed, 1.0)
        c_d = distance * self.power_consumption * self.electricity_price
        return t_i, c_d

    def step(self, worker, task_location, action):
        if action == 0:
            self.episode_rejects += 1
            return 0.0, {'accepted': False, 't_i': 0.0, 'c_d': 0.0, 'q_i': worker.q_i}

        self.episode_accepts += 1
        t_i, c_d = self.calculate_worker_basics(worker, task_location)

        worker_utility = (21 * worker.q_i) - c_d - self.c_n
        if worker_utility < 0:
            self.episode_rejects += 1
            self.episode_accepts -= 1
            return 0.0, {'accepted': False, 't_i': 0.0, 'c_d': 0.0, 'q_i': worker.q_i}

        worker.current_location = task_location
        worker.available = False
        worker.remaining_time -= t_i * 3600

        return worker_utility, {
            'accepted': True,
            't_i': t_i,
            'c_d': c_d,
            'q_i': worker.q_i,
            'worker_id': worker.id,
            'worker_utility': worker_utility
        }

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            return self.model(state.unsqueeze(0)).max(1)[1].item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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



class ORTAAllocator:

    def __init__(self):
        self.matched_pairs = []

    def pre_allocate(self, workers: List[Worker], tasks: List[Task], current_time):
        bipartite_graph = []

        for worker in workers:
            if not worker.available:
                continue
            for task in tasks:
                if task.completed:
                    continue

                if self._can_perform_task(worker, task, current_time):
                    platform_utility = self._calculate_platform_utility(worker, task)
                    if platform_utility >= 0:
                        bipartite_graph.append((worker.id, task.task_id, platform_utility))

        matched_pairs = self._simple_greedy_matching(bipartite_graph, workers, tasks)
        return matched_pairs

    def _can_perform_task(self, worker, task, current_time):
        if worker.remaining_time <= 0:
            return False

        distance = self._calculate_distance(worker.current_location, task.location)
        travel_time = distance / max(worker.speed, 1.0)

        if travel_time * 3600 > worker.remaining_time:
            return False

        worker_utility = self._calculate_worker_utility(worker, task)
        return worker_utility >= 0

    def _calculate_worker_utility(self, worker, task):
        distance = self._calculate_distance(worker.current_location, task.location)
        travel_cost = distance * 0.2
        total_payment = task.b_j + task.extra_reward
        return total_payment * worker.q_i - travel_cost

    def _calculate_platform_utility(self, worker, task):
        worker_utility = self._calculate_worker_utility(worker, task)
        if worker_utility < 0:
            return -1
        return (task.b_j * 0.7) - (task.b_j * worker.q_i)

    def _calculate_distance(self, loc1, loc2):
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111

    def _simple_greedy_matching(self, bipartite_graph, workers, tasks):
        if not bipartite_graph:
            return []

        bipartite_graph.sort(key=lambda x: x[2], reverse=True)
        matched_pairs = []
        matched_workers = set()
        matched_tasks = set()

        for worker_id, task_id, utility in bipartite_graph:
            if worker_id not in matched_workers and task_id not in matched_tasks:
                matched_pairs.append((worker_id, task_id, utility))
                matched_workers.add(worker_id)
                matched_tasks.add(task_id)

        return matched_pairs


class OPTAAllocator(ORTAAllocator):

    def __init__(self, worker_env):
        super().__init__()
        self.worker_env = worker_env

    def allocate_with_continuation(self, workers: List[Worker], tasks: List[Task], time_window):
        all_matched_pairs = []
        remaining_tasks = tasks.copy()

        for task in remaining_tasks:
            task.assigned_workers = []
            task.sum_t_k = 0.0
            task.completed = False

        current_time = time_window[0]

        while current_time <= time_window[1] and remaining_tasks:
            available_workers = [w for w in workers if w.available and w.remaining_time > 0]

            if not available_workers:
                break

            matched_pairs = self.pre_allocate(available_workers, remaining_tasks, current_time)

            for worker_id, task_id, utility in matched_pairs:
                worker = next(w for w in available_workers if w.id == worker_id)
                task = next(t for t in remaining_tasks if t.task_id == task_id)

                worker_state = self.worker_env.get_worker_state(worker, task.location)
                worker_action = self.worker_env.act(worker_state)
                reward, worker_info = self.worker_env.step(worker, task.location, worker_action)

                if worker_info['accepted']:
                    task.assigned_workers.append({
                        'worker': worker,
                        't_i': worker_info['t_i'],
                        'c_d': worker_info['c_d'],
                        'q_i': worker_info['q_i'],
                        'utility': worker_info.get('worker_utility', 0)
                    })
                    task.sum_t_k += worker_info['t_i']

                    worker.available = False
                    all_matched_pairs.append((worker_id, task_id, utility))

                    if len(task.assigned_workers) >= task.n_workers_needed:
                        task.completed = True

                next_state = self.worker_env.get_worker_state(worker, task.location)
                self.worker_env.remember(worker_state, worker_action, reward, next_state, True)

            remaining_tasks = [t for t in remaining_tasks if not t.completed]
            current_time += 300

        return all_matched_pairs, remaining_tasks


class QLearningIncentiveMechanism:

    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: [0, 0])

    def choose_action(self, state, available_budget):
        if random.random() < self.exploration_rate:
            return random.choice([0, 1])
        else:
            if available_budget <= 0:
                return 1
            q_values = self.q_table[state]
            return 0 if q_values[0] >= q_values[1] else 1

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * next_max_q)
        self.q_table[state][action] = new_q

    def calculate_incentive_reward(self, task, base_reward=3.0):
        assigned_ratio = len(task.assigned_workers) / task.n_workers_needed if task.assigned_workers else 0
        return base_reward * (1 + assigned_ratio)


class OPTA_QD_System:

    def __init__(self, worker_env, platform_budget=500.0):
        self.worker_env = worker_env
        self.opta_allocator = OPTAAllocator(worker_env)
        self.q_learning = QLearningIncentiveMechanism()
        self.platform_budget = platform_budget

    def run_complete_allocation(self, tasks: List[Task], time_window, workers: List[Worker]):
        print("开始初始分配...")

        initial_matches, pre_churn_tasks = self.opta_allocator.allocate_with_continuation(
            workers, tasks, time_window)

        print(f"初始匹配尝试: {len(initial_matches)} 对")
        print(f"预流失任务: {len(pre_churn_tasks)} 个")

        completed_tasks = [t for t in tasks if t.completed]
        print(f"实际完成的任务: {len(completed_tasks)} 个")

        if pre_churn_tasks:
            print("开始激励机制...")
            incentivized_matches = self._apply_incentive_mechanism(workers, pre_churn_tasks, time_window)
            all_matches = initial_matches + incentivized_matches
        else:
            all_matches = initial_matches

        task_profits, worker_profits, task_times = self._calculate_final_metrics(tasks)

        print(f"总有效匹配: {len(all_matches)} 对")
        return all_matches, task_profits, worker_profits, task_times

    def _apply_incentive_mechanism(self, workers, pre_churn_tasks, time_window):
        incentivized_matches = []
        current_budget = self.platform_budget

        unfinished_tasks = [t for t in pre_churn_tasks if not t.completed]

        if not unfinished_tasks:
            return incentivized_matches

        self._train_q_learning(workers, unfinished_tasks, time_window, episodes=10)

        for task in unfinished_tasks:
            if task.completed:
                continue

            state = task.task_id
            q_values = self.q_learning.q_table[state]

            if q_values[0] > q_values[1] and current_budget > 0:
                incentive_reward = self.q_learning.calculate_incentive_reward(task)

                if current_budget >= incentive_reward:
                    task.extra_reward = incentive_reward

                    available_workers = [w for w in workers if w.available and w.remaining_time > 0]
                    if available_workers:
                        temp_matches, _ = self.opta_allocator.allocate_with_continuation(
                            available_workers, [task], time_window)

                        if temp_matches and task.completed:
                            incentivized_matches.extend(temp_matches)
                            current_budget -= incentive_reward
                            print(f"任务 {task.task_id} 激励成功")

        return incentivized_matches

    def _train_q_learning(self, workers, pre_churn_tasks, time_window, episodes=10):
        for episode in range(episodes):
            current_budget = self.platform_budget
            shuffled_tasks = pre_churn_tasks.copy()
            random.shuffle(shuffled_tasks)

            for i, task in enumerate(shuffled_tasks):
                if task.completed:
                    continue

                state = task.task_id
                next_state = shuffled_tasks[i + 1].task_id if i < len(shuffled_tasks) - 1 else "terminal"

                action = self.q_learning.choose_action(state, current_budget)

                if action == 0:
                    incentive_reward = self.q_learning.calculate_incentive_reward(task)

                    if current_budget >= incentive_reward:
                        task.extra_reward = incentive_reward
                        available_workers = [w for w in workers if w.available and w.remaining_time > 0]

                        if available_workers:
                            temp_matches, _ = self.opta_allocator.allocate_with_continuation(
                                available_workers, [task], time_window)

                            if temp_matches and task.completed:
                                task_profits, _, _ = self._calculate_final_metrics([task])
                                reward = task_profits[0] * 0.5 if task_profits else 0  # 降低奖励
                                current_budget -= incentive_reward
                            else:
                                reward = -incentive_reward * 0.7  # 降低惩罚
                        else:
                            reward = 0
                    else:
                        reward = 0
                else:
                    reward = 0.01

                self.q_learning.update_q_value(state, action, reward, next_state)
                task.extra_reward = 0

    def _calculate_final_metrics(self, tasks):
        task_profits = []
        worker_profits = []
        task_times = []

        for task in tasks:
            if not task.completed or len(task.assigned_workers) == 0:
                continue

            total_time = sum(worker_info['t_i'] for worker_info in task.assigned_workers)
            task_times.append(total_time)

            total_payment = 0
            worker_task_profits = []

            for worker_info in task.assigned_workers:
                if task.sum_t_k > 0:
                    share = (min(worker_info['q_i'], 0.7) * task.b_j * 0.6 *
                             (worker_info['t_i'] / task.sum_t_k))
                else:
                    share = 0

                total_payment += share
                worker_profit = share - worker_info['c_d'] - 0.2
                worker_task_profits.append(worker_profit)

            log_terms = [np.log(1 + worker_info['t_i'] * worker_info['q_i'] * 0.6)
                         for worker_info in task.assigned_workers]
            sum_log = np.sum(log_terms) if log_terms else 0
            task_profit = 30 * np.log(1 + sum_log) - total_payment

            task.total_profit = task_profit
            task.total_payment = total_payment

            task_profits.append(task_profit)
            worker_profits.extend(worker_task_profits)

        return task_profits, worker_profits, task_times


def train_multitask_system(episodes=100, n_tasks_per_episode=80, max_workers=1000, n_workers_pool=100):

    worker_env = WorkerEnvironment('electricVehicle_csv/merged_electricVehicle_gps.csv',
                                   max_workers=max_workers,
                                   n_workers_pool=n_workers_pool)
    task_generator = TaskGenerator(n_tasks_per_episode=n_tasks_per_episode)
    opta_qd_system = OPTA_QD_System(worker_env)

    all_task_profits = []
    all_worker_profits = []
    all_task_times = []
    all_accept_rates = []

    for episode in range(episodes):
        print(f"\n=== Episode {episode + 1}/{episodes} ===")

        worker_env.reset_episode_stats()

        tasks = task_generator.generate_tasks()
        episode_workers = worker_env.get_workers_for_episode(n_workers_pool)

        print(f"使用工人数量: {len(episode_workers)}")
        print(f"任务数量: {len(tasks)}")

        time_window = (0, 8 * 3600)

        matches, task_profits, worker_profits, task_times = opta_qd_system.run_complete_allocation(
            tasks, time_window, episode_workers)

        all_task_profits.extend(task_profits)
        all_worker_profits.extend(worker_profits)
        all_task_times.extend(task_times)

        total_decisions = worker_env.episode_accepts + worker_env.episode_rejects
        accept_rate = worker_env.episode_accepts / total_decisions if total_decisions > 0 else 0
        all_accept_rates.append(accept_rate)

        if len(worker_env.memory) >= 32:
            worker_env.replay(32)

        if (episode + 1) % 10 == 0:
            completed_tasks = len([t for t in tasks if t.completed])
            print(f"\n进度报告 - Episode {episode + 1}:")
            print(f"完成的任务: {completed_tasks}/{len(tasks)}")
            print(f"平均平台利润: {np.mean(task_profits) if task_profits else 0:.2f}")
            print(f"平均工人利润: {np.mean(worker_profits) if worker_profits else 0:.2f}")
            print(f"平均任务时间: {np.mean(task_times) if task_times else 0:.2f}")
            print(f"接受率: {accept_rate:.2%}")
            print(f"工人ε: {worker_env.epsilon:.3f}")

    final_task_profits = [p for p in all_task_profits if p is not None]
    final_worker_profits = [p for p in all_worker_profits if p is not None]
    final_task_times = [t for t in all_task_times if t is not None]

    print("\n===== 训练结果 =====")
    print(f"平均平台利润: {np.mean(final_task_profits) if final_task_profits else 0:.2f}")
    print(f"平均工人利润: {np.mean(final_worker_profits) if final_worker_profits else 0:.2f}")
    print(f"平均任务时间: {np.mean(final_task_times) if final_task_times else 0:.2f}")
    print(f"最终接受率: {all_accept_rates[-1] if all_accept_rates else 0:.2%}")
    print(f"总接受决策数: {worker_env.episode_accepts}")

    return {
        'task_profits': final_task_profits,
        'worker_profits': final_worker_profits,
        'task_times': final_task_times,
        'accept_rates': all_accept_rates
    }


if __name__ == "__main__":
    results = train_multitask_system(
        episodes=100,
        n_tasks_per_episode=80,
        max_workers=1000,
        n_workers_pool=100
    )