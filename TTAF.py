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


class CooperativeUnit:

    def __init__(self, cu_id, workers):
        self.cu_id = cu_id
        self.workers = workers
        self.proxy = None
        self.assigned_tasks = []
        self.expertise_guarantee = {}
        self.worker_profits = {}
        self.available_workers = workers.copy()

    def add_task(self, task, expertise_guarantee):
        self.assigned_tasks.append(task)
        self.expertise_guarantee[task.task_id] = expertise_guarantee

    def get_available_workers(self, task):
        available = []
        for worker in self.available_workers:
            if self._check_expertise(worker, task):
                available.append(worker)
        return available

    def assign_worker_to_task(self, worker, task):
        if worker in self.available_workers:
            self.available_workers.remove(worker)
            return True
        return False

    def reset_workers(self):
        self.available_workers = self.workers.copy()
        self.assigned_tasks = []
        self.expertise_guarantee = {}

    def _check_expertise(self, worker, task):
        worker_expertise = worker.get('q_i', 1.0)
        required_expertise = task.expected_expertise / 6.0
        return worker_expertise >= required_expertise * 0.8

    def record_worker_profit(self, worker_id, profit):
        if worker_id not in self.worker_profits:
            self.worker_profits[worker_id] = []
        self.worker_profits[worker_id].append(profit)


class Task:
    def __init__(self, task_id, location, b_j=21, n_workers_needed=5, keywords=None):
        self.task_id = task_id
        self.location = location
        self.b_j = b_j
        self.n_workers_needed = n_workers_needed
        self.keywords = keywords or ["default"]
        self.assigned_CUs = []
        self.assigned_workers = []
        self.sum_t_k = 0.0
        self.total_profit = 0.0
        self.total_payment = 0.0
        self.completed = False
        self.expected_expertise = 3
        self.start_time = 0
        self.end_time = 0


class TwoTierAssignmentFramework:

    def __init__(self, CUs, Y_U=2, Y_T=5, budget=1000):
        self.CUs = CUs
        self.Y_U = Y_U
        self.Y_T = Y_T
        self.budget = budget
        self.remaining_budget = budget

    def higher_tier_assignment(self, tasks):
        task_CU_assignments = {}

        for task in tasks:
            suitable_CUs = self._select_suitable_CUs(task)
            task_CU_assignments[task.task_id] = suitable_CUs[:self.Y_U]

            for cu in suitable_CUs[:self.Y_U]:
                avg_quality = np.mean([w.get('q_i', 1.0) for w in cu.workers])
                expertise_guarantee = max(1, min(6, int(avg_quality * 6)))
                cu.add_task(task, expertise_guarantee)
                task.assigned_CUs.append(cu)

        return task_CU_assignments

    def _select_suitable_CUs(self, task):
        scored_CUs = []
        for cu in self.CUs:
            if len(cu.assigned_tasks) < self.Y_T:
                avg_quality = np.mean([w.get('q_i', 1.0) for w in cu.workers])
                available_workers = len(cu.get_available_workers(task))
                score = avg_quality * 0.7 + (available_workers / len(cu.workers)) * 0.1
                random_factor = np.random.uniform(0.9, 1.1)
                score *= random_factor
                scored_CUs.append((score, cu))

        scored_CUs.sort(key=lambda x: x[0], reverse=True)
        return [cu for _, cu in scored_CUs]

    def lower_tier_assignment(self, task_CU_assignments):
        all_worker_assignments = {}

        for task_id, CUs in task_CU_assignments.items():
            for cu in CUs:
                task = next((t for t in cu.assigned_tasks if t.task_id == task_id), None)
                if not task:
                    continue

                available_workers = cu.get_available_workers(task)

                assigned_workers = []
                for worker in available_workers[:task.n_workers_needed]:
                    if cu.assign_worker_to_task(worker, task):
                        assigned_workers.append(worker)

                if assigned_workers:
                    all_worker_assignments[(task_id, cu.cu_id)] = {
                        worker['id']: task for worker in assigned_workers
                    }

        return all_worker_assignments

    def calculate_profits(self, task, worker_assignments, worker_env):
        platform_profit = 0
        worker_profits = []
        task_times = []

        task.start_time = np.random.uniform(0, 24)

        total_quality = 0
        total_worker_payment = 0
        worker_count = 0

        for (task_id, cu_id), assignments in worker_assignments.items():
            if task_id != task.task_id:
                continue

            cu = next((c for c in self.CUs if c.cu_id == cu_id), None)
            if not cu:
                continue

            for worker_id, assigned_task in assignments.items():
                if assigned_task.task_id != task.task_id:
                    continue

                worker_data = next((w for w in cu.workers if w['id'] == worker_id), None)
                if not worker_data:
                    continue

                distance = geodesic(
                    (worker_data['latitude'], worker_data['longitude']),
                    task.location
                ).kilometers * np.random.uniform(0.8, 1.2)

                travel_time = distance / worker_data['speed']
                work_time = np.random.uniform(2.0, 5.0)
                total_time = travel_time + work_time
                task_times.append(total_time)

                travel_cost = distance * worker_env.power_consumption * worker_env.electricity_price
                network_cost = worker_env.c_n
                total_cost = travel_cost + network_cost

                quality_factor = worker_data['q_i']
                time_factor = total_time / (sum(task_times) if task_times else 1)
                reward_random_factor = np.random.uniform(0.9, 1.1)
                worker_reward = task.b_j * quality_factor * time_factor * 0.63 * reward_random_factor

                worker_profit = worker_reward * 0.6 - total_cost
                worker_profits.append(worker_profit)
                total_worker_payment += worker_reward
                total_quality += quality_factor
                worker_count += 1

                cu.record_worker_profit(worker_id, worker_profit)

        if worker_profits:
            platform_random_factor = np.random.uniform(0.8, 1.2)
            avg_quality = total_quality / worker_count if worker_count > 0 else 0

            platform_profit = (worker_env.lambda_param * np.log(1 + avg_quality) -
                               total_worker_payment)
        else:
            platform_profit = 0

        task.end_time = task.start_time + (max(task_times) if task_times else 0)
        total_assigned_workers = sum(
            len(assignments) for (tid, _), assignments in worker_assignments.items() if tid == task.task_id)
        task.completed = total_assigned_workers >= task.n_workers_needed
        task.total_profit = platform_profit
        task.sum_t_k = sum(task_times) if task_times else 0

        return platform_profit, worker_profits, task_times


class TaskGenerator:
    def __init__(self, n_tasks_per_episode=100):
        self.n_tasks_per_episode = n_tasks_per_episode
        self.base_location = (22.54, 114.05)

    def generate_tasks(self):
        tasks = []
        keyword_options = [["traffic", "monitoring"], ["environment", "data"],
                           ["image", "processing"], ["survey", "collection"]]

        for i in range(self.n_tasks_per_episode):
            lat = self.base_location[0] + random.uniform(-0.1, 0.1)
            lon = self.base_location[1] + random.uniform(-0.1, 0.1)

            keywords = random.choice(keyword_options)

            budget_variation = np.random.uniform(0.8, 1.2)
            adjusted_budget = 21 * budget_variation

            tasks.append(
                Task(
                    task_id=i,
                    location=(lat, lon),
                    n_workers_needed=5,
                    keywords=keywords,
                    b_j=adjusted_budget
                )
            )
        return tasks


class SimpleWorkerEnvironment:
    def __init__(self, data_path, n_workers=1000):
        self.electricity_price = 0.5
        self.power_consumption = 0.2
        self.c_n = 0.1
        self.lambda_param = 50

        self.data = pd.read_csv(data_path)

        if len(self.data) > n_workers:
            self.data = self.data.sample(n=n_workers, random_state=54)
        else:
            self.data = self.data.copy()

        if 'latitude' not in self.data.columns:
            self.data['latitude'] = np.random.uniform(22.49, 22.54, len(self.data))
        if 'longitude' not in self.data.columns:
            self.data['longitude'] = np.random.uniform(114.05, 114.00, len(self.data))

        if 'id' not in self.data.columns:
            self.data['id'] = range(len(self.data))

        self.data['q_i'] = np.random.uniform(0.3, 1.0, len(self.data))
        self.data['speed'] = np.random.uniform(20, 60, len(self.data))

        self.worker_dicts = self.data.to_dict('records')

        print(f"从数据集中随机选择了 {len(self.worker_dicts)} 个工人")


class SimplePlatformEnvironment:
    def __init__(self, worker_env, CUs):
        self.worker_env = worker_env
        self.CUs = CUs
        self.ttaf = TwoTierAssignmentFramework(CUs, Y_U=2, Y_T=5, budget=1000)

    def process_episode(self, tasks):
        for cu in self.CUs:
            cu.reset_workers()

        task_CU_assignments = self.ttaf.higher_tier_assignment(tasks)

        worker_assignments = self.ttaf.lower_tier_assignment(task_CU_assignments)

        total_platform_profit = 0
        all_worker_profits = []
        all_task_times = []
        completed_tasks = 0

        for task in tasks:
            if task.assigned_CUs:
                platform_profit, worker_profits, task_times = self.ttaf.calculate_profits(
                    task, worker_assignments, self.worker_env
                )

                total_platform_profit += platform_profit
                all_worker_profits.extend(worker_profits)
                all_task_times.extend(task_times)

                if task.completed:
                    completed_tasks += 1

        return total_platform_profit, all_worker_profits, all_task_times, completed_tasks


def create_CUs(worker_data, n_CUs=5, workers_per_CU=60):
    CUs = []

    total_workers_needed = n_CUs * workers_per_CU
    if len(worker_data) < total_workers_needed:
        print(f"工人数量不足，从{len(worker_data)}个工人中创建协作单元")
        for i in range(n_CUs):
            workers = random.choices(worker_data, k=workers_per_CU)
            cu = CooperativeUnit(cu_id=i, workers=workers)
            CUs.append(cu)
    else:
        available_workers = worker_data.copy()
        random.shuffle(available_workers)

        for i in range(n_CUs):
            workers = available_workers[i * workers_per_CU:(i + 1) * workers_per_CU]
            cu = CooperativeUnit(cu_id=i, workers=workers)
            CUs.append(cu)

    print(f"创建了 {len(CUs)} 个协作单元，共 {sum(len(cu.workers) for cu in CUs)} 个工人")
    return CUs


def train_simplified_system(episodes=1000, n_tasks_per_episode=100, n_workers=80):
    tracemalloc.start()

    worker_env = SimpleWorkerEnvironment('electricVehicle_csv/merged_electricVehicle_gps.csv', n_workers=n_workers)
    task_generator = TaskGenerator(n_tasks_per_episode=n_tasks_per_episode)

    CUs = create_CUs(worker_env.worker_dicts, n_CUs=5, workers_per_CU=60)
    platform_env = SimplePlatformEnvironment(worker_env, CUs)

    platform_profits = []
    worker_profits = []
    task_times = []
    completion_rates = []

    for episode in range(episodes):
        tasks = task_generator.generate_tasks()

        total_platform_profit, episode_worker_profits, episode_task_times, completed_tasks = platform_env.process_episode(
            tasks)

        platform_profits.append(total_platform_profit)
        worker_profits.extend(episode_worker_profits)
        task_times.extend(episode_task_times)

        completion_rate = completed_tasks / len(tasks) if tasks else 0
        completion_rates.append(completion_rate)

        if episode % 50 == 0:
            current, peak = tracemalloc.get_traced_memory()
            avg_worker_profit = np.mean(episode_worker_profits) if episode_worker_profits else 0
            avg_task_time = np.mean(episode_task_times) if episode_task_times else 0
            current_avg_platform = np.mean(platform_profits) if platform_profits else 0

            print(f"\nEpisode: {episode}/{episodes}")
            print(f"回合总利润: {total_platform_profit:.2f}")
            print(f"当前平均平台利润: {current_avg_platform:.2f}")
            print(f"平均工人利润: {avg_worker_profit:.2f}")
            print(f"平均任务时间: {avg_task_time:.2f}小时")
            print(f"完成率: {completion_rate:.2%}")
            print(f"内存使用: {current / 1e6:.2f}MB (峰值: {peak / 1e6:.2f}MB)")

    final_avg_platform = np.mean(platform_profits) if platform_profits else 0
    final_avg_worker = np.mean(worker_profits) if worker_profits else 0
    final_avg_time = np.mean(task_times) if task_times else 0
    final_avg_completion = np.mean(completion_rates) if completion_rates else 0

    print("\n===== TTAF 训练结果 =====")
    print(f"平均平台利润: {final_avg_platform:.2f}")
    print(f"平均工人利润: {final_avg_worker:.2f}")
    print(f"平均任务时间: {final_avg_time:.2f}小时")
    print(f"平均完成率: {final_avg_completion:.2%}")
    print(f"总回合数: {episodes}")

    tracemalloc.stop()

    return {
        'platform_profits': platform_profits,
        'worker_profits': worker_profits,
        'task_times': task_times,
        'completion_rates': completion_rates
    }


if __name__ == "__main__":
    task_configs = [100]

    for n_tasks in task_configs:
        print(f"\n{'=' * 60}")
        print(f"测试配置: 任务数量={n_tasks}, 工人数量=60")
        print(f"{'=' * 60}")

        results = train_simplified_system(
            episodes=1000,
            n_tasks_per_episode=n_tasks,
            n_workers=80
        )

