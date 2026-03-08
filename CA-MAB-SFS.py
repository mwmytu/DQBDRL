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

np.random.seed(54)
random.seed(54)
torch.manual_seed(54)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


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
            lat = self.base_location[0] + random.uniform(-0.19, 0.19)
            lon = self.base_location[1] + random.uniform(-0.19, 0.19)
            tasks.append(
                Task(
                    task_id=i,
                    location=(lat, lon),
                    n_workers_needed=5
                )
            )
        return tasks


class MABWorker:
    def __init__(self, worker_id, n_task_types=10, free_sensing_threshold=15):
        self.worker_id = worker_id
        self.n_task_types = n_task_types
        self.free_sensing_threshold = free_sensing_threshold

        self.utility_estimates = np.zeros(n_task_types)
        self.effort_estimates = np.zeros(n_task_types)
        self.task_counts = np.zeros(n_task_types)
        self.rejection_counts = np.zeros(n_task_types)
        self.last_choice = 0

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.lambda_param = 0.15

        self.bargaining_power = np.random.uniform(1.2, 1.8)

    def get_exploration_rate(self, t):
        return max(self.epsilon_min, self.epsilon * (self.epsilon_decay ** t))

    def calculate_payment_proposal(self, task_type, base_cost):

        if self.effort_estimates[task_type] > 0:

            return self.bargaining_power * self.effort_estimates[task_type]
        return base_cost * self.bargaining_power

    def decide(self, available_tasks, current_time, last_payments):

        if current_time > 1 and np.random.random() < self.lambda_param:
            if self.last_choice in available_tasks:
                payment = self.calculate_payment_proposal(self.last_choice, 1.5)
                return self.last_choice, payment, False

        free_sensing = False
        plausible_tasks = []

        for task_type in available_tasks:
            if self.rejection_counts[task_type] > self.free_sensing_threshold:
                free_sensing = True
                plausible_tasks.append(task_type)
                break

        if not free_sensing:

            for task_type in available_tasks:
                payment = self.calculate_payment_proposal(task_type, 1.5)

                if payment <= last_payments.get(task_type, float('inf')) * 1.2 or last_payments.get(task_type, 0) == 0:
                    plausible_tasks.append(task_type)

        if not plausible_tasks:
            task_type = np.random.choice(available_tasks)
            payment = self.calculate_payment_proposal(task_type, 1.5)
            self.last_choice = task_type
            return task_type, payment, False

        epsilon = self.get_exploration_rate(current_time)

        if np.random.random() < epsilon:
            task_type = np.random.choice(plausible_tasks)
        else:
            utilities = [self.utility_estimates[t] for t in plausible_tasks]
            task_type = plausible_tasks[np.argmax(utilities)]

        free_sensing_prob = min(0.3, self.rejection_counts[task_type] / 50)
        free_sensing = free_sensing or (np.random.random() < free_sensing_prob)

        payment = 0.0 if free_sensing else self.calculate_payment_proposal(task_type, 1.5)
        self.last_choice = task_type

        return task_type, payment, free_sensing

    def update_knowledge(self, task_type, accepted, actual_utility, actual_effort):

        if accepted:

            if self.task_counts[task_type] == 0:
                self.utility_estimates[task_type] = actual_utility
                self.effort_estimates[task_type] = actual_effort
            else:
                self.utility_estimates[task_type] += (actual_utility - self.utility_estimates[task_type]) / (
                            self.task_counts[task_type] + 1)
                self.effort_estimates[task_type] += (actual_effort - self.effort_estimates[task_type]) / (
                            self.task_counts[task_type] + 1)

            self.task_counts[task_type] += 1
            self.rejection_counts[task_type] = max(0, self.rejection_counts[task_type] - 2)
        else:
            self.rejection_counts[task_type] += 1


class MABPlatformEnvironment:
    def __init__(self, data_path, n_workers_pool=100, n_task_types=10):

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

        self.n_task_types = n_task_types
        self.workers = {}
        for worker_id in self.data['id'].unique():
            self.workers[worker_id] = MABWorker(worker_id, n_task_types)

        self.n_workers_pool = n_workers_pool
        self.available_workers = None
        self.current_task = None
        self.last_payments = np.zeros(n_task_types)

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

    def reset_for_task(self, task, all_workers):

        self.current_task = task
        self.available_workers = all_workers.copy()

        self.available_workers['distance_to_task'] = self.available_workers.apply(
            lambda row: geodesic(
                (row['latitude'], row['longitude']),
                task.location
            ).kilometers, axis=1
        )

        task_type = self._get_task_type(task.location)
        self.current_task_type = task_type

    def _get_task_type(self, location):

        lat, lon = location

        lat_zone = int((lat - 31.23) / 0.04)
        lon_zone = int((lon - 121.48) / 0.04)
        task_type = (lat_zone * 5 + lon_zone) % self.n_task_types
        return max(0, min(self.n_task_types - 1, task_type))

    def calculate_worker_basics(self, worker):

        t_i = worker['distance_to_task'] / max(worker['speed'], 1)

        c_d = worker['distance_to_task'] * self.power_consumption * self.electricity_price * 1.5
        return t_i, c_d

    def platform_strategy(self, current_time):

        if self.available_workers.empty:
            return None, 0.0, False

        worker_offers = []
        for idx, worker in self.available_workers.iterrows():
            worker_id = worker['id']
            mab_worker = self.workers[worker_id]

            task_type, payment, free_sensing = mab_worker.decide(
                [self.current_task_type], current_time,
                {self.current_task_type: self.last_payments[self.current_task_type]}
            )

            worker_offers.append({
                'index': idx,
                'worker_id': worker_id,
                'worker_data': worker,
                'payment': payment,
                'free_sensing': free_sensing
            })

        if worker_offers:
            def worker_score(offer):
                worker_data = offer['worker_data']
                payment = offer['payment']
                quality = worker_data['q_i']
                distance = worker_data['distance_to_task']
                return quality * 0.6 - payment * 0.3 - distance * 0.1

            selected_offer = max(worker_offers, key=worker_score)
            selected_worker = selected_offer['worker_data']

            t_i, c_d = self.calculate_worker_basics(selected_worker)

            self.total_accepts += 1

            self.current_task.assigned_workers.append({
                'worker_id': selected_worker['id'],
                't_i': t_i,
                'c_d': c_d,
                'q_i': selected_worker['q_i'],
                'payment': selected_offer['payment'],
                'free_sensing': selected_offer['free_sensing']
            })
            self.current_task.sum_t_k += t_i

            actual_utility = selected_offer['payment'] - c_d - self.c_n
            actual_effort = c_d + self.c_n
            self.workers[selected_worker['id']].update_knowledge(
                self.current_task_type, True, actual_utility, actual_effort
            )

            self.available_workers = self.available_workers.drop(selected_offer['index'])

            if selected_offer['payment'] > 0:
                self.last_payments[self.current_task_type] = selected_offer['payment']

            return selected_worker, 0.0, True

        return None, -0.1, False

    def calculate_task_rewards(self):
        task = self.current_task
        if not task.assigned_workers:
            task.total_profit = 0.0
            return

        total_reward = 0.0
        total_actual_payment = 0.0

        for worker in task.assigned_workers:
            if task.sum_t_k == 0:
                share = 0.0
            else:
                quality_bonus = worker['q_i'] * 3.0
                time_share = worker['t_i'] / task.sum_t_k
                share = (
                        (min(worker['q_i'], 0.8) + quality_bonus) *
                        task.b_j *
                        time_share
                )

            actual_payment = min(share, worker.get('payment', share))
            worker['reward_share'] = actual_payment
            worker['final_profit'] = actual_payment - worker['c_d'] - self.c_n
            total_reward += actual_payment
            total_actual_payment += actual_payment

        if task.assigned_workers:
            log_terms = np.log([1 + w['t_i'] * w['q_i'] for w in task.assigned_workers])
            sum_log = np.sum(log_terms)
            platform_utility = self.lambda_param * np.log(1 + sum_log)
            operational_cost = total_actual_payment * 2.25
            task.total_profit = platform_utility - operational_cost
        else:
            task.total_profit = 0.0

        task.total_payment = total_actual_payment
        task.completed = True

    def release_workers(self):
        for worker in self.current_task.assigned_workers:
            self.reset_worker(worker['worker_id'])


def train_mab_agent(episodes=1000, n_tasks_per_episode=100, n_workers_pool=100):
    platform_env = MABPlatformEnvironment('sh_csv/merged_shanghai_taxi_gps.csv', n_workers_pool)
    task_generator = TaskGenerator(n_tasks_per_episode=n_tasks_per_episode)

    all_task_profits = []
    all_worker_profits = []
    all_task_times = []
    all_accept_rates = []
    free_sensing_counts = []

    for episode in range(episodes):
        all_workers = platform_env.data.drop_duplicates(subset='id').head(n_workers_pool)
        tasks = task_generator.generate_tasks()
        episode_profit = 0.0
        episode_worker_profit = []
        episode_free_sensing = 0

        for task in tasks:
            platform_env.reset_for_task(task, all_workers)
            task_done = False

            while not task_done:
                selected_worker, reward, accepted = platform_env.platform_strategy(episode + 1)

                if accepted and platform_env.current_task.assigned_workers:
                    latest_worker = platform_env.current_task.assigned_workers[-1]
                    if latest_worker.get('free_sensing', False):
                        episode_free_sensing += 1

                if (len(task.assigned_workers) >= task.n_workers_needed
                        or len(platform_env.available_workers) == 0):
                    task_done = True
                    platform_env.calculate_task_rewards()
                    final_reward = task.total_profit
                    episode_profit += final_reward
                    all_task_profits.append(task.total_profit)
                    all_task_times.append(task.sum_t_k)

                    if task.assigned_workers:
                        avg_worker_profit = np.mean([w['final_profit'] for w in task.assigned_workers])
                        episode_worker_profit.append(avg_worker_profit)
                        all_worker_profits.append(avg_worker_profit)

                    platform_env.release_workers()

        total_workers = platform_env.total_accepts + platform_env.total_rejects
        accept_rate = platform_env.total_accepts / total_workers if total_workers > 0 else 0
        all_accept_rates.append(accept_rate)
        free_sensing_counts.append(episode_free_sensing)

        if episode % 50 == 0:
            recent_profits = all_task_profits[-len(tasks):] if len(all_task_profits) >= len(tasks) else all_task_profits
            print(f"\nEpisode: {episode}/{episodes}")
            print(f"回合总利润: {episode_profit:.2f}, 平均单任务利润: {np.mean(recent_profits):.2f}")
            print(
                f"平均工人利润: {np.mean(episode_worker_profit) if episode_worker_profit else 0:.2f}, 接受率: {accept_rate:.2%}")
            print(f"免费感知次数: {episode_free_sensing}")
            print(f"工人接受/拒绝: {platform_env.total_accepts}/{platform_env.total_rejects}")

    print("\n===== MAB训练结果 =====")
    print(f"平均平台利润: {np.mean(all_task_profits):.2f}")
    print(f"平均工人利润: {np.mean(all_worker_profits):.2f}")
    print(f"平均任务时间: {np.mean(all_task_times):.2f}")
    print(f"最终接受率: {all_accept_rates[-1]:.2%}")
    print(f"总接受任务数: {platform_env.total_accepts}")
    print(f"平均免费感知次数: {np.mean(free_sensing_counts):.2f}")

    return {
        'task_profits': all_task_profits,
        'worker_profits': all_worker_profits,
        'task_times': all_task_times,
        'accept_rates': all_accept_rates,
        'free_sensing_counts': free_sensing_counts
    }


if __name__ == "__main__":
    results = train_mab_agent(episodes=1000, n_tasks_per_episode=100)