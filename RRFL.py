# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import random
import time
import math
from geopy.distance import geodesic
import tracemalloc
from scipy.spatial.distance import euclidean
import copy
import warnings
import gc

warnings.filterwarnings("ignore")

np.random.seed(54)
random.seed(54)

device = 'cpu'
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
        self.global_model = np.clip(np.random.randn(120).astype(np.float32) * 0.01, -0.5, 0.5)
        self.train_round = 0


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


class FLWorker:
    def __init__(self, worker_id, latitude, longitude, q_i, speed):
        self.worker_id = worker_id
        self.latitude = latitude
        self.longitude = longitude
        self.q_i = q_i
        self.speed = speed
        self.distance_to_task = 0.0
        self.local_model = np.clip(np.random.randn(120).astype(np.float32) * 0.01, -0.5, 0.5)
        self.sign = 0
        self.risk_score = np.clip(random.random() * 0.1, 0, 1)
        self.reputation = np.clip(np.random.normal(0.6, 0.1), 0, 1)
        self.is_removed = False
        self.total_reward = 0.0

class RRFL_FederatedEnv:
    def __init__(self, data_path):
        self.electricity_price = 0.5
        self.power_consumption = 0.2
        self.c_n = 0.1
        self.lambda_param = 50

        self.K = 5
        self.m = 3
        self.g = 3
        self.risk_threshold = 0.4
        self.rep_threshold = 0.4
        self.gamma = 0.7
        self.eta = 3.0
        self.b = 2.0
        self.T2 = 5.0

        chunk_size = 10000
        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=chunk_size,
                                 usecols=['id', 'latitude', 'longitude', 'quantity', 'volume', 'speed']):
            chunks.append(chunk)
        self.raw_data = pd.concat(chunks, ignore_index=True)

        if len(self.raw_data) > 300:
            self.raw_data = self.raw_data.sample(n=300, random_state=54)

        self.workers = self._preprocess_to_fl_worker()
        self.original_worker_states = {
            w.worker_id: (w.latitude, w.longitude) for w in self.workers
        }
        self.total_accepts = 0
        self.total_rejects = 0

    def _preprocess_to_fl_worker(self):
        data = self.raw_data.drop_duplicates(subset=['id', 'latitude', 'longitude'])
        unique_ids = data['id'].unique()

        if len(unique_ids) > 300:
            selected_ids = np.random.choice(unique_ids, 300, replace=False)
            data = data[data['id'].isin(selected_ids)]
        data = data[(data['quantity'] != 0) & (data['speed'] > 0)]
        data['q_i'] = 0.5 * data['quantity'] / 100 + 0.5 * data['volume'] / 100
        data['distance_to_task'] = 0.0

        fl_workers = []
        for _, row in data.iterrows():
            worker_id = int(row['id']) if str(row['id']).isdigit() else random.randint(1000, 9999)
            worker = FLWorker(
                worker_id=worker_id,
                latitude=row['latitude'],
                longitude=row['longitude'],
                q_i=row['q_i'],
                speed=row['speed']
            )
            fl_workers.append(worker)

        while len(fl_workers) < 300:
            fl_workers.append(FLWorker(
                worker_id=random.randint(10000, 99999),
                latitude=31.23 + random.uniform(-0.1, 0.1),
                longitude=121.48 + random.uniform(-0.1, 0.1),
                q_i=random.uniform(0.1, 1.0),
                speed=random.uniform(20, 60)
            ))
        return fl_workers

    def reset_worker(self, worker_id):
        if worker_id in self.original_worker_states:
            orig_lat, orig_lon = self.original_worker_states[worker_id]
            for w in self.workers:
                if w.worker_id == worker_id:
                    w.latitude = orig_lat
                    w.longitude = orig_lon
                    w.sign = 0
                    w.risk_score = 0.0
                    w.is_removed = False
                    break

    def get_available_workers(self):
        available = []
        for w in self.workers:
            if not w.is_removed and w.reputation >= self.rep_threshold:
                available.append(w)

        while len(available) < 10:
            new_worker = FLWorker(
                worker_id=random.randint(10000, 99999),
                latitude=31.23 + random.uniform(-0.1, 0.1),
                longitude=121.48 + random.uniform(-0.1, 0.1),
                q_i=random.uniform(0.1, 1.0),
                speed=random.uniform(20, 60)
            )
            self.workers.append(new_worker)
            available.append(new_worker)

        random.shuffle(available)
        return available[:80]

    def calculate_euclidean_dist(self, model1, model2):
        dist = euclidean(model1, model2) ** 2
        return np.clip(dist, 0, 100)

    def compute_risk_score(self, worker, train_round):
        worker.risk_score = np.clip(worker.sign / max(train_round, 1), 0, 1)
        self.risk_threshold = 0.3 + 0.1 * math.sin(train_round / self.K * math.pi)
        return worker.risk_score

    def update_reputation(self, worker):
        new_rep = (1 - self.gamma) * worker.reputation + self.gamma * (1 - worker.risk_score)
        worker.reputation = np.clip(new_rep, 0.0, 1.0)
        return worker.reputation

    def fl_train_step(self, task, workers):
        task.train_round += 1
        current_round = task.train_round
        model_updates = []

        for w in workers:
            if w.is_removed:
                continue
            if random.random() < 0.1:
                w.local_model = np.clip(w.local_model + np.random.randn(120).astype(np.float32) * 0.05, -0.5, 0.5)
                w.sign = min(w.sign + 1, 5)
            else:
                w.local_model = np.clip(w.local_model + np.random.randn(120).astype(np.float32) * 0.01, -0.5, 0.5)
            model_updates.append((w, w.local_model))

        dist_matrix = []
        global_model = task.global_model
        for w, m in model_updates:
            dist = self.calculate_euclidean_dist(m, global_model)
            dist_matrix.append((w, dist))

        select_num = min(self.g, len(dist_matrix))
        if select_num <= 0:
            select_num = 1
        dist_matrix.sort(key=lambda x: x[1])
        selected_updates = dist_matrix[:select_num]
        selected_workers = [w for w, _ in selected_updates]

        new_global_model = np.zeros(120, dtype=np.float32)
        for w, m in selected_updates:
            new_global_model += m / select_num
        task.global_model = np.clip(new_global_model, -0.5, 0.5)

        for w in workers:
            self.compute_risk_score(w, current_round)
            if w.risk_score > self.risk_threshold:
                w.is_removed = True
            self.update_reputation(w)

        return selected_workers, []

    def calculate_incentive_reward(self, worker):
        noise = np.random.normal(0, 0.05)
        reward = self.eta * math.exp(-self.b * worker.risk_score) + noise
        reward = np.clip(reward, 0.01, 10)
        worker.total_reward += reward
        return reward

    def assign_worker_to_task(self, task, worker):
        step_start = time.time()
        t_i = worker.distance_to_task / worker.speed
        c_d = worker.distance_to_task * self.power_consumption * self.electricity_price
        cost = c_d + self.c_n

        available_workers = self.get_available_workers()
        available_workers = [w for w in available_workers if w.worker_id != worker.worker_id]
        sample_num = min(1, len(available_workers))
        if sample_num > 0:
            sampled_workers = random.sample(available_workers, sample_num)
        else:
            sampled_workers = []
        train_workers = [worker] + sampled_workers

        selected_workers, _ = self.fl_train_step(task, train_workers)
        if worker in selected_workers:
            self.total_accepts += 1
            reward = self.calculate_incentive_reward(worker)
            worker_profit = reward - cost
            task.assigned_workers.append({
                'worker_id': worker.worker_id,
                't_i': t_i,
                'c_d': c_d,
                'q_i': worker.q_i,
                'reward': reward,
                'profit': worker_profit
            })
            task.sum_t_k += t_i
            task.total_payment += reward
            assigned = True
        else:
            self.total_rejects += 1
            worker_profit = -cost
            assigned = False

        step_time = time.time() - step_start
        return assigned, worker_profit, step_time


class RRFL_Platform:
    def __init__(self, fl_env, n_workers_pool=80):
        self.fl_env = fl_env
        self.n_workers_pool = n_workers_pool
        self.available_workers = None
        self.current_task = None

    def reset_for_task(self, task, all_workers):
        self.current_task = task
        self.available_workers = all_workers.copy()
        for w in self.available_workers:
            w.distance_to_task = geodesic(
                (w.latitude, w.longitude),
                task.location
            ).kilometers

    def select_worker(self):
        if not self.available_workers:
            return None
        self.available_workers.sort(key=lambda x: x.reputation, reverse=True)
        shuffle_num = min(3, len(self.available_workers))
        random.shuffle(self.available_workers[:shuffle_num])
        return self.available_workers[0]

    def calculate_task_profit(self):
        task = self.current_task
        if not task.assigned_workers:
            task.total_profit = 0.0
            return 0.0
        log_terms = np.log([1 + w['t_i'] * w['q_i'] for w in task.assigned_workers])
        sum_log = np.sum(log_terms)
        platform_income = self.fl_env.lambda_param * np.log(1 + sum_log)
        task.total_profit = platform_income - task.total_payment
        task.completed = True
        return task.total_profit

    def release_workers(self):
        for w_info in self.current_task.assigned_workers:
            worker_id = w_info['worker_id']
            self.fl_env.reset_worker(worker_id)


def train_rrfl_contrast_experiment(episodes=200, n_tasks_per_episode=100, n_workers_pool=80):
    tracemalloc.start()

    fl_env = RRFL_FederatedEnv('sh_csv/merged_shanghai_taxi_gps.csv')
    platform_env = RRFL_Platform(fl_env, n_workers_pool=n_workers_pool)
    task_generator = TaskGenerator(n_tasks_per_episode=n_tasks_per_episode)

    agg_task_profits = []
    agg_worker_profits = []
    agg_task_times = []
    agg_accept_rates = []
    recent_step_times = []
    episode_times = []

    print(f"\n===== 开始训练（保留CSV数据集） =====")
    print(f"工人总数: {len(fl_env.workers)} | 每轮任务数: {n_tasks_per_episode} | 总轮数: {episodes}")

    for episode in range(episodes):
        episode_start = time.time()
        all_workers = fl_env.get_available_workers()[:n_workers_pool]
        tasks = task_generator.generate_tasks()
        episode_profit = 0.0
        episode_worker_profit = []

        for task in tasks:
            platform_env.reset_for_task(task, all_workers)
            task_done = False
            step = 0

            while not task_done:
                selected_worker = platform_env.select_worker()
                if selected_worker is None:
                    break
                assigned, worker_profit, step_time = fl_env.assign_worker_to_task(task, selected_worker)

                recent_step_times.append(step_time)
                if len(recent_step_times) > 500:
                    recent_step_times.pop(0)

                step += 1

                if assigned:
                    episode_worker_profit.append(worker_profit)

                if len(task.assigned_workers) >= task.n_workers_needed or len(platform_env.available_workers) == 0:
                    task_done = True
                    task_profit = platform_env.calculate_task_profit()
                    episode_profit += task_profit
                    agg_task_profits.append(task_profit)
                    agg_task_times.append(task.sum_t_k)
                    platform_env.release_workers()

        if episode_worker_profit:
            agg_worker_profits.append(np.mean(episode_worker_profit))
        else:
            agg_worker_profits.append(0.0)

        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        total_workers = fl_env.total_accepts + fl_env.total_rejects
        accept_rate = fl_env.total_accepts / total_workers if total_workers > 0 else 0
        agg_accept_rates.append(accept_rate)

        if episode % 50 == 0:
            current, peak = tracemalloc.get_traced_memory()
            recent_profits = agg_task_profits[-n_tasks_per_episode:] if agg_task_profits else []
            profit_stats = calculate_statistics(recent_profits)
            time_stats = calculate_statistics(recent_step_times)

            print(f"\n===== Episode: {episode}/{episodes} =====")
            print(f"回合总利润: {episode_profit:.2f}, 近期单任务利润: 均值={profit_stats['mean']}, 方差={profit_stats['var']}")
            print(
                f"平均单步耗时: 均值={time_stats['mean']:.6f}s, 方差={time_stats['var']:.8f}, 95%CI=[{time_stats['ci95_lower']:.6f}, {time_stats['ci95_upper']:.6f}]")
            print(f"平均工人利润: {np.mean(episode_worker_profit):.2f}, 接受率: {accept_rate:.2%}")
            print(f"内存使用: {current / 1e6:.2f}MB (峰值: {peak / 1e6:.2f}MB)")
            print(f"工人接受/拒绝: {fl_env.total_accepts}/{fl_env.total_rejects}")

            gc.collect()
            del recent_profits, profit_stats, time_stats

    print("\n" + "=" * 50)
    print("RRFL联邦学习 对比试验结果 - 完整统计分析（保留CSV）")
    print("=" * 50)

    profit_stats = calculate_statistics(agg_task_profits)
    worker_profit_stats = calculate_statistics(agg_worker_profits)
    task_time_stats = calculate_statistics(agg_task_times)
    step_time_stats = calculate_statistics(recent_step_times)
    accept_rate_stats = calculate_statistics(agg_accept_rates)

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
    print(f"总接受任务数: {fl_env.total_accepts}")
    print(f"总训练回合数: {episodes}")
    print(f"总处理任务数: {len(agg_task_profits)}")
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
    import os
    if not os.path.exists('statistics'):
        os.makedirs('statistics')
    stats_df.to_csv('statistics/contrast_rrfl.csv', index=False, encoding='utf-8-sig')
    print("\nRRFL联邦学习对比试验统计结果已保存到 statistics/contrast_rrfl.csv")

    return {
        'task_profits': agg_task_profits,
        'worker_profits': agg_worker_profits,
        'task_times': agg_task_times,
        'step_times': recent_step_times,
        'accept_rates': agg_accept_rates,
        'profit_stats': profit_stats,
        'worker_profit_stats': worker_profit_stats,
        'task_time_stats': task_time_stats,
        'step_time_stats': step_time_stats,
        'accept_rate_stats': accept_rate_stats
    }


if __name__ == "__main__":
    results = train_rrfl_contrast_experiment(episodes=200, n_tasks_per_episode=100)