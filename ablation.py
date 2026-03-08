# -*- coding: UTF-8 -*-
# @Author:zhuxiao ASUS
import pandas as pd
import numpy as np
from geopy.distance import geodesic

task_latitude = 31.23
task_longitude = 121.48

data = pd.read_csv('sh_csv/shanghai_taxi_gps.csv')
data = data[(data['latitude'] > 31.21) &
            (data['latitude'] < 31.26) &
            (data['longitude'] > 121.46) &
            (data['longitude'] < 121.52)]

data = data[data['quantity'] != 0]
data = data[data['speed'] > 0]


data['q_i'] = data['quantity']/100

electricity_price = 0.5
power_consumption = 0.2
c_n = 0.1
c_b = 21
b_j = c_b
lambda_param = 50

data['distance_to_task'] = data.apply(lambda row: geodesic(
    (row['latitude'], row['longitude']), (task_latitude, task_longitude)).kilometers, axis=1)

data_sorted_by_qi = data.sort_values('q_i', ascending=False)

unique_workers = data_sorted_by_qi.drop_duplicates(subset='id')

top_100_unique_workers = unique_workers.head(100)

nearest_workers = top_100_unique_workers.nlargest(5, 'distance_to_task')

nearest_workers['c_d'] = nearest_workers['distance_to_task'] * power_consumption * electricity_price

nearest_workers['t_i'] = nearest_workers['distance_to_task'] / nearest_workers['speed']

sum_t_k = nearest_workers['t_i'].sum()

nearest_workers['reward_share'] = nearest_workers['q_i'] * b_j * (nearest_workers['t_i'] / sum_t_k)

R = nearest_workers['reward_share'].sum()

log_sum = np.sum(np.log(1 + nearest_workers['t_i'] * nearest_workers['q_i']))
U_p = lambda_param * np.log(1 + log_sum) - R

nearest_workers['U_i'] = nearest_workers['reward_share'] - nearest_workers['c_d'] - c_n

print("平台利润 U_p:", U_p)
print("Q_i排名前100中距离最近的5个工人利润 U_i:")
print(nearest_workers[['id', 'q_i', 'distance_to_task', 'reward_share', 'c_d', 't_i', 'U_i']])
