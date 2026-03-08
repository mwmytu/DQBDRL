# -*- coding: UTF-8 -*-
# @Author:zhuxiao ASUS
# -*- coding: UTF-8 -*-
# @Author:zhuxiao ASUS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.weight'] = 'bold'

'''profit1=pd.read_csv('electricVehicle_csv/BDMTA_W.csv')
profit2=pd.read_csv('electricVehicle_csv/OPTA_W.csv')
profit3=pd.read_csv('electricVehicle_csv/MAB_W.csv')
profit4=pd.read_csv('electricVehicle_csv/TTAF_W.csv')'''

'''plt.plot(profit1['count'],profit1['quality'],marker='o',color='purple',label='CUTA')
plt.plot(profit2['count'],profit2['quality'],marker='o',color='pink',label='TSMA')
plt.plot(profit3['count'],profit3['quality'],marker='o',color='cyan',label='BMA')
plt.plot(profit4['count'],profit4['quality'],marker='o',color='orange',label='QSTA')
plt.xlabel('|T|')
plt.ylabel('Quality')
plt.title('The influence of task count on quality')
plt.legend()
plt.savefig('bj_csv_data/The influence of task count on quality_bj.pdf',format='pdf')

plt.show()'''

'''bar_width=0.15

index=np.arange(len(profit1['count']))

plt.bar(index,profit1['Uw'],bar_width,color='red',label='DQBDRL')
plt.bar(index+bar_width,profit2['Uw'],bar_width,color='#32CD32',label='OPTA')
plt.bar(index+2*(bar_width),profit3['Uw'],bar_width,color='blue',label='CA-MAB-SFS')
plt.bar(index+3*(bar_width),profit4['Uw'],bar_width,color='purple',label='TTAF')

plt.xlabel('|W|', fontsize=18, fontweight='bold')
plt.ylabel('The utility of workers', fontsize=18, fontweight='bold')
# The utility of workers The completion time of tasks The utility of platform
plt.xticks(index+bar_width,profit1['count'])
plt.legend(fontsize=16, loc='upper right')
plt.tight_layout()
# 增大坐标轴刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('sh_csv/The influence of W on the utility of workers_sh_bar.pdf',format='pdf')

plt.show()'''

'''# bj是56 sz是24 sh是13
profit1=pd.read_csv('sh_csv/qi_W.csv')
profit2=pd.read_csv('sh_csv/BDMTA_W.csv')

bar_width=0.3

index=np.arange(len(profit1['count']))

plt.bar(index,profit1['Uw'],bar_width,color='orange',label='DQBDRL_X')
plt.bar(index+bar_width,profit2['Uw'],bar_width,color='#32CD32',label='DQBDRL')
plt.xlabel('|T|', fontsize=18, fontweight='bold')
plt.ylabel('The utility of workers', fontsize=18, fontweight='bold')
# The utility of workers The completion time of tasks The utility of platform
plt.xticks(index+bar_width,profit1['count'])
plt.legend(fontsize=16, loc='upper left')
plt.tight_layout()
# 增大坐标轴刻度字体大小 , loc='lower center'
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('sh_csv/X_The influence of T on the utility of workers_sh.pdf',format='pdf')

plt.show()'''

'''# 56是sh 43sh 21sz
profit1=pd.read_csv('sh_csv/bi_W.csv')
profit2=pd.read_csv('sh_csv/BDMTA_W.csv')

bar_width=0.3

index=np.arange(len(profit1['count']))

plt.bar(index,profit1['Uw'],bar_width,color='red',label='DQBDRL_Y')
plt.bar(index+bar_width,profit2['Uw'],bar_width,color='green',label='DQBDRL')
plt.xlabel('|T|', fontsize=18, fontweight='bold')
plt.ylabel('The utility of workers', fontsize=18, fontweight='bold')
# The utility of workers The completion time of tasks The utility of platform
plt.xticks(index+bar_width,profit1['count'])
plt.legend(fontsize=16, loc='upper left')
plt.tight_layout()
# 增大坐标轴刻度字体大小 , loc='lower center'
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('sh_csv/Y_The influence of T on the utility of workers_sh.pdf',format='pdf')

plt.show()'''

# 56是sh 43sh 21sz
profit1=pd.read_csv('statistics/IMARL_W.csv')
profit2=pd.read_csv('statistics/DQBDRL_W.csv')

bar_width=0.3

index=np.arange(len(profit1['count']))

plt.bar(index,profit1['B'],bar_width,color='#edb120',label='IMARL')
plt.bar(index+bar_width,profit2['B'],bar_width,color='red',label='DQBDRL')
plt.xlabel('|W|', fontsize=18, fontweight='bold')
plt.ylabel('The coefficient of variation', fontsize=18, fontweight='bold')
# The utility of workers The completion time of tasks The utility of platform
plt.xticks(index+bar_width,profit1['count'])
plt.legend(fontsize=16, loc='upper left')
plt.tight_layout()
# 增大坐标轴刻度字体大小 , loc='lower center'
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('statistics/The influence of W on the coefficient of variation_sh.pdf',format='pdf')

plt.show()

'''plt.plot(profit1['count'],profit1['Uw'],marker='o',color='red',label='DQBDRL')
plt.plot(profit2['count'],profit2['Uw'],marker='^',color='#32CD32',label='OPTA')
plt.plot(profit3['count'],profit3['Uw'],marker='x',color='blue',label='CA-MAB-SFS')
plt.plot(profit4['count'],profit4['Uw'],marker='s',color='purple',label='TTAF')
plt.xlabel('|T|', fontsize=18)
plt.ylabel('The utility of workers', fontsize=18)
plt.legend(fontsize=16)
# The utility of workers The completion time of tasks The utility of platform
# 增大坐标轴刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(True)  # 添加网格线
plt.savefig('electricVehicle_csv/The influence of T on the utility of workers_ele_line.pdf',format='pdf')

plt.show()'''