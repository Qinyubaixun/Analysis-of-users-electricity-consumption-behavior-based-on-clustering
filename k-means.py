import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
import sqlite3
from datetime import datetime
from datetime import date
from sklearn.cluster import KMeans
from pylab import mpl
from matplotlib.dates import AutoDateLocator, DateFormatter
autodates = AutoDateLocator()
yearsFmt = DateFormatter('%Y-%m-%d %H:%M:%S')

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
cwd = os.getcwd()  #获得当前的文件路径
conn = sqlite3.connect(str(cwd) + "/dataport_sqlite")
cursor = conn.cursor()  #构建指针
query = "SELECT * FROM new_table;"
cursor.execute(query)    #执行读取数据命令
data = cursor.fetchall() #抓取数据

loads_df = pd.DataFrame(data, columns=['id','date','energy_use'])
loads_df.head()
time = loads_df.date[0:95]
x_time= [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in time]  # 将时间字符串转换为时间格式
print(x_time)
print("当前数据集含有%s行,%s列"%(loads_df.shape[0],loads_df.shape[1]))
print(" 最早时间: %s \n 最晚时间: %s"%(loads_df.date.min(),loads_df.date.max()))

loads_df['id'].value_counts()

loads_df = loads_df.replace('',np.nan)

loads_df = loads_df.dropna()

loads_df.loc[:,'energy_use'] = loads_df.energy_use.astype(float)

loads_df.loc[:,'id'] = loads_df['id'].astype(int)

loads_df.loc[:,'date'] = pd.to_datetime(loads_df.date)

# 添加一代表星期的列，isoweekday会根据日期判定是周几
loads_df.loc[:,'type_day'] = loads_df.date.apply(lambda x: x.isoweekday())

# 添加一代表日期的列，day会根据具体日期判定是几号
loads_df.loc[:,'day_of_month'] = loads_df.date.apply(lambda x: x.day)

# 按照id和日期进行重新排序
loads_df = loads_df.sort_values(['id', 'date'], ascending=[True, True])
loads_df = loads_df.reset_index(drop=True)

loads_df = loads_df[loads_df.type_day <= 5]

loads_wide_df = pd.pivot_table(data=loads_df,columns=['date','day_of_month'],values='energy_use',index=['id'])
loads_wide_df = loads_wide_df.dropna()
print(loads_wide_df)
# unique_days = loads_df.day_of_month.unique()
# print(unique_days)
# loads_wide_df = loads_wide_df.xs(10,level='day_of_month',axis=1)
# loads_wide_df = loads_wide_df.dropna()
# print(loads_wide_df)

class EnergyFingerPrints():

    def __init__(self, data):
        # 统计每个聚类簇的中心点
        self.means = []
        self.data = data

    def elbow_method(self, n_clusters):
        fig, ax = plt.subplots(figsize=(8, 4))
        distortions = []
        for i in range(1, n_clusters):
            km = KMeans(n_clusters=i,
                        init='k-means++',  # 初始中心簇的获取方式，k-means++一种比较快的收敛的方法
                        n_init=10,  # 初始中心簇的迭代次数
                        max_iter=300,  # 数据分类的迭代次数
                        random_state=0)  # 初始化中心簇的方式
            km.fit(self.data)
            distortions.append(km.inertia_)  # inertia计算样本点到最近的中心点的距离之和

        plt.plot(range(1, n_clusters), distortions, marker='o', lw=1)
        plt.xlabel('聚类数量')
        plt.ylabel('至中心点距离之和')
        plt.show()

    def get_cluster_counts(self):  # 统计聚类簇和每个簇中样本的数量
        return pd.Series(self.predictions).value_counts()

    def labels(self, n_clusters):  # 确定每簇中样本的具体划分
        self.n_clusters = n_clusters
        return KMeans(self.n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0).fit(self.data).labels_

    def fit(self, n_clusters):  # 基于划分簇的数量，对数据进行聚类分析
        self.n_clusters = n_clusters
        self.kmeans = KMeans(self.n_clusters)
        self.predictions = self.kmeans.fit_predict(self.data)

    def plot(self):  # 分别绘制各簇中的用户用电行为曲线，并绘制各簇的平均用电行为曲线
        self.cluster_names = [str(x) for x in range(self.n_clusters)]
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for i in range(0, self.n_clusters):
            # all_data = []
            plt.subplot(2, 2, i + 1)
            for x, y in zip(self.data, self.predictions):
                if y == i:
                    # all_data.append(x)
                    plt.plot(x, alpha=0.06, color="blue")
                    plt.ylim(0, 20)
                    plt.xlim(0, 96)
                    plt.title('Cluster {}'.format(i + 1))
                    plt.ylabel('用电量/kW')

            # all_data_array = np.array(all_data)
            # mean = all_data_array.mean(axis=0)
            # self.means.append(mean)
            plt.plot(self.kmeans.cluster_centers_[i], color="black", linewidth=4)
            plt.xlim(0, 96)
            plt.ylim(0, 20)
        plt.show()

    def plot_energy_fingerprints(self):  # 将各簇的用电行为数据绘制在一张表上
        fig, ax = plt.subplots(figsize=(8, 5))

        for i in range(0, self.n_clusters):
            plt.plot(self.kmeans.cluster_centers_[i], label="cluster %s" % (str(i + 1)))
            plt.xlim(0, 96)
        plt.ylabel('用电量/kW')
        plt.legend()
        plt.show()

load_data=np.array(loads_wide_df)
energy_clusters = EnergyFingerPrints(load_data)
energy_clusters.elbow_method(n_clusters=13)
energy_clusters.fit(n_clusters=4)
energy_clusters.plot()
energy_clusters.get_cluster_counts()
energy_clusters.plot_energy_fingerprints()
loads_wide_df.insert(loc=len(loads_wide_df.columns), column='pred', value=energy_clusters.predictions)
print(loads_wide_df)
# 获得属于第一分类簇的用户id
print(loads_wide_df[loads_wide_df['pred'] == 2].index)



