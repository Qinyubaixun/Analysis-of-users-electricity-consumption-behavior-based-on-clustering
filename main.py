import inline
from pandas import read_sql
import pymysql

inline
import numba
from numba import jit
import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
from sklearn.cluster import KMeans
import sqlite3
import datetime
from pylab import mpl
from sklearn_extra.cluster import KMedoids
import tslearn.metrics as metrics
from tslearn.clustering import silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.generators import random_walks
from sklearn.preprocessing import StandardScaler

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# # 设置起始日期和结束日期
# start_date = datetime.datetime(2021, 1, 1, 0, 0)
# end_date = datetime.datetime(2021, 1, 2, 0, 0)
#
# # 定义时间间隔为15分钟
# delta_time = datetime.timedelta(minutes=1)
#
# # 生成时间间隔数组
# time_arr = []
# while start_date < end_date:
#     time_arr.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
#     start_date += delta_time
#
#
# time_arr = np.tile(time_arr, 23)
# time_arr = pd.DataFrame(time_arr.T)
# print(time_arr)
# conn = pymysql.connect(
#     host="localhost",
#     port=3306,  # 端口号是数字类型
#     database="person",  # 写自己本地的数据库名字
#     user="root",
#     password="123456",
#     charset="utf8"   # 千万记得没有-
#  )
cwd = os.getcwd()  #获得当前的文件路径
conn = sqlite3.connect(str(cwd) + "/dataport_sqlite")
cursor = conn.cursor()  #构建指针
query = "SELECT * FROM new_table;"
cursor.execute(query)    #执行读取数据命令
data = cursor.fetchall() #抓取数据

loads_df = pd.DataFrame(data, columns=['id','date','energy_use'])
print(loads_df['id'].unique().shape)

print("当前数据集含有%s行,%s列"%(loads_df.shape[0],loads_df.shape[1]))
print(" 最早时间: %s \n 最晚时间: %s"%(loads_df.date.min(),loads_df.date.max()))

loads_df['id'].value_counts()

loads_df = loads_df.replace('',np.nan)
print(loads_df.isnull().sum())
loads_df = loads_df.dropna()
print(loads_df.isnull().sum())
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

unique_days = loads_df.day_of_month.unique()

# loads_wide_df = pd.concat([loads_wide_df.xs(10,level='day_of_month',axis=1) for day in unique_days])
loads_wide_df = loads_wide_df.dropna()
loads_wide_df = np.array(loads_wide_df)
print(loads_wide_df.shape)
# elbow法则找最佳聚类数，结果：elbow = 5

def test_elbow():
    distortions = []
    dists = metrics.cdist_dtw(loads_wide_df)  # dba + dtw
    # dists = metrics.cdist_soft_dtw_normalized(X,gamma=.5) # softdtw
    for i in range(2, 15):
        km = KMedoids(n_clusters=i, random_state=0, metric="precomputed")
        km.fit(dists)
        # 记录误差和
        distortions.append(km.inertia_)
    plt.plot(range(2, 15), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


def test_kmedoids():
    num_cluster = 8
    # 声明precomputed自定义相似度计算方法
    km = KMedoids(n_clusters=num_cluster, random_state=0, metric="precomputed")
    # 采用tslearn中的DTW系列及变种算法计算相似度，生成距离矩阵dists
    dists = metrics.cdist_dtw(loads_wide_df)  # dba + dtw
    # dists = metrics.cdist_soft_dtw_normalized(X,gamma=0.5) # softdtw
    print(dists)
    y_pred = km.fit_predict(dists)
    np.fill_diagonal(dists, 0)
    score = silhouette_score(dists, y_pred, metric="precomputed")
    print(loads_wide_df.shape)
    print(y_pred)
    print("silhouette_score: " + str(score))
    # loads_wide_df.insert(loc=len(loads_wide_df), column='pred', value=y_pred)
    print(loads_wide_df)
    for yi in range(num_cluster):
        plt.subplot(4, 2, yi + 1)
        for xx in loads_wide_df[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.3)
        # 注意这里的_cluster_centers要写成X[km.medoid_indices_[yi]]，因为你是precomputed，源码里面当precomputed时_cluster_centers等于None
        print(loads_wide_df[km.medoid_indices_[yi]])
        plt.plot(loads_wide_df[km.medoid_indices_[yi]], "r-")
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("KMedoids" + " + DBA-DTW")

    plt.tight_layout()
    plt.show()


# test_elbow()
test_kmedoids()
