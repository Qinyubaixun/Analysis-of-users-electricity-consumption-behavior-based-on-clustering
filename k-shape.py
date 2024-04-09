import os
import csv
import sqlite3
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pylab import mpl
from scipy.io import arff
from sklearn import preprocessing
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import rand_score, normalized_mutual_info_score, adjusted_rand_score
from tslearn.clustering import KShape
from kshape.core import kshape
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

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
loads_wide_df = loads_wide_df.xs(3, level='day_of_month',axis=1)
loads_wide_df = loads_wide_df.dropna()
loads_wide_df = np.array(loads_wide_df)
print(loads_wide_df.shape)
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
data_scaled = scaler.fit_transform(loads_wide_df)

tsl_times = []
def test_elbow():
    distortions = []
    for i in range(2, 20):
        km = KShape(n_clusters=i, n_init=1, random_state=0).fit(data_scaled)
        # 记录误差和
        distortions.append(km.inertia_)
    plt.plot(range(2, 20), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
# for i in range(5):
#     start_time = time.time()
#
#     start_time = time.time()
#     ks = KShape(n_clusters=num_clusters, n_init=1, random_state=0).fit(loads_wide_df)
#
#     tsl_times.append(time.time() - start_time)
# # %%
# print('Mean TSLearn Benchmark for 5 Runs:', np.mean(tsl_times))
def test_kmedoids():
    num_cluster = 16
    # 声明precomputed自定义相似度计算方法
    ks = KShape(n_clusters=num_cluster, n_init=1, random_state=0).fit(data_scaled)
    y_pred = ks.fit_predict(data_scaled)
    print(data_scaled.shape)
    print(y_pred)
    loads = np.insert(loads_wide_df, 0, values=y_pred, axis=1)
    # loads_wide_df.insert(loc=len(loads_wide_df), column='pred', value=y_pred)
    print(loads[0:5, 0])
    fig, ax = plt.subplots(figsize=(18, 15))
    for yi in range(num_cluster):
        plt.subplot(8, 2, yi + 1)
        for xx in loads[loads[:, 0] == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.3)

        print(np.std(loads_wide_df[yi, :]))
        plt.plot(ks.cluster_centers_[yi, :,:] * np.std(loads_wide_df[yi, :]) + np.mean(loads_wide_df[yi, :]), "r-")
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        plt.ylabel('用电量/kW')
        plt.xlabel('采样点')
        if yi == 1:
            plt.title("KMedoids" + " + DBA-DTW")

    plt.tight_layout()
    plt.show()
    for i in range(num_cluster):
        plt.plot(ks.cluster_centers_[i, :, :] * np.std(loads_wide_df[i, :]) + np.mean(loads_wide_df[i, :]), label="cluster %s" % (str(i + 1)))
    plt.ylabel('用电量/kW')
    plt.legend()
    plt.xlabel('采样点')
    plt.show()
# %%
test_elbow()
test_kmedoids()
# ri_ks = rand_score(predictions, labels)
# print('Rand Score:', ri_ks)
# ari_ks = adjusted_rand_score(predictions, labels)
# print('Adjusted Rand Score:', ari_ks)
# nmi_ks = normalized_mutual_info_score(predictions, labels)
# print('Normalized Mutual Information:', nmi_ks)