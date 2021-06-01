#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt
import plotly.express as px

import plotly.figure_factory as ff

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#匯入資料ㄌ
df = pd.read_excel("./機器學習-本國銀行.xlsx",usecols="A:BB")  
col_names = [
 '代號', '名稱', '年月', '股本', '資產', '權益', '稅前損益',
    '授信總額', '存款', '放款', '權益比率', '資產報酬率(稅前)',
    '存放比率', '催收款', '逾期放款', '甲類逾期放款', '乙類逾期放款',
    '放款(含催收款)', '逾放比率', '備抵呆帳','備抵呆帳覆蓋率','收益總計'
    ,'費損總計','淨收益','利息收入','利息費用','淨利息收入比','資本適足率']

#重設index 欄位 > 設定欄位名稱
df = df.set_axis(col_names, axis=1, inplace=False)

#尋找遺失值
df1_lack = df[df.isnull().values == True]
df1_lack

#補值
df = df.fillna(0)


# # 抓取比率欄位

# In[3]:


# df = df[['名稱','年月','權益比率','資產報酬率(稅前)','存放比率','備抵呆帳覆蓋率','淨利息收入比','資本適足率']]
df = df[['名稱','年月','權益比率','資產報酬率(稅前)','存放比率','備抵呆帳覆蓋率','淨利息收入比']]
df['資產報酬率(稅前)'] = [x*100 for x in df['資產報酬率(稅前)']]


# In[4]:


df[df['名稱'] == '一銀']


# # 將銀行每個月份的資料作平均(可能需要做調整)

# In[5]:


mean_df = df.groupby('名稱').mean()
#刪除部分資料不完整銀行
less_list = ['匯豐台灣', '星展台灣', '樂天銀行', '澳盛台灣', '大眾銀', '慶豐銀行', '寶華', '中華銀行']
#less_list.extend(['花旗台灣','京城銀','中輸銀'])
mean_df = mean_df.drop(less_list)
mean_df.reset_index(inplace=True)


# In[6]:


#各欄位相關係數矩陣
res = []
for i in mean_df.corr().values:
    x = []
    for j in i:
        x.append(round(j,2))
    res.append(x)
z = res
fig = ff.create_annotated_heatmap(z,
                                  colorscale='AGsunset',
                                 x = [x for x in mean_df.corr().columns],
                                 y = [x for x in mean_df.corr().columns])
fig.show()


# # 透過 pca 將至2維

# In[7]:


X = mean_df[mean_df.columns[1:]]
pca = PCA(n_components=2)
pca.fit(X)
pca_result = pca.transform(X)
pca_df = pd.DataFrame(pca_result)
pca_df


# In[8]:


# 用Kmeans的肘分析看要選多少群


# In[9]:


SSE = []
for i in range(1,11):
    estimator = KMeans(n_clusters=i)  
    estimator.fit(pca_result)
    SSE.append(estimator.inertia_)
x = range(1,11)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(x,SSE,'o-')


# In[10]:


silhouette_avg = []
for i in range(2,11):
    kmeans_fit = KMeans(n_clusters = i).fit(pca_result)
    silhouette_avg.append(silhouette_score(pca_result, kmeans_fit.labels_))
plt.plot(range(2,11), silhouette_avg)


# In[11]:


sco = []
for i in range(2,11):
    kmeans = KMeans(n_clusters = i).fit(pca_result)
    score = calinski_harabaz_score(pca_result , kmeans.labels_)
    sco.append(score)
#     print('分%d群的calinski_harabaz分數為：%f'%(i , score))
plt.plot(range(2,11),sco)


# # 決定分幾群
# 試了幾種評估方法：Kmeans的肘分析、Silhouette 輪廓分析、calinski_harabaz_score，最後**輪廓分析**的結果較為明顯>以輪廓係數法決定分成五群

# In[24]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(pca_df[[0,1]])
labels = [str(x) for x in kmeans.predict(pca_df[[0,1]])]
pca_df['分群結果'] = labels
mean_df['分群結果'] = labels

fig = px.scatter(pca_df, x=0, y=1,color="分群結果",text=[x for x in mean_df.名稱])
fig.update_layout(
    height=800,
    font=dict(
    size=10
    )
)
fig.update_traces(textposition='top center')


# In[14]:


# fig = px.scatter_matrix(mean_df, dimensions=['權益比率','資產報酬率(稅前)','存放比率','逾放比率','備抵呆帳覆蓋率','淨利息收入比'], color="分群結果")
fig = px.scatter_matrix(mean_df, dimensions=['權益比率','資產報酬率(稅前)','存放比率','備抵呆帳覆蓋率','淨利息收入比'], color="分群結果")

fig.update_layout(
    height=800,
)
fig.show()


# # 權益比率在各群的分布狀況

# In[15]:


mean_df.columns


# In[16]:


px.box(mean_df, x="分群結果", y=mean_df.columns[1], points="all")


# # 資產報酬率(税前)在各群的分布狀況

# In[17]:


px.box(mean_df, x="分群結果", y=mean_df.columns[2], points="all")


# # 存放比率在各群的分布狀況

# In[18]:


px.box(mean_df, x="分群結果", y=mean_df.columns[3], points="all")


# # 備抵呆帳覆蓋率在各群的分布狀況

# In[19]:


px.box(mean_df, x="分群結果", y=mean_df.columns[4], points="all")


# # 淨利息收入比在各群的分布狀況

# In[20]:


px.box(mean_df, x="分群結果", y=mean_df.columns[5], points="all")


# In[21]:


cluster_mean_df = mean_df.groupby('分群結果').mean().reset_index()
cluster_mean_df


# In[22]:


import plotly.graph_objects as go

cols = cluster_mean_df.columns
fig = go.Figure(data=[
    go.Bar(name='群 0', x= cols[1:], y = [x for x in cluster_mean_df.loc[0][1:]],text = [x for x in cluster_mean_df.loc[0][1:]],textposition='auto'),
    go.Bar(name='群 1', x= cols[1:], y = [x for x in cluster_mean_df.loc[1][1:]],textposition='auto',text=[x for x in cluster_mean_df.loc[1][1:]]),
    go.Bar(name='群 2', x= cols[1:], y = [x for x in cluster_mean_df.loc[2][1:]],textposition='auto',text=[x for x in cluster_mean_df.loc[2][1:]]),
    go.Bar(name='群 3', x= cols[1:], y = [x for x in cluster_mean_df.loc[3][1:]],textposition='auto',text=[x for x in cluster_mean_df.loc[3][1:]]),
    go.Bar(name='群 4', x= cols[1:], y = [x for x in cluster_mean_df.loc[4][1:]],textposition='auto',text=[x for x in cluster_mean_df.loc[4][1:]]),  
])
# Change the bar mode
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(barmode='group',title='指標於各群的平均')
fig.show()


# In[23]:


res_bank_df = pd.DataFrame()
for g in [str(x) for x in range(5)]:
    res_bank_df['第'+g+'群'] =['、'.join(mean_df[mean_df.分群結果 == str(g)]['名稱'].unique())]

d = dict(selector="th", 
    props=[('text-align', 'center')]) 

res_bank_df.style.set_properties(**{'width':'15em'}).set_table_styles([d]) 


# # 結論

# 共分五個群，其中:群`0`、`2`、`3` 個數較少，是甚麼可能原因造就了這幾個outlier?
# 
# - 在`權益比率`、`資產報酬率`、`淨利息收入比`，群會有`高中低`之分
# - 在`備抵呆帳覆蓋率`、`存放比率`，群會有很明顯的`高低`之分
# 
# 五個類別特色如下:
# 
# - 群 - 0: 權益比率低，資產報酬率中、存放比率高、備抵呆帳覆蓋率低、淨利息收入比高
# - 群 - 1: 權益比率中，資產報酬率高、存放比率高、備抵呆帳覆蓋率低、淨利息收入比中
# - 群 - 2: 權益比率低，資產報酬率中、存放比率高、備抵呆帳覆蓋率低、淨利息收入比中
# - 群 - 3: 權益比率高，資產報酬率中、存放比率低、備抵呆帳覆蓋率高、淨利息收入比低
# - 群 - 4: 權益比率中，資產報酬率低、存放比率高、備抵呆帳覆蓋率低、淨利息收入比低

# In[ ]:




