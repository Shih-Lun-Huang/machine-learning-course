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
df = pd.read_excel("./../dataset/機器學習-本國銀行.xlsx",usecols="A:AA")  
col_names = [
 '代號', '名稱', '年月', '股本', '資產', '權益', '稅前損益',
    '授信總額', '存款', '放款', '權益比率', '資產報酬率(稅前)',
    '存放比率', '催收款', '逾期放款', '甲類逾期放款', '乙類逾期放款',
    '放款(含催收款)', '逾放比率', '備抵呆帳','備抵呆帳覆蓋率','收益總計'
    ,'費損總計','淨收益','利息收入','利息費用','淨利息收入比']

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


df


# # 將銀行每個月份的資料作平均(可能需要做調整)

# In[5]:


mean_df = df.groupby('名稱').mean()
#刪除部分資料不完整銀行
less_list = ['匯豐台灣', '星展台灣', '樂天銀行', '澳盛台灣', '大眾銀', '慶豐銀行', '寶華', '中華銀行']
less_list.extend(['中輸銀'])
#less_list.extend(['花旗台灣','京城銀','中輸銀'])
mean_df = mean_df.drop(less_list)
mean_df.reset_index(inplace=True)


# In[6]:


mean_df


# In[7]:


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

# In[8]:


X = mean_df[mean_df.columns[1:]]
pca = PCA(n_components=2)
pca.fit(X)
pca_result = pca.transform(X)
pca_df = pd.DataFrame(pca_result)
pca_df


# In[9]:


# 用Kmeans的肘分析看要選多少群


# In[10]:


SSE = []
for i in range(1,11):
    estimator = KMeans(n_clusters=i)  
    estimator.fit(pca_result)
    SSE.append(estimator.inertia_)
x = range(1,11)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(x,SSE,'o-')


# In[11]:


silhouette_avg = []
for i in range(2,11):
    kmeans_fit = KMeans(n_clusters = i).fit(pca_result)
    silhouette_avg.append(silhouette_score(pca_result, kmeans_fit.labels_))
plt.plot(range(2,11), silhouette_avg)


# In[12]:


sco = []
for i in range(2,11):
    kmeans = KMeans(n_clusters = i).fit(pca_result)
    score = calinski_harabaz_score(pca_result , kmeans.labels_)
    sco.append(score)
#     print('分%d群的calinski_harabaz分數為：%f'%(i , score))
plt.plot(range(2,11),sco)


# # 決定分幾群
# 試了幾種評估方法：Kmeans的肘分析、Silhouette 輪廓分析、calinski_harabaz_score，最後**輪廓分析**的結果較為明顯>以輪廓係數法決定分成五群

# In[13]:


kmeans = KMeans(n_clusters=3)
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
# fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
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


# # 分群結果

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
])
# Change the bar mode
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(barmode='group',title='指標於各群的平均')
fig.show()


# In[23]:


res_bank_df = pd.DataFrame()
for g in [str(x) for x in range(3)]:
    res_bank_df['第'+g+'群'] =['、'.join(mean_df[mean_df.分群結果 == str(g)]['名稱'].unique())]

d = dict(selector="th", 
    props=[('text-align', 'center')]) 

res_bank_df.style.set_properties(**{'width':'15em'}).set_table_styles([d]) 


# ## 群0
# - 銀行：三信銀行、凱基銀行、日盛銀行、板信銀、瑞興銀、臺企銀、臺銀、華泰銀行、高雄銀
# - 指標較高：淨利息收入比
# - 指標較低：權益比率、資產報酬率
# - 指標較不明顯：存放比率、備抵呆帳覆蓋率
# 
# ## 群1
# - 銀行：上海商銀、中信銀、京城銀、花旗台灣
# - 指標較高：權益比率、資產報酬率
# - 指標較低：淨利息收入比
# - 指標較不明顯：存放比率、備抵呆帳覆蓋率（但有離群值）
# 
# ## 群2
# - 銀行：一銀、元大銀、兆豐商銀、台中銀、台北富邦銀、台新銀、合庫、國泰世華、土銀、安泰銀、彰銀、新光銀行、永豐銀行、渣打銀行、玉山銀、王道銀行、聯邦銀、華銀、遠東銀、陽信銀
# - 指標較高：無
# - 指標較低：權益比率、資產報酬率、淨利息收入比
# - 指標較不明顯：存放比率、備抵呆帳覆蓋率
# 
# ## 小結
# 利用我們所做的分群結果可以得知每家銀行的指標特性，客戶可以針對這些指標去選擇符合自己所需之銀行。也可以檢視自己手中的資產，是否處在一個相對安全的金融環境。

# In[ ]:




