#!/usr/bin/env python
# coding: utf-8

# # 銀行分類問題-是否能正確分類當月信用觀察評等？
# # (未完成)

# - 查找銀行各銀行當月信用評等標籤 <br />
# <br />
# - 對照信用評等的資料 - 信用展望 成為類別標籤 <br />
# <br />
# - 以兆豐為例
# 2008/5/9 信用平等是穩的 代表到下一次信用評等錢都是穩的 <br />
# 2008/9/25 信用平等是負的 代表到下一次信用評等錢都是負的 <br />
# 2009/10/01 穩 <br />
# <br />
# - 由日期判斷評等標籤：
# 如果 > 月中(15號) 代表下月的信用評等 ＝ 當月信用評等 <br />
# < 15 = 當月信用評等 <br />
# <br />
# - 兆豐：
# 2008/5.6.7.8.9 穩  <br />
# 2008/10.11.12 ~ 2009/9 負  <br />
# 2009/10 穩 <br />

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


# # 資料匯入，並刪除部分資料：
# - 刪除大部分銀行月份資料缺失的銀行 ex.樂天銀行
# - 刪除該銀行大部分缺少信用觀察的類別標籤資料

# In[2]:


df = pd.read_csv('銀行_class.csv',index_col = 0)
# df = df[['名稱','年月','權益比率','資產報酬率(稅前)','存放比率','逾放比率','備抵呆帳覆蓋率','淨利息收入比','信用觀察']]


# In[3]:


bank_list = [x for x in {k:v for k,v in df.名稱.value_counts().iteritems() if v == 156}.keys()]
new_df = df[df['名稱'].isin(bank_list)]


# In[4]:


new_df[new_df.isnull().values == True]['名稱'].value_counts()


# In[5]:


bank_list2 = [x for x in {k:v for k,v in new_df[new_df.isnull().values == True]['名稱'].value_counts().iteritems() if v > 100}.keys()]


# In[6]:


df = df[df['名稱'].isin([x for x in set(bank_list) - set(bank_list2)])]


# In[7]:


df.isnull().sum()


# # 各標籤計數

# In[8]:


pd.DataFrame(df.信用觀察.value_counts()).reset_index()


# In[9]:


fig = px.bar(pd.DataFrame(df.信用觀察.value_counts().reset_index()), x='index', y='信用觀察')
fig.show()


# # 以眾數 - 穩定 填入剩下的缺失值

# In[10]:


df = df.fillna('穩定')
df.reset_index(drop=True,inplace=True)


# 產生數值型別類別標籤，因為目前的標籤是有程度上的差異，所以依序將`正向`、`穩定`、`發展中`、`負向`標為`4`、`3`、`2`、`1`

# In[11]:


arr = []
for i in range(len(df)):
    if df['信用觀察'][i] == '正向':
        arr.append(4)
    elif df['信用觀察'][i] == '穩定':
        arr.append(3)
    elif df['信用觀察'][i] == '發展中':
        arr.append(2)
    else: #負向
        arr.append(1)
df['class'] = arr


# # 各欄位相關係數矩陣

# In[12]:


def make_corr_plot(data):
    res = []
    for i in data.corr().values:
        x = []
        for j in i:
            x.append(round(j,2))
        res.append(x)
    z = res
    fig = ff.create_annotated_heatmap(z,
                                      colorscale='AGsunset',
                                     x = [x for x in data.corr().columns],
                                     y = [x for x in data.corr().columns])
    fig.show()


# In[13]:


make_corr_plot(df)


# In[14]:


# sns.set(font_scale=1)
# fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,8))
# col_names = ['股本','逾放比率','備抵呆帳','資產報酬率(稅前)']
# for i in range(len(col_names)):
#     sns.boxplot(x="class", y=col_names[i], data=df, palette="Set1",ax=axes[i])


# In[15]:


df.columns


# # 推測信用展望的變化會考慮到上個月份的數據，將各特徵ex.'資產報酬率(稅前)、存放比率的月成長率，轉換成新的dataframe
# 
# # 將銀行後四年的資料變成我們預測的目標

# In[16]:


arr = []
bank_name = [x for x in df.名稱.unique()]
col_name = ['股本', '資產', '權益', '稅前損益', '授信總額', '存款', '放款',
       '權益比率', '資產報酬率(稅前)', '存放比率', '催收款', '逾期放款', '甲類逾期放款', '乙類逾期放款',
       '放款(含催收款)', '逾放比率', '備抵呆帳', '備抵呆帳覆蓋率', '收益總計', '利息收入', '利息費用', '淨利息收入比']
t = pd.DataFrame()
train_data = pd.DataFrame()
test_data = pd.DataFrame()

for bank in bank_name:
    _df = df[df['名稱'] == bank]
    _df.reset_index(drop=True,inplace=True)
    t['class'] = _df['class'][1:]
    t['名稱'] = _df['名稱'][1:]
    t['年月'] = _df['年月'][1:]

    # _df
    for col in col_name:
        arr = []
        for i in range(1,156):  
            last_month = _df[col][i-1]
            this_month = _df[col][i]
            arr.append(round((this_month - last_month)/last_month,2))
#         print(len(arr))
        t[col+'月成長率'] = arr
    train_data = train_data.append(t[:-48])
    test_data = test_data.append(t[-48:])


# In[17]:


train_data.reset_index(drop=True,inplace=True)
test_data.reset_index(drop=True,inplace=True)


# In[18]:


print(train_data.shape)
print(test_data.shape)


# # 填補轉換後的空值、無限大的值，並以0填入

# In[19]:


print(train_data.isnull().sum())
print(test_data.isnull().sum())


# In[20]:


train_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
test_data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)


# In[22]:


#各欄位相關係數矩陣
make_corr_plot(train_data)


# In[23]:


test_data


# In[24]:


print('訓練資料集類別')
print(train_data['class'].value_counts())
print('測試資料集類別')
print(test_data['class'].value_counts())


# In[25]:


# sns.set(font_scale=1)
# fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,8))
# col_names = ['股本月差','逾放比率月差','備抵呆帳月差','資產報酬率(稅前)月差']
# for i in range(len(col_names)):
#     sns.boxplot(x="class", y=col_names[i], data=t, palette="Set1",ax=axes[i])


# In[26]:


train_info = train_data[['名稱','年月']]
train_type = train_data['class']
test_info = test_data[['名稱','年月']]
test_type = test_data['class']

train = train_data.drop(['class','名稱','年月'],axis=1)
test = test_data.drop(['class','名稱','年月'],axis=1)


# In[30]:


col = ['資產報酬率(稅前)月成長率','稅前損益月成長率','資產月成長率']
train_data = train_data[col]
test_data = train_data[col]


# In[31]:


train_data


# # 使用SVM建模

# In[32]:


#區分訓練資料、測試資料
X_train,X_test,y_train,y_test = train_test_split(train[train.columns].values,train_type,
                                                test_size = 0.3,
                                                random_state = 1,
                                                stratify = train_type)


# In[33]:


svc = svm.SVC()
svc.fit(X_train,y_train)

y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)

svc_train = accuracy_score(y_train, y_train_pred)
svc_test = accuracy_score(y_test, y_test_pred)

print('SVM train/test accuracies %.3f/%.3f' 
      % (svc_train, svc_test))


# In[36]:


test_predict = svc.predict(test)
final_score = accuracy_score(test_type, test_predict)
print('SVM on testdata\'s accuracies: %.3f' 
      % (final_score))


# In[37]:


res_df = pd.DataFrame()
res_df['label'] = [x for x in test_type]
res_df['pred'] = [x for x in test_predict]


# In[38]:


#res_df


# # 待處理：
# - 特徵篩選: 目前特徵重要程度不大，需要再檢視原因
# - SMOTE演算法: 解決資料類別不平衡的問題，少類別會產生更多相似資料...其他方法。

# In[ ]:




