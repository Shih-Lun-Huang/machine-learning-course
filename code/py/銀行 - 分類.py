#!/usr/bin/env python
# coding: utf-8

# # 分類 - 預測銀行當月風險性

# ## 標記資料
# 利用資本適足率(BIS)作為風險分類標準。
# 由於本國銀行資本適足率全都符合法規標準（皆高於8%），因此以當季資本適足率平均值作為界，當月資本適足率高於平均值的銀行標記為“表現穩健”，低於平均值的銀行標記為“風險承受度稍弱“。
# 
# 例：2020年第四季本國銀行資本足率平均值為14.89，台灣銀行2020年第四季的BIS為14.95，則台灣銀行在2020年10月、11月、12月的分類標記為“表現穩健”。
# 
# ## 資料前處理與建模
# 1.刪除六間銀行（2020年不營業的本國銀行五間、2020年開始營業的銀行一間）<br/>
# 2.空值補0：銀行可能沒有承辦該項業務，因此沒有資料 <br/>
# 3.從特徵相關係數矩陣挑出與資本適足率高度相關的特徵作為input <br/>
# 4.將2008-2017年的資料用SVM建模，預測2018-2020年的分類是否正確

# In[1]:


# 載入所需套件
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


# In[2]:


#匯入資料
df = pd.read_csv('./../dataset/機器學習-本國銀行v3.csv',na_values=['#VALUE!', '#DIV/0!'])
df


# In[3]:


#刪除2020年不營業的銀行
del_list = ['大眾銀','中華銀行','慶豐銀行','澳盛台灣','寶華','樂天銀行']

for i in del_list:
    df.drop(index=(df.loc[(df['名稱']==i)].index),inplace=True)


# In[4]:


df.info()


# In[5]:


# 檢查空值
df.isnull().sum()


# In[6]:


# 風險分類計數
pd.DataFrame(df.分類.value_counts()).reset_index()


# In[7]:


# 風險分類計數長條圖

fig = px.bar(pd.DataFrame(df.分類.value_counts().reset_index()), x='index', y='分類')
fig.show()


# ## 特徵挑選

# In[8]:


# 欄位相關係數矩陣

corrs = df.corr()

corrs_heatmap = ff.create_annotated_heatmap(z=corrs.values,x=list(corrs.columns),y=list(corrs.index), annotation_text=corrs.round(2).values, showscale=True)

corrs_heatmap


# In[9]:


# 挑出跟資本適足率強相關的特徵（係數>0.5）：權益比率、存放比率
corrs['資本適足率'].sort_values(ascending=False)


# In[10]:


# 分類與權益比的關係

fig = px.box(df, x='分類', y='權益比率')
fig.update_layout(title='2008-2020年銀行各風險分類的權益比率比較')
fig.show()


# In[11]:


# 分類與存放比的關係

fig = px.box(df, x='分類', y='存放比率')
fig.update_layout(title='2008-2020年銀行各風險分類的存放比率比較')
fig.show()


# ## 建立訓練模型

# In[12]:


# 重設index
df.reset_index(drop=True,inplace=True) 

# 將分類轉成數值（1:表現穩健/2:風險承受度稍弱）
arr = []
for i in range(len(df)):
    if df['分類'][i] == '表現穩健':
        arr.append(1)
    else: #風險承受度稍弱
        arr.append(2)

df['class'] = arr


# In[13]:


# 分割train and test(2008-2017年的資料)
X = df[1296:]

X_train, X_test, y_train, y_test = train_test_split(
    X[['權益比率','存放比率']], X[['class']], test_size=0.3, random_state=0)


# In[14]:


# 找最佳參數

def svm_cross_validation(train_x, train_y):    
    from sklearn.model_selection import GridSearchCV   
    from sklearn.svm import SVC    
    model = SVC(kernel='rbf', probability=True)    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
    grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)    
    grid_search.fit(train_x, train_y)    
    best_parameters = grid_search.best_estimator_.get_params()    
    for para, val in list(best_parameters.items()):    
        print(para, val)    
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
    model.fit(train_x, train_y)    

svm_cross_validation(X_train,y_train)


# In[15]:


# SVM建模
svc = svm.SVC(kernel='rbf', C=100, gamma=0.0001)

svc.fit(X_train,y_train)


# In[16]:


# 預測測試集

predictions = svc.predict(X_test)

#載入classification_report & confusion_matrix來評估模型好壞

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# ## 預測2018-2020年的風險分類

# In[17]:



X_new = df[:1296][['權益比率','存放比率']]
y_new = df[:1296][['class']]

predictions_new = svc.predict(X_new)


# In[18]:


# 混淆矩陣評估預測準確度

print(confusion_matrix(y_new,predictions_new))
print('\n')
print(classification_report(y_new,predictions_new))


# In[19]:


X_new['原分類']=y_new
X_new['預測分類']=predictions_new


# In[20]:


# 轉換標籤與重整2018-2020年資料
X_new[['名稱','年月']] = df[:1296][['名稱','年月']]

arr_1 = []
for i in range(len(X_new)):
    if X_new['原分類'][i] == 1:
        arr_1.append('表現穩健')
    else:
        arr_1.append('風險承受度稍弱')

X_new['原分類'] = arr_1

arr_2 = []
for i in range(len(X_new)):
    if X_new['預測分類'][i] == 1:
        arr_2.append('表現穩健')
    else:
        arr_2.append('風險承受度稍弱')

X_new['預測分類'] = arr_2

X_new = X_new[['名稱','年月','權益比率','存放比率','原分類','預測分類']] 
X_new


# In[21]:


fig = px.box(X_new, x='預測分類', y='權益比率')
fig.update_layout(title='預測2018-2020年銀行風險分類後的權益比率分佈')
fig.show()


# In[22]:


fig = px.box(X_new, x='預測分類', y='存放比率')
fig.update_layout(title='預測2018-2020年銀行風險分類後的存放比率分佈')
fig.show()


# # 結論
# 
# 1.從相關係數矩陣中挑選權益比與存放比與資本適足率有高度相關，另外，從盒型圖中觀察到：<br/>
# (1)表現穩健銀行的權益比率數值分布較風險承受度較弱的銀行‘高’<br/>
# (2)表現穩健銀行的存放比率數值分布較風險承受度較弱的銀行‘廣’
# 
# 2.以訓練好的模型預測2018-2020年的風險分類，在預測“風險承受度較弱”比”表現穩健“的分類準確（0.84>0.59）

# In[ ]:




