#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff


# In[2]:


#匯入資料
df = pd.read_excel("機器學習-銀行.xlsx",usecols="A:T")  


# In[3]:


col_names = [
 '代號',
 '名稱',
 '年月',
 '股本',
 '資產',
 '權益',
 '稅前損益',
 '授信總額',
 '存款',
 '放款',
 '權益比率',
 '資產報酬率(稅前)',
 '存放比率',
 '催收款',
 '逾期放款',
 '甲類逾期放款',
 '乙類逾期放款',
 '放款(含催收款)',
 '逾放比率',
 '備抵呆帳']


# In[4]:


#重設index 欄位 > 設定欄位名稱
df = df.set_axis(col_names, axis=1, inplace=False)

#刪除row資料(本國小計,外國小計,陸銀小計,總計)
df_idx=df[df['代號']=='Z7777'].index|df[df['代號']=='Z8888'].index|df[df['代號']=='Z8890'].index|df[df['代號']=='Z9999'].index
df=df.drop(df_idx)

#按時間由近到遠排列
df = df.reindex(index=df.index[::-1])
df.reset_index(drop=True,inplace=True)
df


# In[5]:


#尋找遺失值
df1_lack = df[df.isnull().values == True]
df1_lack


# In[6]:


df = df.fillna(0)


# In[7]:


#各欄位相關係數矩陣 > 按照欄位間的相關係數分配工作
res = []
for i in df.corr().values:
    x = []
    for j in i:
        x.append(round(j,2))
    res.append(x)
z = res
fig = ff.create_annotated_heatmap(z,
                                  colorscale='AGsunset',
                                 x = [x for x in df.corr().columns],
                                 y = [x for x in df.corr().columns])
fig.show()


# # 股本 資產 權益 稅前損益 授信總額 存款 放款欄位 EDA

# In[24]:


#部分銀行沒有全部時段資料，如果要繪製時間序咧圖可以分開來看
bank_df = pd.DataFrame()
bank_df['名稱'] = df['名稱'].value_counts().keys()
bank_df['總數'] = df['名稱'].value_counts().values
fig = px.bar(bank_df, x='名稱', y='總數')
fig.show()


# In[8]:


mean_share_df = pd.DataFrame()
mean_share_df['名稱'] = df.groupby('名稱').mean()['股本'].sort_values(ascending = False).keys()
mean_share_df['股本'] = df.groupby('名稱').mean()['股本'].sort_values(ascending = False).values
# mean_share_df
# fig = px.histogram(mean_share_df, x="股本",nbins=50)
# fig.show()
fig = px.bar(mean_share_df, 
             x='名稱',
             y='股本',
             title='各公司每年平均股本',text='股本')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# In[9]:


mean_share_df[:10]


# In[10]:


#顯示所有名稱的時間序列圖
def make_plot_all(col_name):
    asset_df = pd.DataFrame()
    asset_df['年月'] = [x for x in df[df['名稱'] == '合庫']['年月']]
    bank_list = df.名稱.unique()
    less_list = []

    for bank in bank_list:
        as_list = [x for x in df[df['名稱'] == bank][col_name]]
        if len(as_list) == 156:
            asset_df[bank + col_name] = as_list
        else:
            less_list.append(bank)
    print(less_list)

    #時間不足
    for bank in less_list:
        _df = df[df['名稱'] == bank][['年月',col_name]]
        _df.columns = ['年月',bank + col_name]
        asset_df = asset_df.merge(_df, how='outer', on='年月')

    fig = px.line(asset_df, x="年月", y=asset_df.columns,
                  title='各公司'+col_name+'時間序列')
    fig.show()


# In[11]:


#顯示未缺少任何時間資料的銀行名稱時間序列圖
def make_plot_more(col_name):
    asset_df = pd.DataFrame()
    asset_df['年月'] = [x for x in df[df['名稱'] == '合庫']['年月']]
    bank_list = df.名稱.unique()
    less_list = []

    for bank in bank_list:
        as_list = [x for x in df[df['名稱'] == bank][col_name]]
        if len(as_list) == 156:
            asset_df[bank + col_name] = as_list

    fig = px.line(asset_df, x="年月", y=asset_df.columns,
                  title='各公司'+col_name+'時間序列')
    fig.show()


# In[12]:


#顯示缺少任何時間資料的銀行名稱時間序列圖
def make_plot_less(col_name):
    plot_df = pd.DataFrame()
    plot_df['年月'] = [x for x in df[df['名稱'] == '合庫']['年月']]
    bank_list = df.名稱.unique()
    less_list = []

    for bank in bank_list:
        as_list = [x for x in df[df['名稱'] == bank][col_name]]
        if len(as_list) == 156:
            plot_df[bank + col_name] = as_list
        else:
            less_list.append(bank)

    #時間不足
    for bank in less_list:
        _df = df[df['名稱'] == bank][['年月',col_name]]
        _df.columns = ['年月',bank + col_name]
        plot_df = plot_df.merge(_df, how='outer', on='年月')
        
    select_col = [x+col_name for x in less_list]
    select_col.append('年月')
#     print(plot_df)
    less_plot_df = plot_df[select_col]
    
    fig = px.line(less_plot_df, x="年月", y=less_plot_df.columns,
                  title='各公司'+col_name+'時間序列')
    fig.show()


# In[13]:


make_plot_all('資產')


# In[14]:


make_plot_more('資產')


# In[15]:


make_plot_less('資產')


# In[16]:


make_plot_more('權益')


# In[17]:


make_plot_more('股本')


# In[18]:


make_plot_more('稅前損益')


# In[19]:


make_plot_more('授信總額')


# In[20]:


make_plot_more('存款')


# In[21]:


make_plot_more('放款')


# In[22]:


df['存放款比率'] = df['放款'] / df['存款']


# In[23]:


make_plot_more('存放款比率')


# In[ ]:




