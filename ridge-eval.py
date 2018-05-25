# -*- coding: utf-8 -*-
"""
Created on Mon May 21 21:14:56 2018

@author: lenovo
"""


# coding: utf-8

# In[1]:


import requests
import pandas as pd
import numpy as np
import datetime
import math

today = datetime.date.today()

today = (today - datetime.timedelta(days=4))
tommorow = (today + datetime.timedelta(days=1)).__format__('%Y-%m-%d')

yesterday = (today - datetime.timedelta(days=1)).__format__('%Y-%m-%d')

fromday = (today - datetime.timedelta(days=6)).__format__('%Y-%m-%d')
#https://biendata.com/competition/airquality/bj/2018-05-01-0/2018-05-03-17/2k0d1d8
url = 'https://biendata.com/competition/airquality/bj/%s-0/%s-23/2k0d1d8' % (fromday, yesterday)
file_name = 'D:/Jupyter/weather/bj_airquality_%s-0-%s-23.csv' % (fromday, yesterday)

url_today = 'https://biendata.com/competition/airquality/bj/%s-0/%s-23/2k0d1d8' % (today,tommorow)
today_file = 'D:/Jupyter/weather/bj_airquality_%s-0-%s-23.csv' % (today,tommorow)

respones= requests.get(url)
with open (file_name,'w') as f:
    f.write(respones.text)
    
respones_t= requests.get(url_today)
with open (today_file,'w') as f:
    f.write(respones_t.text)

df = pd.read_csv(file_name)

df = df.rename(columns = {'PM25_Concentration': 'PM2.5', 'PM10_Concentration':'PM10' , 'O3_Concentration':'O3'})
# df.head()


# In[2]:


df[['PM2.5', 'PM10', 'O3']] = df[['PM2.5', 'PM10', 'O3']].fillna(df[['PM2.5', 'PM10', 'O3']].mean())
# df.head()


# In[3]:


from sklearn.externals import joblib
s = pd.read_csv('D:/Jupyter/weather/sample_submission.csv')

df_set = set(df['station_id'].value_counts().to_dict().keys())
df_set_9 = set(['daxing_aq','fangshan_aq','huairou_aq','mentougou_aq','miyun_aq','pingchang_aq', 
                'pinggu_aq','shunyi_aq','tongzhou_aq'])

# submit_index = {i:i for i in df_set}
# submit_index['aotizhongxin_aq'] = 'aotizhongx_aq'
# submit_index['xizhimenbei_aq'] = 'xizhimenbe_aq'
# submit_index['wanshouxigong_aq'] = 'wanshouxig_aq'
# submit_index['miyunshuiku_aq'] = 'miyun_aq'
# submit_index['nongzhanguan_aq'] = 'nongzhangu_aq'
# submit_index['yongdingmennei_aq'] = 'yongdingme_aq'
# submit_index['fengtaihuayuan_aq'] = 'fengtaihua_aq'


# In[4]:


def score(estimator, X, y):
    y_prediction = estimator.predict(X)
    return np.sum(np.abs(y_prediction - y) / (np.abs(y_prediction) + np.abs(y))) / y.shape[0]

print(df.iat[-1,2])

for station in df_set_9:
    aq = df[df['station_id'] == station]
    for p in ['PM2.5', 'PM10', 'O3']:
        array = np.array(aq[p])[-120:]
        grid = joblib.load('D:/Jupyter/weather/%s_%s.pkl' % (station, p))
        y = grid.predict(np.expand_dims(array, axis=0))
        for i in range(48):
            try:
                s.loc[s[s['test_id'] == '%s#%d' % (station, i)].index[0], p] = y[0][i] if y[0][i] > 0 else abs(y[0][i])
            except:
                print(station, i)


# In[8]:


s.set_index('test_id').to_csv('D:/Jupyter/weather/sample_submission.csv')
