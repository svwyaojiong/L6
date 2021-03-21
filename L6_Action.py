# -*- coding: utf-8 -*-
"""
@author: 91523/YaoJiong

"""


import pandas as pd
from fbprophet import Prophet

train=pd.read_csv('./train.csv')

#转换为pandas中的日期格式
train['Datetime']=pd.to_datetime(train['Datetime'])

#将datetime作为index
train.index=train['Datetime']

#数据清洗
train.drop(['ID','Datetime'],axis=1,inplace=True)

#按照天进行采样
daily_train=train.resample('D').sum()

daily_train['ds']=daily_train.index
daily_train['y']=daily_train['Count']
daily_train.drop(['Count'],axis=1,inplace=True)


#创建模型
m=Prophet(yearly_seasonality=True,seasonality_prior_scale=0.1)
m.fit(daily_train)

#预测未来7个月，213天
future=m.make_future_dataframe(periods=213)
forecast=m.predict(future)
m.plot(forecast)

