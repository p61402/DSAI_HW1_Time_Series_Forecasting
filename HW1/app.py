#!/usr/bin/env python
# coding: utf-8

# ## 載入函式庫

# In[61]:


import math
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly
from plotly import tools
from plotly.offline import iplot
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM, GRU
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


# ## 載入資料

# 使用 2017-01-01 到 2019-2-28 的「台灣電力公司_過去電力供需資訊」結合「政府行政機關辦公日曆表」中的例假日資料。
# 
# 資料合併的細節請參考程式碼：`Data Preparation.ipynb`。

# In[2]:


df = pd.read_csv('data/elec_merge.csv')


# In[3]:


df[:10]


# 將日期欄位轉為 datetime 格式。

# In[4]:


df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d')


# 選擇要使用的欄位。

# In[5]:


selected_features = ['尖峰負載(MW)', '淨尖峰供電能力(MW)', '備轉容量(MW)', 
                     '備轉容量率(%)', '工業用電(百萬度)', '民生用電(百萬度)',
                     'isHoliday']


# In[6]:


df = df[['日期'] + selected_features]


# ## 資料視覺化

# 下圖為尖峰負載相對時間的變化，雖然只有兩年的資料，但還是可以觀察到兩年的資料似乎有些相似的特徵。
# 
# 巨觀來看會發現資料有季節性的週期變化，夏季較冬季的數值要來得高。微觀來看的話似乎週末時用電量較工作日來得高。
# 
# 若仔細觀察就會發現，這張折線圖有三個低點，都是分佈在這三年農曆新年的時段，因此可以推測尖峰負載與例假日可能有高度的關聯性。

# In[7]:


trace = go.Scatter(x=df['日期'], y=df['尖峰負載(MW)'])
layout = go.Layout(
    title = '尖峰負載相對於時間的變化',
    xaxis = dict(title = '日期'),
    yaxis = dict(title = '尖峰負載')
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# 若只觀察 2017 年 1 月的資料，可以很清楚地觀察的這個週期。
# 
# 例如：1/7 與 1/8 是週末，尖峰負載大幅降低，但 1/9 是工作日，尖峰負載又大幅上升。

# In[8]:


time_range = pd.date_range('2017-01-01', periods=28, freq='D')
trace = go.Scatter(x=time_range, y=df['尖峰負載(MW)'])
layout = go.Layout(
    title = '尖峰負載相對於時間的變化-2017年1月',
    xaxis = dict(title = '日期'),
    yaxis = dict(title = '尖峰負載')
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# 使用 2017 與 2018 的資料當作訓練集，2019 的資料當作測試集。

# In[9]:


split_date = pd.Timestamp('2018-12-29')
train = df.loc[df['日期'] <= split_date]
test = df.loc[df['日期'] > split_date]
trace_train = go.Scatter(x=train['日期'], y=train['尖峰負載(MW)'], name='train')
trace_test = go.Scatter(x=test['日期'], y=test['尖峰負載(MW)'], name='test')
layout = go.Layout(
    title = '尖峰負載相對於時間的變化',
    xaxis = dict(title = '日期'),
    yaxis = dict(title = '尖峰負載')
)
fig = go.Figure(data=[trace_train, trace_test], layout=layout)
iplot(fig)


# ### 觀察每個特徵相對於時間的關係

# In[10]:


trace1 = go.Scatter(
    x = df['日期'],
    y = df[selected_features[0]],
    name = selected_features[0]
)
trace2 = go.Scatter(
    x = df['日期'],
    y = df[selected_features[1]],
    name = selected_features[1]
)
trace3 = go.Scatter(
    x = df['日期'],
    y = df[selected_features[2]],
    name = selected_features[2]
)
trace4 = go.Scatter(
    x = df['日期'],
    y = df[selected_features[3]],
    name = selected_features[3]
)
trace5 = go.Scatter(
    x = df['日期'],
    y = df[selected_features[4]],
    name = selected_features[4]
)
trace6 = go.Scatter(
    x = df['日期'],
    y = df[selected_features[5]],
    name = selected_features[5]
)


# In[11]:


fig = tools.make_subplots(rows=len(selected_features)-1, cols=1, subplot_titles=(selected_features[:-1]))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 4, 1)
fig.append_trace(trace5, 5, 1)
fig.append_trace(trace6, 6, 1)


# In[12]:


fig['layout'].update(height=2500, width=1000, title='Features')
iplot(fig, filename='stacked-subplots')


# ### 分析假日與尖峰負載的關係

# In[13]:


df[df['isHoliday'] == 1]['尖峰負載(MW)'].describe()


# In[14]:


df[df['isHoliday'] == 0]['尖峰負載(MW)'].describe()


# 尖峰負載在假日與非假日的分布。
# 
# 可以觀察到假日的尖峰負載明顯低於非假日。

# In[15]:


trace1 = go.Box(y = df[df['isHoliday'] == 1]['尖峰負載(MW)'], name = 'Holiday')
trace2 = go.Box(y = df[df['isHoliday'] == 0]['尖峰負載(MW)'], name = 'Not Holiday')
layout = go.Layout(
    title = '尖峰負載在假日與非假日的數值分佈',
    yaxis = dict(title = '尖峰負載')
)
fig = go.Figure(data=[trace1, trace2], layout=layout)
iplot(fig)


# ### 處理 `Holiday` 欄位

# 因為要預測的是未來 7 天的數值，所以將 `isHoliday` 欄位左移 7 天，並手動補足剩餘欄位的值。

# In[16]:


df['isHoliday'] = df['isHoliday'].shift(-7)


# In[17]:


df.loc[782:785, ("isHoliday")] = 1


# In[18]:


df.loc[785:, ("isHoliday")] = 0


# In[19]:


df[-10:]


# ### 由於無法取得 3 月份的部分欄位，因此只採用可取得的欄位

# In[20]:


selected_features = ['尖峰負載(MW)', '備轉容量(MW)', 'isHoliday']


# ### 將資料轉為 `Numpy` 的格式，方便放入模型計算

# In[21]:


raw_data = df[selected_features].values


# ### 處理預測的資料，使用 3/24~3/30 的資料來預測 3/31~4/6 的尖峰負載。

# In[22]:


last_week = [[24812.0, 1859.0, 1.0],
             [28535.0, 1853.0, 0.0],
             [28756.0, 1887.0, 0.0],
             [29140.0, 1933.0, 0.0],
             [30093.0, 1892.0, 1.0],
             [29673.0, 2054.0, 1.0],
             [25810.0, 2155.0, 1.0],]


# In[23]:


last_week = np.array(last_week)
last_week = last_week.reshape((1, last_week.shape[0], last_week.shape[1]))


# In[24]:


this_week = [[24466.0, 2298.0, 1.0],
             [28300.0, 1870.0, 0.0],
             [28700.0, 1860.0, 0.0],
             [28600.0, 1960.0, 0.0],
             [25700.0, 2440.0, 0.0],
             [24600.0, 2460.0, 0.0],
             [24300.0, 2670.0, 1.0],]


# In[25]:


this_week = np.array(this_week)
this_week = this_week.reshape((1, this_week.shape[0], this_week.shape[1]))


# ## 模型訓練：Encoder-Decoder LSTM with Multivariate Input

# ### 函式：將 multivariate 資料集切成 train set 與 test set

# In[26]:


def split_dataset(data, split_num=728):
    # split into standard weeks
    train, test = data[:split_num], data[split_num:-5]
    # restructure into windows of weekly data
    train = np.array(np.split(train, len(train) / 7))
    test = np.array(np.split(test, len(test) / 7))
    return train, test


# ### 函式：計算 RMSE

# In[27]:


def evaluate_forecasts(actual, predicted):
    scores = []
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = math.sqrt(mse)
        scores.append(rmse)
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# In[28]:


def summarize_socres(name, score, scores):
    s_scores = ', '.join(['{:.1f}'.format(s) for s in scores])
    print('{}: [{:.3f}] {}'.format(name, score, s_scores))


# ### 函式：將資料轉為 input (X) 與 output (y)

# In[29]:


def to_supervised(train, n_input, n_out=7):
    # flatten the data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end < len(data):
            x_input = data[in_start:in_end, ]
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)


# ### 函式：Normalization (將數值限制在 -1 與 1 之間)

# In[30]:


def scale(train, test, pred1, pred2):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    train = train.reshape(train.shape[0], train.shape[1] * train.shape[2])
    test = test.reshape(test.shape[0], test.shape[1] * test.shape[2])
    pred1 = pred1.reshape(pred1.shape[0], pred1.shape[1] * pred1.shape[2])
    pred2 = pred2.reshape(pred2.shape[0], pred2.shape[1] * pred2.shape[2])
    
    scaler = scaler.fit(np.concatenate((train, test, pred1, pred2)))
    
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    
    # transform pred
    pred1 = pred1.reshape(pred1.shape[0], pred1.shape[1])
    pred1_scaled = scaler.transform(pred1)
    
    pred2 = pred2.reshape(pred2.shape[0], pred2.shape[1])
    pred2_scaled = scaler.transform(pred2)
    
    return scaler, train_scaled, test_scaled, pred1_scaled, pred2_scaled


# ### 函式：建立模型

# In[31]:


# train the model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 1, 100, 50
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# ### 函式：預測

# In[32]:


def forecast(model, history, n_input):
    # flatten the data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    input_x = data[-n_input:, :]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    yhat = model.predict(input_x, verbose=1)
    yhat = yhat[0]
    return yhat


# ### 函式：模型評估

# In[33]:


def evaluate_model(model, train, test, n_input):
    history = [x for x in train]
    predictions = []
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])
        
    t = scaler.inverse_transform(test.reshape((test.shape[0], test.shape[1] * test.shape[2])))
    t = t.reshape(t.shape[0], train.shape[1], t.shape[1] // train.shape[1])[:,:,0]
    
    predictions = np.array(predictions)
    predictions = np.concatenate((predictions, np.zeros((predictions.shape[0], train.shape[1], train.shape[2] - 1))),
                                 axis=2)
    predictions = predictions.reshape((predictions.shape[0], predictions.shape[1] * predictions.shape[2]))
    p = scaler.inverse_transform(predictions)
    p = p.reshape(p.shape[0], train.shape[1], p.shape[1] // train.shape[1])[:,:,0]
    
    score, scores = evaluate_forecasts(t, p)
    return score, scores, t, p


# In[34]:


train, test = split_dataset(raw_data, 728)


# In[35]:


scaler, train, test, last_week, this_week = scale(train, test, last_week, this_week)


# In[36]:


train = train.reshape((train.shape[0], train.shape[1] // raw_data.shape[1], raw_data.shape[1]))

test = test.reshape((test.shape[0], test.shape[1] // raw_data.shape[1], raw_data.shape[1]))

last_week = last_week.reshape((last_week.shape[0], last_week.shape[1] // raw_data.shape[1], raw_data.shape[1]))

this_week = this_week.reshape((this_week.shape[0], this_week.shape[1] // raw_data.shape[1], raw_data.shape[1]))


# 使用前 7 天的數值來進行預測未來一週的數值。

# In[37]:


n_input = 7


# ### 模型建構與訓練

# In[38]:


model = build_model(train, n_input)


# ### 模型架構

# In[39]:


model.summary()


# ### 預測

# In[40]:


score, scores, truth, pred = evaluate_model(model, train, test, n_input)


# ### 計算 RMSE

# In[41]:


summarize_socres('RMSE', score, scores)


# In[42]:


truth = truth.reshape((truth.shape[0] * truth.shape[1]))
pred = pred.reshape((pred.shape[0] * pred.shape[1]) )


# In[43]:


time_range = pd.date_range('2018-12-30', periods=truth.shape[0], freq='D')


# In[44]:


ground_truth = go.Scatter(x=time_range, y=truth, name='truth')
predict_answer = go.Scatter(x=time_range, y=pred, name='predict')
layout = go.Layout(
    title = '預測值 vs 實際值',
    xaxis = dict(title = '日期'),
    yaxis = dict(title = '尖峰負載')
)
fig = go.Figure(data=[ground_truth, predict_answer], layout=layout)
iplot(fig)


# ## 預測 4/2 ~ 4/8 的尖峰負載

# 使用 3/26~4/1 的資料來預測 4/2~4/8 的尖峰負載。

# In[45]:


last_week.shape


# In[46]:


def forecast(model, data, n_input):
    yhat = model.predict(data, verbose=1)
    yhat = yhat[0]
    return yhat


# In[47]:


prediction = forecast(model, last_week, 7)


# In[48]:


prediction = prediction.reshape((1, prediction.shape[0], prediction.shape[1]))
prediction = np.concatenate((prediction, np.zeros((prediction.shape[0], train.shape[1], train.shape[2] - 1))),
                             axis=2)
prediction = prediction.reshape((prediction.shape[0], prediction.shape[1] * prediction.shape[2]))
p = scaler.inverse_transform(prediction)
p = p.reshape(p.shape[0], train.shape[1], p.shape[1] // train.shape[1])[:,:,0]


# In[49]:


p1 = p.reshape((7,))


# In[50]:


p1


# 3/31~4/6

# In[51]:


prediction = forecast(model, this_week, 7)


# In[52]:


prediction = prediction.reshape((1, prediction.shape[0], prediction.shape[1]))
prediction = np.concatenate((prediction, np.zeros((prediction.shape[0], train.shape[1], train.shape[2] - 1))),
                             axis=2)
prediction = prediction.reshape((prediction.shape[0], prediction.shape[1] * prediction.shape[2]))
p = scaler.inverse_transform(prediction)
p = p.reshape(p.shape[0], train.shape[1], p.shape[1] // train.shape[1])[:,:,0]


# In[53]:


p2 = p.reshape((7,))


# In[54]:


p2


# In[55]:


ans = np.array([28700, 28600, 25700, 24600, 24300, 24500, 28500])


# In[57]:


my_pred = np.concatenate((p1[2:],p2[:2]))


# In[59]:


math.sqrt(mean_squared_error(my_pred, ans))


# In[62]:


dates = ['20190402', '20190403', '20190404', '20190405', '20190406', '20190407', '20190408']


# In[64]:


with open('submission.csv', mode='w') as csv_file:
    csv_file.write('date,peak_load(MW)\n')
    for d, a in zip(dates, ans):
        csv_file.write(d + ',' + str(a) + '\n')

