import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from datetime import date
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st

# start=datetime(2020,12,1)
# end=datetime(2022,12,15)

yf.pdr_override()
st.title('Stock Trend prediction')

# st.write('enter the range for which you want to visualize and forecast the stock')
start=st.text_input('enter the starting date','2010-11-01')
end=st.text_input('enter the ending date','2022-12-12')
# start='2010-11-01'
# end='2022-12-12'

user_input=st.text_input('Enter Stock Ticker','AAPL')
df=pdr.get_data_yahoo(user_input,start,end)




st.subheader('Data for the stock are as follows')

st.write(df.describe())

# visualizations
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='100 days MA')
plt.plot(ma200,'g',label='200 days MA')
plt.plot(df.Close,'b',label='Closing price')
plt.legend()
st.pyplot(fig)


# #split data into training and testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


data_training_array=scaler.fit_transform(data_training)
# data_training_array


# x_train=[]
# y_train=[]
# # # # for i in range(100,data_training_array.shape[0]):
# # # #     x_train=np.append(x_train,data_training_array[i-100:i])#changed
# # # #     y_train=np.append(y_train,data_training_array[i,0])
# # # #  x_train,y_train=np.array(x_train),np.array(y_train)
# # # # x_train=[]
# # # # y_train=[] 

# for i in range(100,data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])
# x_train=np.array(x_train)
# y_train=np.array(y_train)

# # # Load my model
model=load_model('keras_model.h5')

# st.write(model.describe)

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])



x_test=np.array(x_test)
y_test=np.array(y_test)

# # st.write(x_test.shape)

y_pred=model.predict(x_test)

# # st.write(y_pred.shape)

scaler=scaler.scale_

scale_factor=1/scaler[0]
y_pred=y_pred*scale_factor
y_test=y_test*scale_factor



# #final graph
st.subheader('Predicted vs original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)





