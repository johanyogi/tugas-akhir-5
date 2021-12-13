import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)

from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import streamlit as st

import warnings 

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Predict with RNN Model','Predict with LSTM Model')
)


Data = st.file_uploader("Choose a file")
if Data is not None:
    df = pd.read_excel(Data)
    st.write(df)


    df = df.set_index(['Tanggal'])
    df.head()

    scaler = MinMaxScaler(feature_range = (0, 1))
    df_scaled = scaler.fit_transform(df)
    df.dropna()

    step_size = 7                 
    x_train = []
    y_train = []

    for i in range(step_size,len(df)):                
        x_train.append(df_scaled[i-step_size:i,0])
        y_train.append(df_scaled[i,0])

    x_train = np.array(x_train)                  
    y_train = np.array(y_train)

if option == 'Home' or option == '':
    st.write("""# Halaman Utama""") #menampilkan halaman utama

elif option == 'Predict with RNN Model':
    st.write("""Prediksi data dengan model RNN""")
    x_train, x_test, y_train, y_test  = train_test_split(x_train,y_train,test_size=0.2, shuffle=False)

    x_train = np.reshape(x_train, (len(x_train), step_size, 1))           
    x_test = np.reshape(x_test, (len(x_test), step_size, 1))

    rnn_model = Sequential()

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True, input_shape=(x_train.shape[1],1)))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(Dense(1))

    rnn_model.compile(optimizer="adam",loss="MSE")

    rnn_model.fit(x_train,y_train,epochs=20,batch_size=25)

    rnn_pred = rnn_model.predict(x_test)
    rnn_score = r2_score(y_test,rnn_pred)

    df1=df.copy()
    df1.drop(df1.index[0:len(x_train)], inplace=True)
    df1.drop(df1.index[-step_size:], inplace=True)
    dfrnn=df1.copy()
    true_predictions_rnn = scaler.inverse_transform(rnn_pred)
    dfrnn['RNN']=true_predictions_rnn

    st.pyplot(df1,color)
    st.pyplot(dfrnn['RNN'])
    plt.legend (loc='best')
    plt.title("Perbandingan hasil prediksi model RNN dengan data Original",fontsize=16)
    plt.xlabel("Tanggal",fontsize=16)
    plt.ylabel("Kasus Harian",fontsize=16)
    plt.show()

    plt.plot(df,color='blue',label='Kasus Harian')
    plt.plot(dfrnn['RNN'],color='red',label='Prediksi RNN (R² Score : %.4f)'%rnn_score)
    plt.legend (loc='best')
    plt.title("Perbandingan hasil prediksi model RNN dengan data Original",fontsize=16)
    plt.xlabel("Tanggal",fontsize=16)
    plt.ylabel("Kasus Harian",fontsize=16)
    plt.show()