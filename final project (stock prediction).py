import numpy as np               
import matplotlib.pyplot as plt 
import pandas as pd             
import pandas_datareader as pdr  
import datetime as dt

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler




# Step 1: Import data

business = input("enter company ticker name: ")
business = business.upper()

start_date = dt.datetime(2019,1,1)
end_date = dt.datetime(2022,1,1)

stock_data = pdr.DataReader(business,'yahoo',start_date,end_date)


# Step 2: Preparation
sc = MinMaxScaler(feature_range=(0,1))
sc_data = sc.fit_transform(stock_data['Adj Close'].values.reshape(-1,1))

days_of_prediction = int(input("enter days of prediction: ")) 

A = []
B = []


for i in range(days_of_prediction,len(sc_data)):
    B.append(sc_data[i,0])
    A.append(sc_data[i-days_of_prediction:i,0])
    

A,B = np.array(A),np.array(B)
A = np.reshape(A,(A.shape[0],A.shape[1],1))



# Step 3: Model construction
prediction_model = Sequential()

prediction_model.add(LSTM(units = 30, return_sequences = True, input_shape=(A.shape[1],1)))
prediction_model.add(Dropout(0.2))
prediction_model.add(LSTM(units = 30, activation = "relu"))
prediction_model.add(Dense(units=1)) 

prediction_model.compile(optimizer = "adam",loss = "mean_squared_error")
prediction_model.fit(A, B, epochs=20,batch_size=30)



# Step 4: Import testing data
test_start_date = dt.datetime(2022,1,1)
test_end_date = dt.datetime.now()

test_stock_data = pdr.DataReader(business, "yahoo", test_start_date,test_end_date)
actual_prices = test_stock_data["Adj Close"].values

total_dataset = pd.concat((stock_data['Adj Close'], test_stock_data['Adj Close']),axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_stock_data)-days_of_prediction:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = sc.transform(model_inputs)


# Step 5: Testing data prediction
test = []

for j in range(days_of_prediction,len(model_inputs)):
    test.append(model_inputs[j-days_of_prediction:j,0])

test = np.array(test)
test = np.reshape(test, (test.shape[0],test.shape[1],1))

price_predicted = prediction_model.predict(test)
price_predicted = sc.inverse_transform(price_predicted)


# Step 6: Graph
plt.plot(actual_prices,color = "black",label="Actual {} Adjusted Close Price".format(business))
plt.plot(price_predicted,color = "blue",label="Predicted {} Adjusted Close Price".format(business))
plt.title("{} Share Adjusted Close Price".format(business))
plt.xlabel("Time")
plt.ylabel("Adjusted closed price")
plt.legend()
plt.show()

