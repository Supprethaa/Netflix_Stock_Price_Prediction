#!/usr/bin/env python
# coding: utf-8

# # Importing Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Loading the data and basic visualizations

# In[2]:


data1=pd.read_csv(r'C:\Users\Acer\Documents\AI-COMP5313\NFLX.csv')
data1.head(10)


# In[3]:


data1.info()


# In[4]:


data1.describe()


# In[23]:


# Visualizing The Data
plt.figure(figsize=(12, 5), dpi=200)

plt.title("Netflix Stock Market")
plt.xlabel('Date')
plt.ylabel('Prices')

plt.plot(data1['Date'],data1['Open'], label='Open', color='green')
plt.plot(data1['Date'],data1['Close'], label='Close', color='red')

plt.legend()
plt.show()


# In[24]:


#Histogram of opening and Closing Prices
plt.figure(figsize=(10, 6), dpi=200)
data1[['Open', 'Close']].plot(kind='hist', bins=20, alpha=0.7)
plt.title('Distribution of Opening and Closing Prices')
plt.xlabel('Price')
plt.show()


# In[ ]:


# Box plot for opening and closing prices

plt.figure(figsize=(10, 6))
data1[['Open', 'Close']].boxplot()
plt.title('Box Plot for Opening and Closing Prices')
plt.show()


# # Filtering only the Date and Close price Columns 

# In[5]:


new_data=data1[['Date','Close']]
new_data.head(10)


# In[6]:


new_data['Date'] = pd.to_datetime(new_data['Date'])
new_data.set_index('Date', inplace=True)


# In[7]:


new_data


# # Importing Libraries to scale the data and LSTM model building

# In[26]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[30]:


class StockPricePredictor:
    def __init__(self, df):
        self.df = df
        self.sequence_length = 10
        self.scaler = MinMaxScaler(feature_range=(0, 1))   #Scaling the data values between 0 to 1
        self.model = self._build_model()

    def _preprocess_data(self):
#         self.df['Date'] = pd.to_datetime(self.df['Date'])
#         self.df.set_index('Date', inplace=True)
        closing_prices = self.df['Close'].values.reshape(-1, 1)
        self.closing_prices_scaled = self.scaler.fit_transform(closing_prices)    #Train-test split (80:20 ratio)
        X, y = self._create_sequences(self.closing_prices_scaled)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), 0])   #Create historical sequences for the LSTM model
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(50, input_shape=(self.sequence_length, 1)))  #Building the LSTM model
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, epochs=50, batch_size=32):
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test))
        self._plot_loss(history)    #Method to train the model

    def _plot_loss(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')        #Plotting model loss over each time step
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.show()

    def evaluate_model(self):
        predictions_scaled = self.model.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))   #Evaluate the model by plotting difference between
                                                                                    #Actual and Predicted values

        plt.figure(figsize=(10, 6))
        plt.plot(y_test_actual, label='Actual Values')
        plt.plot(predictions, label='Predicted Values')
        plt.title('Actual vs. Predicted Values for Test Set')
        plt.xlabel('Time Step')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.show()

        differences = y_test_actual - predictions
        plt.figure(figsize=(10, 6))
        plt.plot(differences, label='Difference (Actual - Predicted)')
        plt.title('Difference Between Actual and Predicted Values for Test Set')
        plt.xlabel('Time Step')
        plt.ylabel('Difference')
        plt.legend()
        plt.show()


# In[31]:


# Replace with the actual path to your dataset
predictor = StockPricePredictor(new_data)
predictor._preprocess_data()
predictor.train_model()


# In[34]:


predictor.evaluate_model()


# # Calculating the Regression errors

# In[35]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_actual, predictions)

        # Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

        # Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_actual, predictions)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')


# # Making future predictions using learnt trend 

# In[32]:


# Make predictions for future dates
num_predictions = 10
future_dates = pd.date_range(start=new_data.index[-1], periods=num_predictions, freq='D')

# Initialize the last sequence with the last known closing prices
last_sequence = closing_prices[-sequence_length:]

# List to store future predictions
future_predictions = []

# Loop to make predictions for each future date
for _ in range(num_predictions):
    # Normalize the last sequence of closing prices
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))

    # Create input sequence for future predictions
    future_sequence = np.array([last_sequence_scaled])

    # Reshape input data for LSTM model
    future_sequence = np.reshape(future_sequence, (future_sequence.shape[0], future_sequence.shape[1], 1))

    # Make future predictions
    future_predictions_scaled = model.predict(future_sequence)

    # Inverse transform the scaled future predictions
    future_prediction = scaler.inverse_transform(future_predictions_scaled)

    # Append the prediction to the list
    future_predictions.append(future_prediction[0, 0])  # Assuming the prediction is a single value

    # Update the last sequence for the next iteration
    last_sequence = np.append(last_sequence[1:], future_prediction[0, 0])

# Create a DataFrame to store and display the future predictions
future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Future Predictions'])
print(future_df)


# In[ ]:




