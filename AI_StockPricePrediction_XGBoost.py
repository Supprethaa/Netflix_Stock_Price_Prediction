#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install xgboost')


# # Installing required libraries

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


# # Loading the data and pre-processing

# In[4]:


data1=pd.read_csv(r'C:\Users\Acer\Documents\AI-COMP5313\NFLX.csv')
data1.head(10)


# In[6]:


new_data=data1[['Date','Close']]
new_data.head(10)


# In[7]:


new_data['Date'] = pd.to_datetime(new_data['Date'])
new_data.set_index('Date', inplace=True)


# In[8]:


new_data


# # Creating new data by getting past 10 days values for each date

# In[9]:


# Create lag features
for i in range(1, 11):  # Using lag values for the last 10 days
    new_data[f'Close_Lag_{i}'] = new_data['Close'].shift(i)


# In[10]:


new_data


# In[11]:


# Drop rows with NaN values due to lag
new_data.dropna(inplace=True)


# In[12]:


new_data


# # Splitting training and Test data (80:20 ratio)

# In[30]:


# Split the data into training and testing sets
train_size = int(len(new_data) * 0.8)
train, test = new_data[:train_size], new_data[train_size:]


# In[31]:


# Separate features (X) and target variable (y)
X_train, y_train = train.drop('Close', axis=1), train['Close']
X_test, y_test = test.drop('Close', axis=1), test['Close']


# # Building the XGBoost model

# In[32]:


# Initialize the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)


# In[33]:


# Fit the model
xgb_model.fit(X_train, y_train)


# In[34]:


# Make predictions on the test set
y_pred = xgb_model.predict(X_test)


# In[35]:


# # Inverse transform predictions and actual values to original scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(new_data[['Close']])
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.values.reshape(-1, 1))


# In[36]:


# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, y_test_inv, label='Actual Values')
plt.plot(test.index, y_pred_inv, label='Predicted Values (XGBoost)')
plt.title('XGBoost Model - Actual vs. Predicted Values')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


# In[37]:


from sklearn.metrics import mean_squared_error
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_inv, y_pred_inv)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print(f'RMSE: {rmse}')


# In[38]:


# Calculate Absolute Error (AE)
absolute_error = np.abs(y_test_inv - y_pred_inv)

# Calculate Mean Absolute Error (MAE)
mean_absolute_error = np.mean(absolute_error)

print(f'Mean Absolute Error (MAE): {mean_absolute_error}')


# In[ ]:




