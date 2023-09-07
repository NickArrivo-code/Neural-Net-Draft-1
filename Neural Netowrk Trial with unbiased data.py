#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:01:09 2023

@author: nickarrivo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:24:26 2023

@author: nickarrivo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt



# Load the dataset
data = pd.read_csv('/Users/nickarrivo/MLB-Hitter-Ratings/Datasets/Improved Vertical Merge Composite Score w stats.csv')

scaler = StandardScaler()

#Which columns do I want to use to train the model
#Power score, contact score, discipline score
X = data[['power_score', 'contact_score', 'discipline_score', 'composite_score_scaled']]
y = data['on_base_plus_slg']


# Drop non-numeric columns
X_numeric = X.select_dtypes(exclude=['object'])

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)
X_normalized = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape


# Define the neural network architecture
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),  # Input layer
    Dense(64, activation='relu'),                    # Hidden layer 1
    Dense(32, activation='relu'),                    # Hidden layer 2
    Dense(1)                                         # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=75, validation_data=(X_test, y_test))


loss = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Data: {loss}")

predictions = model.predict(X_test)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')  # Diagonal line
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()
