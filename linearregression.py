#linear regression model

#train a linear regression model     with x_train
#test the model with x_test and y_test


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.metrics 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score

file= pd.read_csv('D:/python/linearregression.csv')

diabetes = datasets.load_diabetes()

diabetes_pand= pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(diabetes_pand.head(3))
print(diabetes_pand.shape)

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)    

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients  
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
        % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'  
        % r2_score(diabetes_y_test, diabetes_y_pred))
# The mean absolute error
print('Mean absolute error: %.2f'
        % mean_absolute_error(diabetes_y_test, diabetes_y_pred))
# The mean squared log error    
print('Mean squared log error: %.2f'
        % mean_squared_log_error(diabetes_y_test, diabetes_y_pred)) 
# The median absolute error
print('Median absolute error: %.2f'                 
        % median_absolute_error(diabetes_y_test, diabetes_y_pred))  
# Explained variance score: 1 is perfect prediction
print('Explained variance score: %.2f'          
        % explained_variance_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black', label='data') 
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(( ))
plt.yticks(())
plt.show()




"""














