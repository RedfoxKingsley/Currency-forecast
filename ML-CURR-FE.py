#!/usr/bin/env python
# coding: utf-8

# In[210]:


#Setting It Up
#I collected all of the data above and combined them into one dataframe. The code and details are located here. One challenge was the periodicity of the various features. Our exchange data is daily, some data is monthly, and others quarterly. For our daily exchange rates, I took the last value of each month. For the quarterly data, I copied the quarterly value to each month in that quarter. This gives us a dataframe of monthly data that is easier to work with.
#First, we will import the libraries we will be using and also load our data into a Pandas dataframe.

# Import needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn

# Python magic to show plots inline in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import datetime as dt
from datetime import datetime
import math
# Import data
df = pd.read_csv("C:/Users/Kingsley/Desktop/m+fdata.csv")
# set index of dataframe to Date feature. 
df = df.set_index('Date')


# In[211]:


print(df)


# In[212]:


#FEATURE ENGINEERING
#Feature Engineering
#Currently our dataframe isn’t exactly what we need. This is because our machine learning algorithms only learn row by row and aren’t aware of other rows when learning or making predictions. We can overcome this challenge by imputing previous time value, or lags, into our data.
#After some trial and error, I determined that 4 lags (or 4 months) work best. The code below creates a function that will create 4 lags for each feature in the ‘features’ list. Our new dataframe now has 41 columns!

# Define custom function to create lag values
def feature_lag(features):
    for feature in features:
        df[feature + '-lag1'] = df[feature].shift(1)
        df[feature + '-lag2'] = df[feature].shift(2)
        df[feature + '-lag3'] = df[feature].shift(3)
        df[feature + '-lag4'] = df[feature].shift(4)

# Define columns to create lags for
features = ['USDNGN', 'M2', 'CPI', 'Month']

# Call custom function
feature_lag(features)


# In[213]:


#You can see that “lag1” has the previous day/months value, “lag2” and two days/months previous, and so on. This gives the algorithm some knowledge of the previous value.
#Next we can add the year-month and day into the column. This gives the model a “sense” of time 'reality' over abstract numbers.
#df['USDNGN'] = df.USDNGN.fillna(method='ffill')
#df['USDNGN_1'] = df.USDNGN_1.fillna(method='ffill')
#df['USDNGN_2'] = df.USDNGN_2.fillna(method='ffill')
#df['USDNGN_3'] = df.USDNGN_3.fillna(method='ffill')
#df['CPI'] = df.CPI.fillna(value=0)
#df['CPI_1'] = df.CPI_1.fillna(value=0)
#df['CPI_2'] = df.CPI_2.fillna(value=0)
#df['CPI_3'] = df.CPI_3.fillna(value=0)
df.describe()


# In[214]:


#Finally, we can take the difference between the current and lag exchange rates. By itself the algorithm isn’t able to reach these sorts of conclusions, so we can calculate this and add it to our dataframe as well.

df['USDNGN-lag1-diff'] = df['USDNGN'] - df['USDNGN-lag1']
df['USDNGN-lag2-diff'] = df['USDNGN-lag1'] - df['USDNGN-lag2']
df['USDNGN-lag3-diff'] = df['USDNGN-lag2'] - df['USDNGN-lag3']


# In[215]:


df['y3'] = df.USDNGN.shift(-3)
df['y6'] = df.USDNGN.shift(-6)
df['y12'] = df.USDNGN.shift(-12)


# In[217]:


df.isnull().values.any()


# In[218]:


df = df.dropna(axis=0,subset=['USDNGN', 'Year', 'Month', 'M2', 'CPI', 'USDNGN-lag1', 'USDNGN-lag2',
       'USDNGN-lag3', 'USDNGN-lag4', 'M2-lag1', 'M2-lag2', 'M2-lag3',
       'M2-lag4', 'CPI-lag1', 'CPI-lag2', 'CPI-lag3', 'CPI-lag4', 'Month-lag1',
       'Month-lag2', 'Month-lag3', 'Month-lag4', 'USDNGN-lag1-diff',
       'USDNGN-lag2-diff', 'USDNGN-lag3-diff', 'y3', 'y6', 'y12'])


# In[219]:


print(df)


# In[220]:


corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap='Blues')
plt.title('Correlation Heatmap of Numeric Features')

#The very light and very dark boxes show a strong positive or negative correlation between the features. A positive correlation would be that as the USDNGN exchange rate increases (the currency depreciates) the other feature increases as well. A negative correlation is the opposite. As the exchange rate rises the other features decreases.
#We are particularly interested in correlations to the USD feature. M2 and Inflation and currency have strong correlations but does not prove a causal relationship.


# In[221]:


#Make final dataset
df.columns


# In[222]:


cols = df.columns.tolist()


# In[223]:


df.to_csv('m+fdata.csv')


# In[224]:


df.dtypes


# In[225]:


#Split into Training and Test Data
#Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

y = df['USDNGN']


# In[226]:


X = df.drop(['USDNGN'], axis=1)


# In[227]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)


# In[228]:


X_train.shape, y_train.shape


# In[229]:


X_test.shape, y_test.shape


# In[230]:


X.columns


# In[231]:


###Decision Forest Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=200, random_state=1234)


# In[258]:


# Train the model using the training sets
regr_rf.fit(X_train, y_train)


# In[259]:


regr_rf.fit(X_test, y_test)


# In[233]:


# Score the model
decision_forest_score = regr_rf.score(X_test, y_test)
decision_forest_score


# In[261]:


decision_forest_score = regr_rf.score(X_train, y_train)
decision_forest_score


# In[234]:


# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)


# In[235]:


from math import sqrt
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_rf_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, regr_rf_pred))


# In[236]:


features = X.columns
importances = regr_rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()


# In[237]:


plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()


# In[238]:


#Extra Trees Regression

from sklearn.ensemble import ExtraTreesRegressor

extra_tree = ExtraTreesRegressor(n_estimators=200, random_state=1234)


# In[239]:


extra_tree.fit(X_train, y_train)


# In[240]:


extratree_score = extra_tree.score(X_test, y_test)
extratree_score


# In[262]:


extratree_score = extra_tree.score(X_train, y_train)
extratree_score


# In[241]:


extratree_pred = extra_tree.predict(X_test)


# In[264]:


print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, extratree_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, extratree_pred))


# In[243]:


features = X.columns
importances = extra_tree.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()


# In[244]:


plt.scatter(y_test, extratree_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Extra Trees Predicted vs Actual')
plt.show()


# In[245]:


#Decision Tree + AdaBoost

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Decision Tree Regressor object
tree_1 = DecisionTreeRegressor()

tree_2 = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=200, learning_rate=.1)


# In[246]:


# Train the model using the training sets
tree_1.fit(X_train, y_train)
tree_2.fit(X_train, y_train)


# In[247]:


# Score the decision tree model
tree_1.score(X_test, y_test)


# In[248]:


# Score the boosted decision tree model
boosted_tree_score = tree_2.score(X_test, y_test)
boosted_tree_score


# In[263]:


boosted_tree_score = tree_2.score(X_train, y_train)
boosted_tree_score


# In[249]:


# Make predictions using the testing set
tree_1_pred = tree_1.predict(X_test)
tree_2_pred = tree_2.predict(X_test)


# In[250]:



# The coefficients

# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, tree_2_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, tree_2_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, tree_2_pred))


# In[251]:


features = X.columns
importances = tree_2.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()


# In[252]:


plt.scatter(y_test, tree_1_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Tree Predicted vs Actual')
plt.show()


# In[253]:


plt.scatter(y_test, tree_2_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Boosted Decision Tree Predicted vs Actual')
plt.show()


# In[254]:


#Evaluate Models
print("Scores:")
print("Decision forest score: ", decision_forest_score)
print("Extra Trees score: ", extratree_score)
print("Boosted decision tree score: ", boosted_tree_score)
print("\n")
print("RMSE:")
print("Decision forest RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
print("Extra Trees RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))
print("Boosted decision tree RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, tree_2_pred)))

