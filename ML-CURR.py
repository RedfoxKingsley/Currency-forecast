#!/usr/bin/env python
# coding: utf-8

# In[1340]:


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


# In[1341]:


print(df)


# In[1342]:


df.isnull().values.any()


# In[1343]:


#### Drop cells with NaN
df = df.dropna(axis=0,subset=['USDNGN'])
df = df.dropna(axis=0,subset=['M2'])
df = df.dropna(axis=0,subset=['CPI'])


# In[1344]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape


# In[1347]:


# Show rows where USDNGN is NaN
df[df['USDNGN'].isnull()].shape


# In[1348]:


# Show rows where M2 is NaN
df[df['M2'].isnull()].shape


# In[1349]:


# Show rows where CPI is NaN
df[df['CPI'].isnull()].shape


# In[1350]:


df.describe()


# In[1351]:


# Seaborn doesn't handle NaN values, so we can fill them with 0 for now.
df = df.fillna(value=0)

# Pair grid of key variables.
g = sns.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Pairwise Grid of Numeric Features');


# In[1352]:


g = sns.FacetGrid(df, col='Month', col_wrap=4)
g.map(sns.distplot, "USDNGN")
plt.show()


# In[1353]:


corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap='Blues')
plt.title('Correlation Heatmap of Numeric Features')
#The very light and very dark boxes show a strong positive or negative correlation between the features. A positive correlation would be that as the USDNGN exchange rate increases (the currency depreciates) the other feature increases as well. A negative correlation is the opposite. As the exchange rate rises the other features decreases.
#We are particularly interested in correlations to the USD feature. M2 and Inflation and currency have strong correlations but does not prove a causal relationship.


# In[1354]:


#Make final dataset
df.columns


# In[1355]:


cols = df.columns.tolist()


# In[1356]:


df.to_csv('m+fdata.csv')


# In[1357]:


df.dtypes


# In[1358]:


#Split into Training and Test Data
#Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

y = df['USDNGN']


# In[1359]:


X = df.drop(['USDNGN'], axis=1)


# In[1360]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)


# In[1361]:


X_train.shape, y_train.shape


# In[1362]:


X_test.shape, y_test.shape


# In[1363]:


X.columns


# In[1364]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Create linear regression object
regr = LinearRegression()


# In[1365]:


# Train the model using the training sets
regr.fit(X_train, y_train)


# In[1366]:


# Make predictions using the testing set
lin_pred = regr.predict(X_test)


# In[1367]:


linear_regression_score = regr.score(X_test, y_test)
linear_regression_score


# In[1417]:


linear_regression_score = regr.score(X_train, y_train)
linear_regression_score


# In[1368]:


from math import sqrt
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lin_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lin_pred))


# In[1369]:


plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()


# In[1370]:


### Neural Network Regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create MLPRegressor object
mlp = MLPRegressor()


# In[1371]:


# Train the model using the training sets
mlp.fit(X_train, y_train)


# In[1372]:


# Score the model
neural_network_regression_score = mlp.score(X_test, y_test)
neural_network_regression_score


# In[1418]:


# Score the model
neural_network_regression_score = mlp.score(X_train, y_train)
neural_network_regression_score


# In[1373]:


# Make predictions using the testing set
nnr_pred = mlp.predict(X_test)


# In[1374]:


# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, nnr_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, nnr_pred))


# In[1376]:


plt.scatter(y_test, nnr_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Neural Network Regression Predicted vs Actual')
plt.show()


# In[1377]:


###Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()


# In[1378]:


lasso.fit(X_train, y_train)


# In[1380]:


# Score the model
lasso_score = lasso.score(X_test, y_test)
lasso_score


# In[1419]:


# Score the model
lasso_score = lasso.score(X_train, y_train)
lasso_score


# In[1381]:


# Make predictions using the testing set
lasso_pred = lasso.predict(X_test)


# In[1421]:


print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))

# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lasso_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lasso_pred))


# In[1383]:


plt.scatter(y_test, lasso_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Lasso Predicted vs Actual')
plt.show()


# In[1384]:


##ElasticNet
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)


# In[1385]:


elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score


# In[1420]:


elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score


# In[1386]:


elasticnet_pred = elasticnet.predict(X_test)


# In[1422]:


# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, elasticnet_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, elasticnet_pred))


# In[1416]:


#Evaluate Models
print("Scores:")
print("Linear regression score: ", linear_regression_score)
print("Neural network regression score: ", neural_network_regression_score)
print("Lasso regression score: ", lasso_score)
print("ElasticNet regression score: ", elasticnet_score)
print("\n")
print("RMSE:")
print("Linear regression RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
print("Neural network RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
print("Lasso RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))
print("ElasticNet RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))

