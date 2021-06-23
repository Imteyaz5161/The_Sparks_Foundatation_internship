#!/usr/bin/env python
# coding: utf-8

# # TASK 1 - Data Science and Business Analytics Internship
# By MD Imteyaz Intern at The Sparks Foundation
# # Task : - Pridiction using Supervised ML
# Description : Pridict the percentage of on student based on the no. of study hours.This is a simple linear regression task as it involves just 2 variables. We used python programming language to pridicted score if a student studies for 9.25hrs/day
# 
# 
# # Importing libraries required

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Importing the dataset
data = 'http://bit.ly/w-data'
df = pd.read_csv(data)
df


# In[30]:


df.shape


# In[31]:


# Finding the datatypes of the columns
df.dtypes


# In[32]:


#Columns present in dataset
df.columns


# In[33]:


# Some basic stats of the dataset
df.describe()


# In[34]:


df.info()


# # visualizing the dataset

# In[35]:



# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[36]:


df.corr()


# In[37]:


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.


# In[38]:


sns.distplot(df.Scores, norm_hist=True)
#This graph depicts a normal distribution


# In[39]:


sns.boxplot(df.Scores)
#This graph depicts that no outliers are present


# # machine learning Application

# In[40]:


#Splitting the dataset
#The first step is to divide the data into "attributes" (inputs) and "labels" (outputs).
X = df.Hours.values
X=X.reshape(-1, 1)
y = df.Scores
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# 
# For Training the Algorithm We have splitted our data into training and testing sets, and now is finally the time to train our algorithm.

# In[41]:


#Applying Linear Regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

m=model.coef_
c=model.intercept_
print(f"Coefficient is {m} and intercept is {c}")


# In[42]:


# Plotting the regression line
line = model.coef_*X+model.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # making predictions

# Now that we have trained our algorithm, it's time to make some predictions.

# In[43]:


print(x_test) # Testing data - In Hours
y_pred = model.predict(x_test) # Predicting the scores
y_pred


# In[44]:


# Comparision
comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
comp


# In[47]:


#Evaluating the model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics  

print('RMSE score:', mean_squared_error(y_test, y_pred))
print('R2 score:', r2_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[46]:


# Final Step
print(f"The predicted score for a student who studies for 9.25 hours/day is {model.predict([[9.25]])}")


# In[ ]:




