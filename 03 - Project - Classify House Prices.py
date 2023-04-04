#!/usr/bin/env python
# coding: utf-8

# # Project: Classify House Prices
# - Put houses in price groups and try to predict based on Latitude and Longitude
# - That will show if the area is a good indicator of the house unit price

# ### Step 1: Import libraries

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score


# ### Step 2: Read the data
# - Use Pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method to read **files/house_prices.csv**

# In[2]:


data = pd.read_csv('files/house_prices.csv')
data.head()


# In[ ]:





# ### Step 3: Prepare data
# - Create 15 bins of house prices
#     - HINT: use [cut](https://pandas.pydata.org/docs/reference/api/pandas.cut.html) on the **'House unit price'** column with **bins=15** and assign the result to column **Class**.
#     - Get the category codes by transforming column **Class** with **.cat.codes** and assign it to **Class id**

# In[3]:


data['Class'] = pd.cut(data['House unit price'], bins=15)
data['Class id'] = data['Class'].cat.codes
data.head()


# In[ ]:





# ### Step 4: Prepare training and test data
# - Assign **X** be all the data (it is needed in final step)
# - Assign **y** to be the **Class id** column.
# - Use **train_test_split** with **test_size=0.15**

# In[4]:


X = data
y = data['Class id']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=42)


# ### Step 5: Train a $k$-Neighbours Classifier
# - Create a model to **KNeighborsClassifier()**
# - Fit the model on **X_train[['Latitude', 'Longitude']]** and **y_train**
# - Predict **X_test[['Latitude', 'Longitude']]** and assign it to **y_pred**
# - Calculate the accuracy score

# In[6]:


model = KNeighborsClassifier()
model.fit(X_train[['Latitude', 'Longitude']], y_train)
y_pred = model.predict(X_test[['Latitude', 'Longitude']])
accuracy_score(y_test, y_pred)


# In[ ]:





# ### Step 6: Make prediction of categories
# - Convert **y_pred** to a DataFrame
#     - HINT: **df_pred = pd.DataFrame(y_pred, columns=['Pred cat'])**
# - Get the middle value of the prediction category.
#     - HINT: **df_pred['Pred'] = df_pred['Pred cat'].apply(lambda x: X_test['Class'].cat.categories[x].mid)**
# - Calculate the **r2_score** of the predicted and real price **'House unit price'** of **X_test**

# In[7]:


df_pred = pd.DataFrame(y_pred, columns=['Pred cat'])
df_pred['Pred'] = df_pred['Pred cat'].apply(lambda x: X_test['Class'].cat.categories[x].mid)


# In[8]:


r2_score(X_test['House unit price'], df_pred['Pred'])


# In[9]:


fig, ax = plt.subplots()

ax.scatter(x=X['Longitude'], y=X['Latitude'], c=data['House unit price'])

