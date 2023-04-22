#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Iris.csv")
df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df['Species'].value_counts()


# In[7]:


df.shape


# In[8]:


df['SepalLengthCm'].hist()


# In[9]:


from pandas.api.types import is_numeric_dtype
for col in df.columns:
 if is_numeric_dtype(df[col]):
     print('%s:' % (col))
     print('\t Mean = %.2f' % df[col].mean())
     print('\t Standard deviation = %.2f' % df[col].std())
     print('\t Minimum = %.2f' % df[col].min())
     print('\t Maximum = %.2f' % df[col].max())


# In[12]:


print('Covariance:')
df.cov()


# In[13]:


df.boxplot()


# In[14]:


color = ['red','Orange','Blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
for i in range(3):
 x = df[df['Species'] == species[i]]
 plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = color[i], label = species[i]) 
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[21]:


corr = df.corr()
fig, ax = plt.subplots(figsize = (8,6))
sns.heatmap(corr, annot = True, ax = ax)


# In[22]:


x = df.drop("Species", axis=1)
y = df["Species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)


# In[23]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)


# In[25]:


x_new = np.array([[99, 2.9, 1, 0.2,2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))

