#!/usr/bin/env python
# coding: utf-8

# In[64]:


from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd


# In[16]:


df = pd.read_csv("keystroke.csv")


# In[17]:


df.head()


# In[18]:


df.info()


# In[45]:


df.describe()


# In[19]:


df.columns


# In[20]:


df = df.drop(['subject','sessionIndex', 'rep'],axis=1)


# In[51]:


df.shape


# In[71]:


# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(df)


# In[73]:


pred = pd.DataFrame(y_pred)
pred.describe()


# In[91]:


pred[0].value_counts()


# In[ ]:




