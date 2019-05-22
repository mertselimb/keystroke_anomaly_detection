#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd


# In[14]:


df = pd.read_csv("keystroke.csv")


# In[15]:


df.head()


# In[16]:


df.info()


# In[17]:


df.describe()


# In[18]:


df.columns


# In[19]:


subject = df.subject
sessionIndex = df.sessionIndex
df = df.drop(['subject','sessionIndex', 'rep'],axis=1)


# In[20]:


df.shape


# In[21]:


# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(df)


# In[22]:


pred = pd.DataFrame(y_pred)
pred.describe()


# In[23]:


pred[0].value_counts()


# In[42]:


pred["sessionIndex"] = sessionIndex
pred["subject"] = subject
pred.columns = ["isAnomaly", 'sessionIndex',"subject"]
pred.head()


# In[46]:


pred.groupby(["subject",'sessionIndex']).isAnomaly.value_counts()

