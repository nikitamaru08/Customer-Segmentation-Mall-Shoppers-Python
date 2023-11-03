#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("Mall_Customers.csv")


# In[3]:


df.head()


# In[4]:


#Univariate analaysis
df.describe()


# In[5]:


sns.displot(df['Annual Income (k$)'])


# In[6]:


df.columns


# In[7]:


columns=(['Age','Annual Income (k$)','Spending Score (1-100)'])
for i in columns:
        plt.figure()
        sns.displot(df[i], kde=True)


# In[8]:


sns.kdeplot(x=df['Annual Income (k$)'],shade = True, hue = df['Gender']);


# In[9]:


columns=(['Age','Annual Income (k$)','Spending Score (1-100)'])
for i in columns:
        plt.figure()
        sns.kdeplot(x=df[i],shade = True, hue = df['Gender'])


# In[12]:


columns=(['Age','Annual Income (k$)','Spending Score (1-100)'])
for i in columns:
        plt.figure()
        sns.boxplot(data=df,x='Gender', y=df[i])


# In[16]:


df['Gender'].value_counts(normalize=True)


# In[18]:


#Bivariante Analysis

sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)')


# In[25]:


#df = df.drop('CustomerID', axis=1)
sns.pairplot(df, hue= 'Gender')


# In[27]:


df.groupby(['Gender'])['Age','Annual Income (k$)','Spending Score (1-100)'].mean()


# In[28]:


df.corr()


# In[29]:


sns.heatmap(df.corr(), annot =True, cmap = 'coolwarm')


# In[30]:


#clustering- univariate, bivariate, multivariatec


# In[34]:


clustering1 = KMeans()


# In[35]:


clustering1.fit(df[['Annual Income (k$)']])


# In[37]:


clustering1.labels_


# In[40]:


df['Income Cluster'] =  clustering1.labels_
df.head()


# In[41]:


df['Income Cluster'].value_counts()


# In[42]:


clustering1.inertia_


# In[46]:


inertia_score=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_score.append(kmeans.inertia_)


# In[47]:


plt.plot(range(1,11),inertia_score)


# In[52]:


df.columns


# In[53]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)', 'Spending Score (1-100)',
       'Income Cluster'].mean()


# In[ ]:


#Bivariate analysis


# In[54]:


clustering2 = KMeans()
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
clustering2.labels_
df['Spending and Income Cluster'] = clustering2.labels_
df.head()


# In[60]:


inertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    inertia_scores2.append(kmeans2.inertia_)
              
plt.plot(range(1,11),inertia_scores2)


# In[75]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[78]:


plt.figure(figsize=(10,8))
sns.scatterplot( x=centers['x'], y=centers['y'],s=100 ,c = 'Black', marker = '*')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',hue = 'Spending and Income Cluster', palette='tab10')


# In[80]:


pd.crosstab(df['Spending Score (1-100)'],df['Gender'])


# In[81]:


df.groupby('Spending Score (1-100)')['Age', 'Annual Income (k$)', 'Spending Score (1-100)'].mean()


# In[ ]:


#MULTIVARIATE CLUSTRING


# In[82]:


from sklearn.preprocessing import StandardScaler


# In[83]:


scale = StandardScaler()


# In[84]:


df.head()


# In[86]:


dff = pd.get_dummies(df,drop_first =True)
dff.head()


# In[87]:


dff.columns


# In[89]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Income Cluster',
        'Gender_Male']]
dff.head()


# In[93]:


dff = scale.fit_transform(dff)


# In[95]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[96]:


inertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    inertia_scores3.append(kmeans3.inertia_)
              
plt.plot(range(1,11),inertia_scores3)


# In[ ]:


df.to_csv('Clustering_csv')

