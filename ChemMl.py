#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[100]:


df=pd.read_csv("chem.csv", error_bad_lines=False, encoding='ISO-8859-1')


# In[101]:


df.head()


# In[102]:


df.corr()


# In[103]:


df.describe()


# In[104]:


df.info()


# In[105]:


df1=df["boiling point (K)"]


# In[106]:


df1.head()


# In[107]:


df2=df.rename(index=str,columns={"name":"name","molweight":"weight","critical temperature (K)":"ctemp","acentric factor":"afactor","boiling point (K)":"bp","Unnamed: 5":"nan1"," Unnamed: 6":"nan2"})


# In[108]:


df2.head()


# In[109]:


df2=df2.iloc[:,0:5]


# In[110]:


df2.head()


# In[111]:


df2.drop("name",axis=1,inplace=True)


# In[112]:


df2.head()


# In[113]:


df2.corr()


# In[114]:


df2.boxplot(column="weight")


# In[115]:


sns.heatmap(df2.corr(),yticklabels=False)


# In[116]:


corrmat=df2.corr()


# In[117]:


sns.heatmap(df2.corr(),annot=True)


# In[25]:


from sklearn.model_selection import train_test_split


# In[118]:


df3=df2.drop(["bp"],axis=1)


# In[119]:


df3.head()


# In[31]:


df1


# In[50]:


#X_train,X_test,y_train,y_test=train_test_split(df3,df1,test_size=0.20,random_state=51)


# In[120]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(df3)


# In[121]:


X


# In[122]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics


# In[123]:


model=LinearRegression()
model.fit(X,df1)


# In[41]:


#y_pred=model.predict(X_test)


# In[42]:


#np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# In[124]:


from sklearn.ensemble import RandomForestRegressor


# In[125]:


rf= RandomForestRegressor(n_jobs=-1)


# In[126]:


rf.fit(X,df1)


# In[ ]:





# In[79]:


#scoreOfModel = rf.score(X_train, y_train)


#print("Score is calculated as: ",scoreOfModel)


# In[80]:


#y_pred=rf.predict(X_test)


# In[89]:


#X_test


# In[81]:


y_pred


# In[97]:


for z in zip(y_test, y_pred):
    print(z, (z[0]-z[1]) /z[0] )


# In[85]:


r = []
for pair in  zip(y_pred, y_test):
    r.append(pair)

plt.plot(r)


# In[56]:


estimators = np.arange(10, 200, 10)
scores = []
for n in estimators:
    rf.set_params(n_estimators=n)
    rf.fit(X_train, y_train)
    scores.append(rf.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)


# In[54]:


y_pred=rf.predict(X_test)


# In[60]:


np.mean(scores)


# In[73]:


scoresi = cross_val_score(rf, df3, df1, cv=10, scoring='neg_mean_absolute_error')


# In[74]:


scoresi


# In[75]:


np.mean(scoresi)


# In[77]:


scores


# In[78]:


np.mean(scores)


# In[151]:


pickle.dump(rf, open('model.pkl','wb'))
pickle.dump(sc,open('model1.pkl','wb'))


# In[152]:


model = pickle.load(open('model.pkl','rb'))
model1=pickle.load(open('model1.pkl','rb'))
print(model.predict(([[a, b, c]])))


# In[160]:


[[a,b,c]]=model1.transform(([[a, b, c]]))


# In[161]:


print(model.predict(([[a, b, c]])))


# In[157]:


[[a,b,c]]=([[18.01528, 547, 0.3449]])


# In[143]:


a


# In[144]:


b


# In[145]:


c


# In[ ]:




