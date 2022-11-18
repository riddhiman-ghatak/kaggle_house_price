#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df= pd.read_csv("C:/Users/riddh/OneDrive/Documents/train.csv")
df.head()


# In[2]:


df1=df.replace('%','',regex=True)


# In[3]:


df2=df1.dropna(subset=['maincateg','platform','fulfilled1'])

import math
df3=df2.fillna({
    'Rating': math.floor(df2.Rating.median()),
    'price1': math.floor(df2.price1.median()),
    'actprice1': math.floor(df2.actprice1.median()),
    
    'norating1': math.floor(df2.norating1.median()),
    'noreviews1': math.floor(df2.noreviews1.median()),
    'star_5f': math.floor(df2.star_5f.median()),
    'star_4f': math.floor(df2.star_4f.median()),
    'star_3f': math.floor(df2.star_3f.median()),
    'star_2f': math.floor(df2.star_2f.median()),
    'star_1f': math.floor(df2.star_1f.median())
   
})


# In[4]:


dummies_gender=pd.get_dummies(df3.maincateg)


# In[5]:


dummies_platform=pd.get_dummies(df3.platform)


# In[6]:


merged1 = pd.concat([df3,dummies_platform],axis='columns')


# In[7]:


merged2 = pd.concat([merged1,dummies_gender],axis='columns')


# In[8]:


final = merged2.drop(['maincateg','platform','title','id'], axis='columns')
final.head()


# In[9]:


x=final[['Rating','actprice1','norating1','noreviews1','star_5f','star_4f','star_3f','star_2f','star_1f','fulfilled1','Amazon','Flipkart','Men','Women']]
y=final.price1
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.1)


# In[31]:


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)


# In[32]:



reg.score( X_test,Y_test)


# In[33]:


from sklearn import linear_model
lasso_reg=linear_model.Lasso(alpha=50,max_iter=150,tol=0.1)
lasso_reg.fit(X_train,Y_train)


# In[34]:


lasso_reg.score(X_test,Y_test)


# In[35]:


import numpy as np
import pandas as pd
df5= pd.read_csv("C:/Users/riddh/OneDrive/Documents/test.csv")


# In[36]:



df7=df5.fillna(method="ffill")


# In[37]:


dummies_gender_test=pd.get_dummies(df7.maincateg)
dummies_platform_test=pd.get_dummies(df7.platform)
merged1_test= pd.concat([df7,dummies_platform_test],axis='columns')
merged2_test= pd.concat([merged1_test,dummies_gender_test],axis='columns')
final_test = merged2_test.drop(['maincateg','platform','title','id'], axis='columns')
final_test


# In[40]:


y_predicted=lasso_reg.predict(final_test)
y_predicted


# In[44]:


from sklearn.ensemble import  GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =42)
gbr_model=gbr.fit(X_train,Y_train)


# In[ ]:





# In[45]:


y_pred_gbr=gbr_model.predict(final_test)
y_pred_gbr


# In[47]:


pred=pd.DataFrame(y_pred_gbr)
df5= pd.read_csv("C:/Users/riddh/OneDrive/Documents/test.csv")
datasets=pd.concat([df5['id'],pred],axis='columns')
datasets.columns=['id','price1']
datasets.to_csv('submission_riddhiman_gbr2.csv',index=False)


# In[ ]:




