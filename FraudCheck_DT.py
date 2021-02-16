#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
fraud=pd.read_csv('Fraud_check.csv')
fraud.head()


# In[20]:


fraud.shape


# In[21]:


fraud.info()


# In[22]:


fraud.isnull().any()


# In[23]:


fraud['Taxable.Income']


# In[24]:


import numpy as np
cond=np.where(fraud["Taxable.Income"]>=30000,"good","risky")
cond


# In[25]:


df=pd.DataFrame(cond)


# In[26]:


df


# In[27]:


df=df.rename({0:'TIncome'},axis=1)
df


# In[28]:


fraud=pd.concat([fraud,df],axis=1)
fraud


# In[29]:


string_col=['Undergrad','Marital.Status','Urban','TIncome']
from sklearn.preprocessing import LabelEncoder
pre=LabelEncoder()
for i in string_col:
    fraud[i]=pre.fit_transform(fraud[i])


# In[30]:


fraud


# In[82]:


fraud_df=fraud.reindex(columns=['Undergrad','Marital.Status','City.Population','Work.Experience','Urban','TIncome'])
fraud_df.head()


# In[83]:


fraud_df.columns


# In[84]:


from sklearn.model_selection import train_test_split
train, test=train_test_split(fraud_df,test_size=0.25)


# In[85]:


from sklearn.tree import DecisionTreeClassifier
help(DecisionTreeClassifier)


# In[86]:


model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,0:5],train.iloc[:,5])


# In[87]:


model.predict(test.iloc[:,0:5])


# In[88]:


test.iloc[:,5]


# In[89]:


import matplotlib.pyplot as plt
plt.hist(fraud_df['TIncome'])


# In[90]:


#testing accurac
from sklearn import metrics
print("Accuracy=",metrics.accuracy_score(test.iloc[:,5],model.predict(test.iloc[:,0:5])))


# In[91]:


#training accuracy
from sklearn import metrics
print("Training accuracy=",metrics.accuracy_score(train.iloc[:,5],model.predict(train.iloc[:,0:5])))


# In[92]:


#Improving the testing accuracy


# In[93]:


model1 = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
model1.fit(train.iloc[:,0:5],train.iloc[:,5])


# In[94]:


model1.predict(test.iloc[:,0:5])


# In[95]:


test.iloc[:,5]


# In[96]:


#testing accuracy
from sklearn import metrics
print("Accuracy=",metrics.accuracy_score(test.iloc[:,5],model1.predict(test.iloc[:,0:5])))


# In[97]:


#training accuracy
from sklearn import metrics
print("Training accuracy=",metrics.accuracy_score(train.iloc[:,5],model1.predict(train.iloc[:,0:5])))


# In[99]:


#Visualizing Decision Tree
#export_graphviz function converts decision tree classifier into dot file.
#pydotplus convert this dot file to png or displayable form on Jupyter.
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(model1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Fraud Check.png')
Image(graph.create_png())


# In[ ]:


#With max_depth=3


# In[100]:


model2 = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model2.fit(train.iloc[:,0:5],train.iloc[:,5])


# In[101]:


model2.predict(test.iloc[:,0:5])


# In[102]:


test.iloc[:,5]


# In[103]:


#testing accuracy
from sklearn import metrics
print("Accuracy=",metrics.accuracy_score(test.iloc[:,5],model2.predict(test.iloc[:,0:5])))


# In[104]:


#training accuracy
from sklearn import metrics
print("Training accuracy=",metrics.accuracy_score(train.iloc[:,5],model2.predict(train.iloc[:,0:5])))


# In[105]:


#Visualizing Decision Tree
#export_graphviz function converts decision tree classifier into dot file.
#pydotplus convert this dot file to png or displayable form on Jupyter.
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(model2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Fraud Check#3.png')
Image(graph.create_png())


# # With feature selection

# In[106]:


fraud_df.columns


# In[108]:


feature_ext=fraud_df[['Undergrad', 'Marital.Status', 'Work.Experience','Urban', 'TIncome']]
feature_ext


# In[109]:


from sklearn.model_selection import train_test_split
train, test=train_test_split(feature_ext,test_size=0.25)


# In[119]:


mod = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
mod.fit(train.iloc[:,0:4],train.iloc[:,4])


# In[120]:


mod.predict(test.iloc[:,0:4])


# In[121]:


test.iloc[:,4]


# In[122]:


#testing accuracy
from sklearn import metrics
print("Accuracy=",metrics.accuracy_score(test.iloc[:,4],mod.predict(test.iloc[:,0:4])))


# In[123]:


#training accuracy
from sklearn import metrics
print("Training accuracy=",metrics.accuracy_score(train.iloc[:,4],mod.predict(train.iloc[:,0:4])))


# In[124]:


#Visualizing Decision Tree
#export_graphviz function converts decision tree classifier into dot file.
#pydotplus convert this dot file to png or displayable form on Jupyter.
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(mod, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Fraud Check_.png')
Image(graph.create_png())


# In[ ]:




