#!/usr/bin/env python
# coding: utf-8

# In[127]:


import pandas as pd
data=pd.read_csv('Company_Data.csv')
data.head()


# In[128]:


data.shape


# In[129]:


data.isnull().any(axis=1)


# In[130]:


data.isnull().any().sum()


# In[131]:


data['Sales'].unique()


# In[132]:


data['Sales'].value_counts


# In[133]:


data['Sales'].min()


# In[134]:


data["Sales"].max()


# In[135]:


x=data['Sales']


# In[136]:


x


# In[137]:


import matplotlib.pyplot as plt
plt.hist(x)


# In[138]:


#setting up of bins
bin=[0,3,6,9,12,15,18]


# In[139]:


category=pd.cut(x,bin, labels=[1,2,3,4,5,6])
category=category.to_frame()
category.columns=["Sales_range_type"]


# In[140]:


df_new=pd.concat([x,category],axis=1)


# In[141]:


df_new


# In[142]:


import matplotlib.pyplot as plt
import seaborn as sn
sn.set(style='darkgrid',color_codes=True)
sn.countplot(x='Sales_range_type',data=df_new, palette='hls')
plt.show()


# In[143]:


df=pd.concat([data,category],axis=1)


# In[144]:


df


# In[21]:


df.shape


# In[67]:


#After dropping null values
df.shape


# In[68]:


df


# In[69]:


from sklearn.model_selection import train_test_split
train, test=train_test_split(df,test_size=0.25)


# In[70]:


from sklearn.tree import DecisionTreeClassifier
help(DecisionTreeClassifier)


# In[71]:


df.columns


# In[72]:


target=df[['Sales_range_type']]
target


# In[73]:


pred=df[['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US']]
pred


# In[74]:


string_col=['ShelveLoc', 'Urban', 'US']
from sklearn.preprocessing import LabelEncoder
num=LabelEncoder()
for i in string_col:
    train[i]=num.fit_transform(train[i])
    test[i]=num.fit_transform(test[i])


# In[28]:


colnames=train.columns


# In[29]:


len(colnames)


# In[34]:


colnames.isnull().any(axis=0)


# In[36]:


df.isnull().any(axis=1)


# In[50]:


df.isnull().sum()


# In[60]:


df.dropna(inplace=True)


# In[63]:


df.isnull().sum()


# In[37]:


trainX.isnull().any().sum()


# In[41]:


trainY.isnull().any()


# In[44]:


trainY[trainY.isnull()]


# In[45]:


df.iloc[174]


# In[76]:


trainY.isnull().any()


# In[42]:


testX.isnull().any().sum()


# In[43]:


testY.isnull().any().sum()


# In[75]:


trainX = train[colnames[1:10]]
trainY = train[colnames[11]]
testX  = test[colnames[1:10]]
testY  = test[colnames[11]]


# In[77]:


model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(trainX,trainY)


# In[78]:


model.predict(testX)


# In[79]:


testY


# In[81]:


t=model.predict(testX)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(testY,t)


# In[82]:


cm


# In[83]:


preds = model.predict(testX)
pd.Series(preds).value_counts()
pd.crosstab(testY,preds)


# In[84]:


# Accuracy = train
np.mean(train.Sales_range_type == model.predict(trainX))


# In[85]:


# Accuracy = Test
np.mean(preds==test.Sales_range_type)


# In[ ]:





# # After Normalization

# In[87]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
std_trainX=scale.fit_transform(trainX)
std_testX=scale.fit_transform(testX)


# In[88]:


model1 = DecisionTreeClassifier(criterion = 'entropy')
model1.fit(std_trainX,trainY)


# In[93]:


y_pred=model1.predict(std_testX)
y_pred


# In[90]:


testY


# In[94]:


from sklearn.metrics import confusion_matrix
confusion_matrix(testY,y_pred)


# In[92]:


# Accuracy = train
np.mean(train.Sales_range_type == model.predict(std_trainX))


# In[95]:


# Accuracy = Test
np.mean(y_pred==test.Sales_range_type)


# In[ ]:





# # Using Feature Selection

# In[96]:


df.columns


# In[97]:


feature_col=['CompPrice', 'Income','Advertising','Population','Price','ShelveLoc','Urban']


# In[110]:


from sklearn.preprocessing import LabelEncoder
num=LabelEncoder()
for i in feature_col:
    df[i]=num.fit_transform(df[i])


# In[112]:


x=df[feature_col]
x


# In[ ]:





# In[100]:


y=df['Sales_range_type']
y


# In[114]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0) 


# In[115]:


model2 = DecisionTreeClassifier()
model2.fit(x_train,y_train)


# In[116]:


y_pred = model2.predict(x_test)
y_pred


# In[117]:


y_test


# In[120]:


#Test Accuracy
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[122]:


#Train Accuracy
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_train, model2.predict(x_train)))


# In[ ]:





# In[ ]:


#Label encoding only two col


# In[145]:


stringcol=['ShelveLoc','Urban']
from sklearn.preprocessing import LabelEncoder
num=LabelEncoder()
for i in stringcol:
    df[i]=num.fit_transform(df[i])


# In[153]:


z=df[feature_col]
z


# In[150]:


df.isnull().any()


# In[152]:


df.dropna(inplace=True)


# In[172]:


z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.30, random_state=0) 


# In[197]:


model3 = DecisionTreeClassifier(criterion='entropy',max_depth=5)
model3.fit(z_train,y_train)


# In[198]:


y_pred = model3.predict(z_test)
y_pred


# In[199]:


y_test


# In[200]:


#Test Accuracy
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:





# # Visualising Decision Tree

# In[123]:


get_ipython().system('pip install graphviz')


# In[124]:


get_ipython().system('pip install pydotplus')


# In[204]:


#export_graphviz function converts decision tree classifier into dot file.
#pydotplus convert this dot file to png or displayable form on Jupyter.
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(model2, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_col,class_names=['1','2','3','4','5','6'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Decision Tree.png')
Image(graph.create_png())


# In[206]:


#With max_depth=5
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(model3, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_col,class_names=['1','2','3','4','5','6'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Decision Tree5.png')
Image(graph.create_png())


# In[201]:


get_ipython().system('pip install --upgrade scikit-learn==0.20.3')


# In[ ]:




