#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all the packages

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from datetime import datetime, date

from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

get_ipython().magic(u'matplotlib inline')


# In[2]:


#Read and combine the training files

work = pd.read_csv('AdvWorksCusts.csv')
print(work.shape)
avemon = pd.read_csv('AW_AveMonthSpend.csv')
print(avemon.shape)
bikebuy = pd.read_csv('AW_BikeBuyer.csv')
print(bikebuy.shape)

work1 = pd.merge(work,avemon, how = 'inner', on = 'CustomerID')
print(work1.shape)

train = pd.merge(work1,bikebuy, how = 'inner', on = 'CustomerID')

print(train.shape)
train.head()

#Read and combine the test files
test = pd.read_csv('AW_test.csv')
print(test.shape)
test.head()


# In[3]:


#drop unnecessary column - training dataset

train.drop(['Title'], axis=1, inplace=True)
train.drop(['MiddleName'], axis=1, inplace=True)
train.drop(['AddressLine2'], axis=1, inplace=True)
train.drop(['Suffix'], axis=1, inplace=True)

print(train.shape)
train.head()

#drop unnecessary column - testing dataset

test.drop(['Title'], axis=1, inplace=True)
test.drop(['MiddleName'], axis=1, inplace=True)
test.drop(['AddressLine2'], axis=1, inplace=True)
test.drop(['Suffix'], axis=1, inplace=True)

print(test.shape)
test.head()


# In[4]:


#check unique rows - Train
print(train.shape)
print(train.CustomerID.unique().shape)

#drop duplicates - Train
train.drop_duplicates(subset='CustomerID',keep='first',inplace=True)

#check unique rows - Train
print(train.shape)
print(train.CustomerID.unique().shape)

#check unique rows - Test
print(test.shape)
print(test.CustomerID.unique().shape)

#drop duplicates - Test
test.drop_duplicates(subset='CustomerID',keep='first',inplace=True)

#check unique rows - Test
print(test.shape)
print(test.CustomerID.unique().shape)


# In[5]:


#calculating age from birthdate and assigning to bins - train

train.dtypes
train['BirthDate'] = pd.to_datetime(train['BirthDate'])
today = datetime.strptime('01 01 1998', "%d %m %Y")
train['age'] = (today - train['BirthDate']).astype('>m8[Y]')
train['age'] = pd.to_numeric(train['age'])

bins = [0, 25, 45, 55, 100]
names = ['<25', '25-45', '45-55', '>55']

train['AgeRange'] = pd.cut(train['age'], bins, labels=names)
train['AgeRange'] = train['AgeRange'].astype('str')

train['agegender'] = train['AgeRange']+train['Gender']
print(train.dtypes)

#calculating age from birthdate and assigning to bins - test

test.dtypes
test['BirthDate'] = pd.to_datetime(test['BirthDate'])
today = datetime.strptime('01 01 1998', "%d %m %Y")
test['age'] = (today - test['BirthDate']).astype('>m8[Y]')
test['age'] = pd.to_numeric(test['age'])

#bins = [0, 25, 45, 55, 100]
#names = ['<25', '25-45', '45-55', '>55']

test['AgeRange'] = pd.cut(test['age'], bins, labels=names)
test['AgeRange'] = test['AgeRange'].astype('str')

test['agegender'] = test['AgeRange']+test['Gender']
print(test.dtypes)


# In[6]:


#define label to be predicted
labels = np.array(train['BikeBuyer'])

#one hot encoding for categorical data 

def encode_string(cat_features):
    
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    
    ## Now, apply one hot encoding
    
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

# encoding training data
categorical_columns = ['Gender', 'MaritalStatus']

Features = encode_string(train['Occupation'])
for col in categorical_columns:
    temp = encode_string(train[col])
    Features = np.concatenate([Features, temp], axis = 1)

Features = np.concatenate([Features, np.array(train[['NumberCarsOwned', 'NumberChildrenAtHome', 
                            'TotalChildren', 'YearlyIncome','age']])], axis = 1)    

# encoding testing data
categorical_columns = ['Gender', 'MaritalStatus']

Test_Features = encode_string(test['Occupation'])
for col in categorical_columns:
    temp = encode_string(test[col])
    Test_Features = np.concatenate([Test_Features, temp], axis = 1)

Test_Features = np.concatenate([Test_Features, np.array(test[['NumberCarsOwned', 'NumberChildrenAtHome', 
                            'TotalChildren', 'YearlyIncome','age']])], axis = 1)    


#defining training data
train_data = Features


#defining testing data
test_data = Test_Features


# In[7]:


#Train Model
model = linear_model.LogisticRegression() 
model.fit(train_data, labels)


# In[14]:


#Test Model
prediction = model.predict(test_data)


# In[20]:


result = pd.DataFrame(prediction)
result.CustomerID=test.CustomerID
result.columns = ["prediction"]
result.to_csv("prediction_results.csv")


# In[ ]:




