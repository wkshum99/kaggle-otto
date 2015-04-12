
# coding: utf-8

# In[44]:

from bs4 import BeautifulSoup 
import urllib
import re
import string
import sys
import csv
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.grid_search
import sklearn.neighbors
import sklearn.metrics
import sklearn.ensemble
import sklearn.tree


# In[2]:

def get_zip_URL(url):
    r = requests.get(zipFileName).content
    s = StringIO.StringIO(r)
    zf = zipfile.ZipFile(s, 'r') # Read in a list of zipped files
    return zf


# In[3]:

def get_zip_FS(filename):
    r = open(filename, 'rb')
    zf = zipfile.ZipFile(r, 'r') # Read in a list of zipped files
    return zf


# In[86]:

infile = get_zip_FS('train.csv.zip')
df = pd.read_csv(infile.open(infile.namelist()[0]), index_col=0)
X = df.ix[:, range(0,len(df.columns)-1)]
Y = df.target


# In[153]:

#infile = get_zip_FS('test.csv.zip')
#df = pd.read_csv(infile.open(infile.namelist()[0]), index_col=0)
#X_test = df.ix[:, range(0,len(df.columns))]


# In[152]:

X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(
    X, Y, test_size=0.33, random_state=42)

# print X_train.shape
# print X_test.shape
# print Y_train.shape
# print Y_test.shape


# In[27]:

def fix_result(in_result):
    out_result = []
    for i in in_result:
        out_result.append(i)
    return out_result


# In[59]:

#def convert_result(in_result):
#    out_result = []
#    for i in in_result:
#        out_result.append(int(i[-1]))
#    return out_result


# In[139]:

def match_submission_format(in_result):
    out_result = []
    j = 1
    for i in in_result:
        a = [0] * 10
        a[0] = j
        a[int(i[-1])] = 1
        out_result.append(a)
        j = j + 1
    return out_result


# In[130]:

def write_submission_result(in_result):
    with open('submission.csv', 'wb') as of:
        csv_file = csv.writer(of, delimiter = ',')
        csv_file.writerow(['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
        for i in in_result:
            csv_file.writerow(i)


# In[146]:

# random forest
rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, Y_train)
y_result = rfc.predict(X_test)
#Y_test = fix_result(Y_test)
#sklearn.metrics.accuracy_score(Y_test, y_result, normalize=True)


# In[147]:

submission = match_submission_format(y_result)


# In[150]:

print submission[0:10]


# In[72]:

#convert_result(Y_train)
#X_train


# In[157]:

#decision tree
#one_hot_encoding = pd.get_dummies(Y_train)
#clf = sklearn.tree.DecisionTreeClassifier(random_state=0)
#sklearn.cross_validation.cross_val_score(clf, X_train, one_hot_encoding, cv=10)


# In[151]:

write_submission_result(submission)

