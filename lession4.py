# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# %%
data = pd.read_csv('DataSet/mnist_train.csv')


# %%
data.head(10)


# %%
a = data.iloc[4,1:].values


# %%
a= a.reshape(28,28).astype('uint8')


# %%
plt.imshow(a)


# %%
#prepocessing data
X = data.iloc[:,1:]
Y= data.iloc[:,0]


# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=4)


# %%
X_train.head()


# %%
model= RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)


# %%
Y_pred = model.predict(X_test)


# %%
Y_pred


# %%
s = Y_test.values
count = 0
for i in range(len(Y_pred)):
    if Y_pred[i] == s[i]:
        count-=-1


# %%
count


# %%
len(Y_pred)


# %%
count/len(Y_pred)


