#LESSION 4
Image Classifier with MNIST Data Set

[1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

[2]
data = pd.read_csv('DataSet/mnist_train.csv')

[19]
data.head(10)
label	1x1	1x2	1x3	1x4	1x5	1x6	1x7	1x8	1x9	...	28x19	28x20	28x21	28x22	28x23	28x24	28x25	28x26	28x27	28x28
0	5	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	9	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5	2	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
6	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
7	3	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
8	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
9	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
10 rows × 785 columns

[16]
a = data.iloc[4,1:].values

[17]
a= a.reshape(28,28).astype('uint8')

[20]
plt.imshow(a)
<matplotlib.image.AxesImage at 0x298481e99c8>

[22]
#prepocessing data
X = data.iloc[:,1:]
Y= data.iloc[:,0]

[24]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=4)

[26]
X_train.head()
1x1	1x2	1x3	1x4	1x5	1x6	1x7	1x8	1x9	1x10	...	28x19	28x20	28x21	28x22	28x23	28x24	28x25	28x26	28x27	28x28
20379	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
53032	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
27005	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
30510	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
508	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5 rows × 784 columns

[27]
model= RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
                       
[29]
Y_pred = model.predict(X_test)

[31]
Y_pred
array([2, 7, 6, ..., 6, 4, 2], dtype=int64)

[32]
s = Y_test.values
count = 0
for i in range(len(Y_pred)):
    if Y_pred[i] == s[i]:
        count-=-1

[33]
count
11610

[35]
len(Y_pred)
12000

[36]
count/len(Y_pred)
0.9675

















































