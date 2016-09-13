'''
Author: Rohan Koodli
Implementing Electricity Consumption Surveys with sklearn
Version 2.0
'''

import pandas as pd
import numpy as np

# reading in aztlan data on income, kilowatt-hour per capita, and wgt
aztlan_seed = pd.read_csv('\Users\Rohan\Documents\GitHub\Electricity_Consumption_Surveys\ipc_microsim_tool\data\example_aztlan_seed.tab.txt',
                     sep='\t', index_col=None, na_values='')
                     
aztlan_elast = pd.read_csv('\Users\Rohan\Documents\GitHub\Electricity_Consumption_Surveys\ipc_microsim_tool\data\example_aztlan_w_elast.tab.txt',
                     sep='\t', index_col=None, na_values='')
'''
X = np.array(aztlan_seed['income'])
#print (X)
#X = X[:, None]
# X is all the income data values
X1 = []
for i in X:
    X1.append([i])

#print X1
y = list(aztlan_seed['kwhpc'])
print type(y)
# y is all the kilowatt-hour values
print len(y)

X2 = np.array(aztlan_elast['income'])
for j in X2:
    X1.append([j])

y2 = (aztlan_elast['kwhpc'])
y.append(y2)
print len(y)
print len(X1)
'''

X1 = list(aztlan_seed['income'])
X2 = list(aztlan_elast['income'])
X = []
for i in X1:
    X.append([i])
for j in X2:
    X.append([j])

print 'X length',len(X)

y1 = list(aztlan_seed['kwhpc'])
y2 = list(aztlan_elast['kwhpc'])
y = []
for k in y1:
    y.append(k)
for l in y2:
    y.append(l)
print 'y len',len(y)

from sklearn import neighbors,svm#,tree,ensemble
#rfr = ensemble.RandomForestRegressor()
#dtr = tree.DecisionTreeRegressor()
knr = neighbors.KNeighborsRegressor()
svr = svm.SVR()
#nnn = neural_network.BernoulliRBM()
''''''
from sklearn.cross_validation import cross_val_score
print 'KNeighbors Regressor', cross_val_score(knr,X,y,cv=2)
print 'Support Vector Regressor', cross_val_score(svr,X,y,cv=2)
''''''
from matplotlib import pyplot as plt
import seaborn; seaborn.set()

plt.scatter(X,y)
plt.xlabel("Income")
plt.ylabel('Kilowatt-hours per capita')
plt.title('Aztlan Seed & Elasticity data')



