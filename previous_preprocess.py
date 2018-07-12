#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

#import operator
import math
import random

import numpy as np
#from functools import partial
from sklearn.ensemble import GradientBoostingRegressor
#from deap import algorithms
#from deap import base
#from deap import creator
#from deap import tools
#from deap import gp
import pandas as pd
def maskColumn(colname, row):
 st = row[colname]
 try:
  f = float(st)
  if math.isnan(f):
   return -10000
  return f
 except:
  sta = [ord(x) for x in st]
  cs = 0
  for i in sta:
   cs += i
  return cs
# data preparation

applications = pd.read_csv('../homecredit/application_train.csv', usecols = ['SK_ID_CURR', 'TARGET']) #.head(1000)
previous = pd.read_csv('../homecredit/previous_application.csv')

#merged = pd.merge(bureau, balances, how = 'left', on = 'SK_ID_BUREAU')
params = {'n_estimators': 300,
   'max_depth': 12,
   'min_samples_split': 2,
   'learning_rate': 0.15,
   'loss': 'ls',
   'subsample': 0.5,
   'verbose': 1}

for colname in previous.columns:
 maskCol = lambda row: maskColumn(colname, row)
 previous[colname] = previous.apply(maskCol, axis = 1)

merged = pd.merge(previous, applications, on = 'SK_ID_CURR', how = 'left')
df_train = merged[merged['TARGET'].isnull() == False]

xtrain = np.nan_to_num(df_train.values[:,2:-1])
ytrain = np.nan_to_num(df_train.values[:,-1])
x = np.nan_to_num(merged.values[:,2:-1])

  
clf = GradientBoostingRegressor(**params)
clf.fit(xtrain, ytrain)
 
#applications_test = pd.read_csv('../homecredit/application_train.csv', usecols = ['SK_ID_CURR']) #.head(1000)
prepro = open('previous_preprocessed.csv','w')
prepro.write('SK_ID_CURR,BUREAU\n')

y = clf.predict(x)
previous['TARGET'] = pd.Series(y)

kk = 0
for i in np.unique(previous['SK_ID_CURR'].values):
 ysum = np.sum(previous[previous['SK_ID_CURR'] == i]['TARGET'].values)
 oline = str(i)+','+str(ysum)+'\n'
 #print(oline)
 prepro.write(oline)
 kk += 1
 if kk % 1000 == 0:
  print(str(100*kk/previous.shape[0]) + ' %')
prepro.close()
