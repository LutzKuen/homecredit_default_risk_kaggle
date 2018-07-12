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

def get_credit_type(row):
  ctype = row['CREDIT_TYPE']
  if ctype == 'Consumer Credit':
   return 1
  else:
   return 2
def get_currency(row):
  curr = row['CREDIT_CURRENCY']
  if curr == 'currency 1':
   return 1
  if curr == 'currency 2':
   return 2
  if curr == 'currency 3':
   return  3
  if curr == 'currency 4':
   return 4
def get_active(row):
  act = row['CREDIT_ACTIVE']
  if act == 'Active':
   return 1
  else:
   return 0
def maskNAN_enndate(row):
 number = row['DAYS_ENDDATE_FACT']
 if math.isnan(number):
  return -1000
 else:
  return number
def maskNAN_amt_overdue(row):
 number = row['AMT_CREDIT_MAX_OVERDUE']
 if math.isnan(number):
  return -1000
 else:
  return number
def get_rmse(a, b):
  mse = 0
  for i in range(0, len(a)):
   mse += (a[i] - b[i])**2
  mse = math.sqrt(mse / len(a))
  return mse
# data preparation

applications = pd.read_csv('../homecredit/application_train.csv', usecols = ['SK_ID_CURR', 'TARGET']) #.head(1000)
balances = pd.read_csv('balances_preprocessed.csv')
bureau = pd.read_csv('../homecredit/bureau.csv')

#merged = pd.merge(bureau, balances, how = 'left', on = 'SK_ID_BUREAU')
params = {'n_estimators': 250,
   'max_depth': 12,
   'min_samples_split': 2,
   'learning_rate': 0.15,
   'loss': 'ls',
   'subsample': 0.5,
   'verbose': 1}
bureau['CREDIT_TYPE'] = bureau.apply(get_credit_type, axis = 1)
bureau['CREDIT_CURRENCY'] = bureau.apply(get_currency, axis = 1)
bureau['CREDIT_ACTIVE'] = bureau.apply(get_active, axis = 1)
bureau['DAYS_ENDDATE_FACT'] = bureau.apply(maskNAN_enndate, axis = 1)
bureau['AMT_CREDIT_MAX_OVERDUE'] = bureau.apply(maskNAN_amt_overdue, axis = 1)

merged = pd.merge(bureau, balances, on = 'SK_ID_BUREAU', how = 'left')
merged = pd.merge(merged, applications, on = 'SK_ID_CURR', how = 'left')
df_train = merged[merged['TARGET'].isnull() == False]

xtrain = np.nan_to_num(df_train.values[:,2:-1])
ytrain = np.nan_to_num(df_train.values[:,-1])
x = np.nan_to_num(merged.values[:,2:-1])

  
clf = GradientBoostingRegressor(**params)
clf.fit(xtrain, ytrain)
 
#applications_test = pd.read_csv('../homecredit/application_train.csv', usecols = ['SK_ID_CURR']) #.head(1000)
prepro = open('bureau_preprocessed.csv','w')
prepro.write('SK_ID_CURR,BUREAU\n')

y = clf.predict(x)
bureau['TARGET'] = pd.Series(y)

kk = 0
for i in np.unique(bureau['SK_ID_CURR'].values):
 ysum = np.sum(bureau[bureau['SK_ID_CURR'] == i]['TARGET'].values)
 oline = str(i)+','+str(ysum)+'\n'
 #print(oline)
 prepro.write(oline)
 kk += 1
 if kk % 1000 == 0:
  print(str(100*kk/bureau.shape[0]) + ' %')
prepro.close()
