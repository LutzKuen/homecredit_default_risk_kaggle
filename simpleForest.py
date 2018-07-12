import pandas as pd
import sys
from random import random
import numpy as np
from sklearn import metrics
#from sklearn.model_selection import train_test_split
import math
#from sklearn import tree
from tpot import TPOTRegressor
import autosklearn.regression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import BaggingRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from evolutionary_search import EvolutionaryAlgorithmSearchCV

import pickle
from os import listdir
from os.path import isdir, join
import os

class simpleEstimator:
 def __init__(self):
  print('Initializing')
  self.application_file = 'application_train.csv'
  self.application_test = 'application_test.csv'
  self.additional_data = ['bureau_preprocessed.csv'] # 0.617 nachdem ich den score aufgenommen habe
  self.outfile = 'submission.csv'
 def submit(self):
  ofile = open(self.outfile,'w')
  ofile.write('SK_ID_CURR,TARGET\n')
  print('Preparing submission')
  df = pd.read_csv(self.application_test, quotechar = '"')
  for additional in self.additional_data:
   dfadd = pd.read_csv(additional)
   df = pd.merge(df , dfadd, on = 'SK_ID_CURR', how = 'left')
  tmat = df.values
  index = tmat[:,0]
  x = tmat[:,1:]
  for i in range(0,x.shape[0]):
   for j in range(0,x.shape[1]):
    x[i,j] = self.strtonum(x[i,j])
  y = self.predict(x)
  for i in range(0,len(y)):
   oline = str(index[i])+','+str(max(min(y[i],1),0))+'\n'
   ofile.write(oline)
  ofile.close()
 
 def prepare(self):
  #print(df.shape)
  df = pd.read_csv(self.application_file, quotechar = '"')
  for additional in self.additional_data:
   dfadd = pd.read_csv(additional)
   df = pd.merge(df , dfadd, on = 'SK_ID_CURR', how = 'left')
  #print(df.shape)
  tmat = df.values
  self.x = tmat[:,2:]
  self.y = np.array(tmat[:,1], dtype = np.float64)
  #self.y = np.array(y.astype('float'), dtype = np.float64)
  
  index = tmat[:,0]
  for i in range(0,self.x.shape[0]):
   for j in range(0,self.x.shape[1]):
    self.x[i,j] = self.strtonum(self.x[i,j])
  #for i in range(len(self.y)):
  # self.y[i] = self.strtonum(self.y[i])
  
 def gridsearch(self, parameters):
  svc = GradientBoostingRegressor()
  self.clf = GridSearchCV(svc, parameters, verbose = 1, n_jobs = 4)
  self.clf.fit(self.x, self.y)
 def train(self, params):
  #print('Training the model')
  #self.clf = RandomForestRegressor() # 0.625
  #self.clf = RandomForestClassifier() # 0.5
  #self.clf = GradientBoostingRegressor() # 0.730
  #self.clf = BaggingRegressor(KNeighborsClassifier(),max_samples=0.5, max_features=0.5) # MemoryError
  #self.clf = AdaBoostRegressor() # 0.674
  #clf = KNeighborsRegressor() # MemoryError
  #self.clf = MLPRegressor(hidden_layer_sizes = (5,)) # MemoryError
  #self.clf = GradientBoostingRegressor(**params)
  #self.clf = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task= 75000, per_run_time_limit= 7500 )
  self.clf = TPOTRegressor(generations=5, population_size=50, verbosity=2, n_jobs = 3)
  self.clf.fit(self.x, self.y)
  self.clf.export('tpot_best_pipeline.py')
  
 def predict(self, x):
  return self.clf.predict(x)
 def test(self):
  #print('Testing the model')
  ypred = self.predict(self.x)
  rmse = self.get_rmse(ypred, self.y)
  print('OOS RMSE: ' + str(rmse))
  return rmse
 def strtonum(self, st):
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
 def get_rmse(self, a, b):
  mse = 0
  for i in range(0, len(a)):
   mse += (a[i] - b[i])**2
  mse = math.sqrt(mse / len(a))
  return mse
def getNearest(inval, minval):
 return [inval ]
 lower = math.floor(inval*0.9)
 upper = math.ceil(inval*1.1)
 if lower >= minval:
  return [lower, inval, upper]
 else:
  return [inval, upper]
def getNearestf(inval, minval):
 return [ inval]
 lower = inval*0.9
 upper = inval*1.1
 if lower >= minval:
  return [lower, inval, upper]
 else:
  return [inval, upper]
if __name__ == '__main__': 
 if sys.argv[1] == 'single':
  estim = simpleEstimator()
  estim.prepare()
  
  #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls', 'subsample': 0.5, 'verbose': 1} # 0.731
  #params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.05, 'loss': 'quantile', 'subsample': 0.5, 'verbose': 1}
  logfile = pd.read_csv('gridsearch.log')# open('gridsearch.log','r')
  bestline = logfile[logfile['rmse'] == min(logfile['rmse'])]
  print('Using best prevailing parameter set ' + str(float(bestline['rmse'])))
  params = {'n_estimators': int(bestline['n_estimators'].values[0]),
  'max_depth': int(bestline['max_depth'].values[0]),
  'min_samples_split': int(bestline['min_samples_split'].values[0]),
  'learning_rate': float(bestline['learning_rate'].values[0]),
  'loss': bestline['loss'].values[0],
  'subsample': float(bestline['subsample'].values[0]),
  'verbose': 1}
  estim.train(params)
  estim.test()
  estim.submit()
  
  
  
 if sys.argv[1] == 'improve':
  estim = simpleEstimator()
  estim.prepare()
  logfile = pd.read_csv('gridsearch.log')# open('gridsearch.log','r')
  bestline = logfile[logfile['rmse'] == min(logfile['rmse'])]
  print('Found best prevailing parameter set ' + str(float(bestline['rmse'])))
  grid = { 'n_estimators': getNearest(bestline['n_estimators'].values[0],10), #int(bestline['n_estimators']),
  'max_depth': getNearest(bestline['max_depth'].values[0],2),
  'min_samples_split': getNearest(bestline['min_samples_split'].values[0],2),
  'learning_rate': getNearestf(bestline['learning_rate'].values[0],0.0001),
  'loss': ['ls', 'lad', 'huber', 'quantile'],
  'subsample': getNearestf(bestline['subsample'].values[0],0.000001),
  'verbose': [ 0 ]}
  estim.gridsearch(grid)
  estim.submit()
  rmse = estim.test()
  params = estim.clf.best_params_
  oline = str(params.get('n_estimators')) + ',' + str(params.get('max_depth'))+','+str(params.get('min_samples_split'))+','+str(params.get('learning_rate'))+',' + params['loss'] + ','+str(params.get('subsample'))+','+str(rmse)+'\n'
  logfile = open('gridsearch.log','a')
  logfile.write(oline)
  logfile.close()
