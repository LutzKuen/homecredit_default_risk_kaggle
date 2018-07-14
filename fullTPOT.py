from tpot import TPOTRegressor
import pandas as pd


def strToNum(st):
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
   return csfrom tpot impot TPOTRegressor
import pandas as pd

def maskColumn(column, row):
 return strToNum(row[column])
 





if __name__ == '__main__':
 dataset = pd.read_csv('feature_matrix.csv')
 training = dataset[dataset['TARGET'].isnull() == False]
 test = dataset[dataset['TARGET'].isnull()]
 target = training['TARGET']
 training.drop(columns = 'TARGET', axis = 1 , inplace = True)
 test.drop(columns = 'TARGET', axis = 1 , inplace = True)
 for column in training.columns: # mask all string values and nan's
  lfun = lambda row: maskColumn(column, row)
  training[column] = training.apply(lfun, axis = 1)
  test[column] = test.apply(lfun, axis = 1)
 tpot = TPOTRegressor(generations = 5, population_size = 50, verbosity = 2)
 tpot.fit(training.values, target.values)
 tpot.export('tpot_homecredit_best_pipeline.py')
 result = tpot.predict(test.values)
 submission = zip(test.values[:,0], result[:])
 submission_file = open('submission.csv','w')
 submission_file.write('SK_ID_CURR,TARGET\n')
 sk_id_curr = test['SK_ID_CURR'].values
 for i in range(len(sk_id_curr)):
  new_line = str(sk_id_curr[i]) + ',' + str(result[i]) + '\n'
  submission_file.write(new_line)
 submission_file.close()