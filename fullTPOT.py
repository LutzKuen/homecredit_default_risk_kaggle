from tpot impot TPOTRegressor
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
 row[column] = strToNum(row[column])





if __name__ == '__main__':
 training_file = 'application_train.csv'
 test_file = 'application_test.csv'
 # join bureau
 training = pd.read_csv(training_file)
 test = pd.read_csv(test_file)
 bureau = pd.read_csv('bureau.csv')
 training = pd.merge(training, bureau, how = 'left', on = 'SK_ID_CURR')
 test = pd.merge(test, bureau, how = 'left', on = 'SK_ID_CURR')
 ccbal = pd.read_csv('credit_card_bala
