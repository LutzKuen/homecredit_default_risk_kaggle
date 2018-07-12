import pandas as pd
import numpy as np
from multiprocessing import Process
from multiprocessing import Manager

def worker(values, queue):
 cind = -1
 nx = 0
 nc = 0
 no = 0
 for index in range(0,values.shape[0]):
  idx = values[index,0]
  status = values[index,2]
  if idx == cind:
   if status == 'X':
    nx += 1  
   if status == 'C':
    nc += 1
   if status == '0':
    no += 1
  else:
   if cind > 0:
    oline = str(cind) +',' + str(nx) + ',' + str(nc) + ','+str(no)+'\n'
    queue.put(oline)
   cind = idx
   nx = 0
   nc = 0
   no = 0
   if status == 'X':
    nx += 1  
   if status == 'C':
    nc += 1
   if status == '0':
    no += 1
 oline = str(cind) +',' + str(nx) + ',' + str(nc) + ','+str(no)+'\n'
 queue.put(oline)
 queue.put('DONE')

def writer(queue, numjobs):
 running_jobs = numjobs
 out = open('balances_preprocessed_tmp.csv','w')
 out.write('SK_ID_BUREAU,NX,NC,NO\n')
 print(str(running_jobs) + ' running')
 while True:
  msg = queue.get()
  if msg == 'DONE':
   running_jobs -= 1
   print(str(running_jobs) + ' running')
   if running_jobs == 0:
    return
  else:
   out.write(msg)
 out.close()
if __name__ == '__main__':
 print('Init Queue')
 queue = Manager().Queue()
 df_all = pd.read_csv('bureau_balance.csv', chunksize = 3*10**5)
 jobarr = []
 kk = 0
 for df in df_all:
  kk += 1
  values = df.values
  njob = Process(target = worker, args = ((values), (queue), ))
  njob.start()
  jobarr.append(njob)
 writer(queue, len(jobarr))
 out_tmp = pd.read_csv('balances_preprocessed_tmp.csv')
 out = open('balances_preprocessed_tmp.csv','w')
 out.write('SK_ID_BUREAU,NX,NC,NO\n')
 kk = 0
 for idx in np.unique(out_tmp['SK_ID_BUREAU']):
  nx = np.sum(out_tmp[out_tmp['SK_ID_BUREAU'] == idx]['NX'].values)
  nc = np.sum(out_tmp[out_tmp['SK_ID_BUREAU'] == idx]['NC'].values)
  no = np.sum(out_tmp[out_tmp['SK_ID_BUREAU'] == idx]['NO'].values)
  oline = str(idx) +',' + str(nx) + ',' + str(nc) + ','+str(no)+'\n'
  out.write(oline)
  if kk % 1000 == 0:
   print(str(100*kk/out_tmp.shape[0]) + '% done')
  kk += 1
 out.close()