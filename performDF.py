from AWSTools import createInstance

cont = createInstance.controller('settings_aws.conf')

cont.createInstance()
files = [ '../data/application_train.csv',
'../data/application_test.csv',
'../data/bureau.csv',
'../data/bureau_balance.csv',
'../data/POS_CASH_balance.csv',
'../data/credit_card_balance.csv',
'../data/previous_application.csv',
'../data/installments_payments.csv',
'../data/df_analysis_kaggle.py']
cont.transferFilesToWorker(files)
cont.execOnRemote('install_deps.sh')
scriptName = '/home/ubuntu/external/homecredit/do_df_analysis.sh'
#cont.execOnRemote('source /home/ubuntu/.bashrc')
cont.execOnRemote(scriptName)
resName = ['feature_matrix.csv']
#'/home/ubuntu/feature_descriptions.list']
cont.retrieveResults(resName)
cont.terminateInstance()
