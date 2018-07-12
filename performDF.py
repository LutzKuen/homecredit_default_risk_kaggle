from AWSTools import createInstance

cont = createInstance.controller('settings_aws.conf')

cont.createInstance()
files = [ '/home/ubuntu/external/homecredit/application_train.csv',
'/home/ubuntu/external/homecredit/application_test.csv',
'/home/ubuntu/external/homecredit/bureau.csv',
'/home/ubuntu/external/homecredit/bureau_balance.csv',
'/home/ubuntu/external/homecredit/POS_CASH_balance.csv',
'/home/ubuntu/external/homecredit/credit_card_balance.csv',
'/home/ubuntu/external/homecredit/previous_application.csv',
'/home/ubuntu/external/homecredit/installments_payments.csv',
'/home/ubuntu/external/homecredit/df_analysis_kaggle.py']
cont.transferFilesToWorker(files)
cont.execOnRemote('install_deps.sh')
scriptName = '/home/ubuntu/external/homecredit/do_df_analysis.sh'
#cont.execOnRemote('source /home/ubuntu/.bashrc')
cont.execOnRemote(scriptName)
resName = ['/home/ubuntu/feature_matrix.csv']
#'/home/ubuntu/feature_descriptions.list']
cont.retrieveResults(resName)
cont.terminateInstance()
