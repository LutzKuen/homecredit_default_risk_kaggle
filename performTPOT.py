from AWSTools import createInstance

cont = createInstance.controller('../settings_aws.conf')

cont.createInstance()
files = [ '../homecredit_results/feature_matrix.csv',
'fullTPOT.py']
cont.transferFilesToWorker(files)
cont.execOnRemote('install_deps_tpot.sh')
scriptName = 'do_tpot.sh'
#cont.execOnRemote('source /home/ubuntu/.bashrc')
cont.execOnRemote(scriptName)
resName = ['submission.csv', 'tpot_homecredit_best_pipeline.py']
#'/home/ubuntu/feature_descriptions.list']
cont.retrieveResults(resName)
cont.terminateInstance()
