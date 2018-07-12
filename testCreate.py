import createInstance

cont = createInstance.controller()

cont.createInstance()
files = [ '/home/ubuntu/external/homecredit/file1', '/home/ubuntu/external/homecredit/file2' ]
cont.transferFilesToWorker(files)
scriptName = '/home/ubuntu/external/homecredit/test.sh'
cont.execOnRemote(scriptName)
resName = ['/home/ubuntu/file3']
cont.retrieveResults(resName)
cont.terminateInstance()
