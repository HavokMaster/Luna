import os
import subprocess
import psutil

class FileSystemManager:
    def ls(self):
        dir = os.listdir()
        dirContents = []
        print("Currently in:", os.getcwd())
        for idx, file in enumerate(dir):
            fileType = "File" if os.path.isfile(file) else "Directory"
            dirContents.append((file, fileType, idx+1))
    
    def cd(self, dirpath):
        os.chdir(dirpath)

    def goToParentDir(self):
        os.chdir(os.path.dirname(os.getcwd()))

    def createDir(self, dirname):
        os.makedirs(dirname)

    #---To Do: Remove Non-empty dirs---#
    def removeDir(self, dirname):
        try:
            os.removedirs(dirname)
        except Exception as e:
            print(e)
    
    def createFile(self, filename):
        try:
            with open(filename, 'x') as file:
                pass
        except FileExistsError:
            print("File with same name already exists!")

    def readFile(self, filename):
        try:
            with open(filename, 'r') as file:
                data = file.read()
            return data
        except Exception as e:
            print(e)

    def writeToFile(self, filename, data):
        try:
            with open(filename, 'w') as file:
                file.write(data)
        except Exception as e:
            print(e)
    
    def appendToFile(self, filename, data):
        try:
            with open(filename, 'a') as file:
                file.write(data)
        except Exception as e:
            print(e)
    
    def remove(self, filename):
        os.remove(filename)

class ProcessManager:
    
    def runCommand(self, command):
        return subprocess.run(command, shell=True, capture_output=True, text=True)
    
    def checkProcessStatus(self, pid):
        process = psutil.Process(pid)
        return (process.name(), pid, process.cpu_percent(), process.memory_info())
    
    def checkAllProcesses(self):
        status = []
        for process in psutil.process_iter():
            status.append((process.name(), process.pid, process.cpu_percent(), process.memory_info()))
        return status
    def terminateProcess(self, pid):
        process = psutil.Process(pid)
        process.kill()
    
    def setEnv(self, varname, val):
        os.environ[varname] = val
    
    def getEnv(self, varname):
        return os.environ.get(varname)

    def removeEnv(self, varname):
        os.environ.pop(varname, None)