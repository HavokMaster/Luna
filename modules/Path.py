import os

def loadPath(ProcessManager):
    with open('PATHS', 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line[0] == '#': continue
            data = line.split(' ', 1)
            ProcessManager.setEnv(data[0], data[1])