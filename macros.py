from modules import Functions
import pyautogui
import time

f = Functions.FileSystemManager()
p = Functions.ProcessManager()

def openFirefox():
    firefoxDir = p.getEnv('firefox')
    firefoxDir = firefoxDir+'\\firefox.exe'
    p.runCommand(f'"{firefoxDir}"')

def searchWeb(data):
    openFirefox()
    time.sleep(0.5)
    pyautogui.write(data)
    pyautogui.press("enter")
    return "Done!"
    