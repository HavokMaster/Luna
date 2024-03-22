import speech_recognition as sr

def createRecognizer():
    return sr.Recognizer()

def listen(recognizer):
    with sr.Microphone() as source:
        print("Say something")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(text)
        except sr.UnknownValueError:
            print("Could not recognize text")
        except sr.RequestError as e:
            print("Could not request Google Service")

def processVoiceCommand(self, recognizer, f, p):
    command = listen(recognizer).lower()
    
    #FileSystem Commands

    for trigger in ["list items", "ls", "show files", "all files"]:
        if trigger in command:
            print(f.ls())


