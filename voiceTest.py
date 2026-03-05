'''Dataset link
https://zenodo.org/record/2867216#.XpuGXsgzaUl
'''

from src.lib.RecognitionLib import *
#print("hello")
def testVoice():
    import os
    audio_path = "upload/test.wav"
    if not os.path.exists(audio_path):
        return 'Healthy', 'No voice file uploaded. Please upload a recording first.'
    
    path = "src/trainedModel.sav" #Définition du chemin du model
    clf = loadModel(path) #Chargement du model

    return(predict(clf, audio_path)) # Returns (label, pattern, accuracy)