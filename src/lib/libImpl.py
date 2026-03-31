from RecognitionLib import *

path = "../trainedModel.sav" #Définition du chemin du model
clf = loadModel(path) #Chargement du model

print(predict(clf, "../../audio/ok.wav"))#Predicition
