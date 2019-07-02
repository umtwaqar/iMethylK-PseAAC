import numpy
import pickle
from webApp.settings import PROJECT_ROOT
import os

class modelLoader:
    model = None
    def loadModel(self):

        model = pickle.load(open(os.path.join(PROJECT_ROOT, 'static/model.pkl'), 'rb'))
        return model

    def loadScaler(self):
        PROJECT_ROOT =os.path.abspath(os.path.dirname(__file__))
        scaler = []
        means = []
        with open(os.path.join(PROJECT_ROOT, 'static/Scale.txt'), 'r') as filehandle:
            for line in filehandle:
                currentPlace = line[:-1]
                scaler.append(float(currentPlace))

        with open(os.path.join(PROJECT_ROOT, 'static/Mean.txt'), 'r') as filehandle:
            for line in filehandle:
                currentPlace = line[:-1]
                means.append(float(currentPlace))
        return scaler, means