import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy
import os.path

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#Read in white wine data
#white = read_csv("iris.csv", sep=';')

# Read in red wine data 
#red = ___________("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-qualiti/winequalitiddaadfada, sep=';')
