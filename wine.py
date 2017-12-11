import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np
import os.path


if not os.path.isfile("../../iris.npy"):
  pos = np.loadtxt('../../iris.csv', delimiter=',', dtype=np.float32)
  np.save('../../iris.npy', pos);
else:
  pos = np.load('../../iris.npy')

#Read in white wine data
#white = read_csv("iris.csv", sep=';')

# Read in red wine data 
#red = ___________("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-qualiti/winequalitiddaadfada, sep=';')
