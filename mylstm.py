#!/usr/bin/env python
import numpy

numpy.random.seed(7)

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Dropout, LSTM, Activation
import datetime
from collections import defaultdict

class MyLSTM():
  #
  #
  
  def __init__(self, classes=25, batch=12, steps=6, feat=12, kfolding=1):
    # Initializes all the variables from my LSTM class
    # number_classes: number of classes, in my case users
    # batch_size: total length of each sequence
    # timesteps: length of sub-sequence
    # features: number of features or attributes of each application accessed (name of application, day,
    #   month, duration, ...
    # kfold: cross-validation. The number of instances of the dataset is divided by this value, the bigger the number, the smaller the number of test instances for each fold.
    # all_dataset: contains all the information
    # rnd_all: contains values to randomize the order the sequences are entered into our customized dataset and so the order of the training dataset. This avoids training for sequences of the same class in order.
    # instances: number of instances or total number of application accesses from all users/classes
    # sequences: total number of sequences. A sequence is a group of application accesses
    # test: number of instances of the test dataset
    # train: number of instances of the train dataset
    # train_x: initial training dataset
    # test_x: initial testing dataset
    # new_batch: final batch size to reorganize the data taking the timestep / batch_size difference
    # new_len_train: final number of training instances
    # new_len_test: final number of testing instances
    # train_y: the target class for each training sequence (although repeated batch_size times)
    # test_y: the target class for each testing sequence (although repeated batch_size times)
    # train_y_cat: used to fulfill the format requirement for a categorical or multi-class model
    # test_y_cat: used to fulfill the format requirement for a categorical or multi-class model
    # new_train_x: temporary training dataset, with new_len_train application accesses and features
    # new_test_x: temporary testing dataset, with new_len_test application accesses and features
    # final_train_x: final training dataset
    # final_test_x: final testing dataset
    # model: model used to train the LSTM
    self.number_classes = classes
    self.batch_size = batch
    self.timesteps = steps
    self.features = feat
    self.kfold = kfolding
    self.all_dataset = numpy.loadtxt("/home/epiferre/top10-lstm-seq12.csv", delimiter=",")
    self.rnd_all = numpy.loadtxt("/home/epiferre/rand6800.csv", delimiter=",", dtype=int)
    self.instances = self.all_dataset.shape[0]
    self.sequences = self.instances/self.batch_size
    self.test = self.setTest()
    self.train = self.instances - self.test
    self.train_x = numpy.zeros([self.train, self.features])
    self.test_x = numpy.zeros([self.test, self.features])
    self.new_batch = self.batch_size - self.timesteps + 1
    self.new_len_train = self.new_batch * (self.train / self.batch_size)
    self.new_len_test = self.new_batch * (self.test / self.batch_size)
    self.train_y = numpy.zeros([self.new_len_train, 1])
    self.test_y = numpy.zeros([self.new_len_test, 1])
    self.train_y_cat = keras.utils.to_categorical(self.train_y, self.number_classes)
    self.test_y_cat = keras.utils.to_categorical(self.test_y, self.number_classes)
    self.new_train_x = numpy.zeros([self.new_len_train, self.timesteps * self.features])
    self.new_test_x = numpy.zeros([self.new_len_test, self.timesteps * self.features])
    self.final_train_x = self.new_train_x.reshape((self.new_train_x.shape[0], self.timesteps, self.features))
    self.final_test_x = self.new_test_x.reshape((self.new_test_x.shape[0], self.timesteps, self.features))
  
  
  def setKFold(self, kf):
    # Maybe then startDataset not needed?
    self.kfold = kf
    self.test = setTest()
    self.train = self.instances - self.test
    self.train_x = numpy.zeros([self.train, self.features])
    self.test_x = numpy.zeros([self.test, self.features])
    self.new_len_train = self.new_batch * (self.train / self.batch_size)
    self.new_len_test = self.new_batch * (self.test / self.batch_size)
    self.train_y = numpy.zeros([self.new_len_train, 1])
    self.test_y = numpy.zeros([self.new_len_test, 1])
    self.train_y_cat = keras.utils.to_categorical(self.train_y, self.number_classes)
    self.test_y_cat = keras.utils.to_categorical(self.test_y, self.number_classes)
    self.new_train_x = numpy.zeros([self.new_len_train, self.timesteps * self.features])
    self.new_test_x = numpy.zeros([self.new_len_test, self.timesteps * self.features])
    self.final_train_x = self.new_train_x.reshape((self.new_train_x.shape[0], self.timesteps, self.features))
    self.final_test_x = self.new_test_x.reshape((self.new_test_x.shape[0], self.timesteps, self.features))
  
  
  def setTest(self):
    # Defines the number of test instances
    # It has to be multiple to the batch_size, and this is validated in the while loop
    test = 0
    if (self.kfold == 1) : test = int(self.instances*0.3)
    else : test = self.instances/self.kfold
    while (test % self.batch_size != 0) :
      test -= 1
    return test
  
  
  def startDataset(self):
    # Initializes the variables train_x, test_x, train_y, test_y, new_train_x, new_test_x. This function is not used when creating the MyLSTM, but when starting each cross-validation
    self.train_x = numpy.zeros([self.train, self.features])
    self.test_x = numpy.zeros([self.test, self.features])
    self.train_y = numpy.zeros([self.new_len_train, 1])
    self.test_y = numpy.zeros([self.new_len_test, 1])
    self.new_train_x = numpy.zeros([self.new_len_train, self.timesteps * self.features])
    self.new_test_x = numpy.zeros([self.new_len_test, self.timesteps * self.features])
  
  
  def setTestOffset(self, kf):
    # Verifies the test offset is the required size. Fixing possible problem from setTest when trying to be modulo batch_size
    offset = 0
    if (kf == self.kfold - 1) : offset = (self.instances - self.test) / self.batch_size
    else : offset = (kf * self.test) / self.batch_size
    return offset
  
  
  def setupDataset(self, kf):
    # Sets up the variables train_x and test_x with random sequences of data from all_dataset, using rnd_all
    # For the first kfold, the test_x will take sequences from random position 0 to random position test-1, being test the number of instances to put for testing in mylstm. For the second kfold, test_x will take sequences from test to 2*test -1. And so on. The last kfold is managed differently to assure the number of testing instances. We are trying to have a number of instances multiple of the sequence size. It might need to be fixed
    
    train_offset = 0
    test_offset = self.setTestOffset(kf)
    
    # The loop iterates through the number of training sequences, which equals train / batch_size
    # This loops sets up train and test dataset at the same time for each loop iteration, avoiding extra iterations if setting up train and test datasets separately
    for i in range(0, self.train / self.batch_size):
      # When this conditional is true, it means that the train and test offsets are in the same position and would copy the same data in the train and test dataset, and we need to avoid this
      if (train_offset == test_offset) : train_offset += self.test / self.batch_size
      
      # While the iteration number i is smaller than total number of testing sequences, we put the testing data into our testing dataset
      if (i < self.test / self.batch_size) :
        self.test_x[i*self.batch_size : (i+1)*self.batch_size] = self.all_dataset[self.rnd_all[test_offset+i]*self.batch_size:(self.rnd_all[test_offset+i]+1)*self.batch_size, 0:self.features]
      
      self.train_x[i*self.batch_size:(i+1)*self.batch_size] = self.all_dataset[self.rnd_all[train_offset]*self.batch_size:(self.rnd_all[train_offset]+1)*self.batch_size, 0:self.features]
      train_offset += 1
    
    print train_offset
  
  
  def setupNewDataset(self, kf):
    # Taking the dataset achieved from calling setupDataset(), create a new dataset reorganizing the data based on timesteps and batch_size
    
    train_offset = 0
    test_offset = self.setTestOffset(kf)
    
    for seq in range (0, self.new_len_train / self.new_batch) :
      # As many iterations as sequences in the training dataset
      
      for bat in range (self.new_batch) :
        # As many iterations as subsequences of the sequence. Each subsequence is put in train_subsequence and test_subsequence
        
        train_subsequence = []
        test_subsequence = []
        
        for tim in range (self.timesteps) : #FROM 0 TO 6
          # As many iterations as application accesses or instances each subsequence has. Each instance is added to the subsequence array
          
          train_subsequence.extend(self.train_x[bat+tim+(seq*self.batch_size),0:self.features])
          
          # The following conditional controls the subsequences to be put in the final testing dataset, as there are less testing sequences than training sequences
          if ((seq * self.new_batch) < self.new_len_test) : test_subsequence.extend(self.test_x[bat+tim+(seq*self.batch_size),0:self.features])
        
        self.new_train_x[bat + (seq * self.new_batch)] = train_subsequence
        
        # The following conditional controls the subsequences to be put in the final testing dataset, as there are less testing sequences than training sequences
        if ((seq * self.new_batch) < self.new_len_test) : self.new_test_x[bat + (seq * self.new_batch)] = test_subsequence
      
      if (train_offset == test_offset) : train_offset += self.test / self.batch_size
      self.train_y[seq * self.new_batch:(seq+1) * self.new_batch] = self.all_dataset[self.rnd_all[train_offset]*self.batch_size, self.features]
      
      train_offset += 1
      if ((seq * self.new_batch) < self.new_len_test) :
        self.test_y[seq * self.new_batch:(seq+1) * self.new_batch] = self.all_dataset[self.rnd_all[seq+test_offset]*self.batch_size, self.features]
    
    # Finally, we need to reshape the training and testing datasets to fulfill the requirements for LSTMs using Keras
    self.final_train_x = self.new_train_x.reshape((self.new_train_x.shape[0], self.timesteps, self.features))
    self.final_test_x = self.new_test_x.reshape((self.new_test_x.shape[0], self.timesteps, self.features))


  def setupY(self):
    self.train_y_cat = keras.utils.to_categorical(self.train_y, self.number_classes)
    self.test_y_cat = keras.utils.to_categorical(self.test_y, self.number_classes)


class MyModel():
  #
  #
  
  def __init__(self, myepochs=1, mybatch=1, myneurons=256, mydropout=0.2, mylstm=MyLSTM()):
    # epochs: Number of epochs of my model
    # batch: batch size used to train and test my model
    # neurons: number of neuron units from my model's LSTM
    # dropout: the value of dropout - used to fight against overfitting and unbalanced data
    # mlp: Multi-layer perceptron used to make the predictions
    # lstm: My LSTM
    # weights: weights assigned to each class. Used to manage unbalanced data
    # model: Keras model used
    # history: data returned when fitting the model
    # preditions: class predictions returned when testing the model with the testing data
    # threshold: used to calculate the EER (Equal Error Rate)
    # values: used to store the True Positive, True Negative, False Positive and False Negative of each class and threshold, together with their rates, which will be used to calculate the EER
    # accuracy:
    # loss:
    # eers:
    self.epochs = myepochs
    self.batch = mybatch
    self.neurons = myneurons
    self.dropout = mydropout
    self.mlp = 'softmax'
    self.lstm = mylstm
#    self.weights = defaultdict(int)
    self.weights = []
    self.model = Sequential()
    self.history = []
    self.predictions = [[]]
    self.threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    self.values = [[{"tp": 0, "tn": 0, "fp": 0, "fn": 0, "tpr": 0, "tnr": 0, "fpr": 0, "fnr": 0} for t in range(len(self.threshold))] for c in range(self.lstm.number_classes)]
    self.accuracy = []
    self.loss = []
    self.eers = []
  
  
  def setLSTM(self, mylstm):
    #
    self.lstm = mylstm
  
  
  def setEpochs(self, myepochs):
    #
    self.epochs = myepochs
  
  
  def setBatch(self, mybatch):
    #
    self.batch = mybatch
  
  
  def setNeurons(self, myneurons):
    #
    self.neurons = myneurons


  def createModel(self):
    # The model consists of two LSTM layers, each containing a Dropout and an Activation(relu)
    # After the two-layer LSTM model, the model adds a Multi-layer Perceptron, the Activation(softmax), to predict the class for each sequence
    
    self.model = Sequential()
    self.model.add(LSTM(self.neurons, input_shape=(self.lstm.final_train_x.shape[1], self.lstm.final_train_x.shape[2]), return_sequences=True, implementation = 2))
    self.model.add(Dropout(self.dropout))
    self.model.add(Activation('relu'))
    
    self.model.add(LSTM(self.neurons, input_shape=(self.lstm.final_train_x.shape[1], self.lstm.final_train_x.shape[2]), return_sequences=False, implementation = 2))
    self.model.add(Dropout(self.dropout + 0.1))
    self.model.add(Activation('relu'))
    
    self.model.add(Dense(self.lstm.number_classes, activation=self.mlp))
  
  
  def compile(self, myloss='categorical_crossentropy', myadam='adam'):
  #
    self.model.compile(loss=myloss, optimizer=myadam, metrics=['accuracy'])


  def initWeights(self):
    # Initializes the weigths to 0
    for i in range(self.lstm.number_classes):
      self.weights.append(0.0)


  def setupWeights(self):
    # Sets up the weights by:
    # - finding the maximum number of class appearances
    # - for each class, dividing the previous maximum by the number of class appearances
    self.initWeights()

    # Count the number of class appearances for each class
    for i in range(0, len(self.lstm.train_y), self.lstm.new_batch) :
      #print ("({}-{})".format(str(i), self.lstm.train_y[i])),
      self.weights[int(self.lstm.train_y[i])] += 1.0

    # Find the class with more appearances
    max = 0.0
    for i in range(self.lstm.number_classes) :
      if (max < self.weights[i]) : max = self.weights[i]

    # Set up weights for each class
    for i in range(self.lstm.number_classes) :
        self.weights[i] = max / self.weights[i]


  def fit(self):
    # Fits the model to train, using the xtrain variable from our MyLSTM, comparing its predictions with the right class stored in train_y_cat from our MyLSTM (and learning from this right/wrong predictions) and with the our model's epochs, batch and recently calculated weights. We do not want to shuffle, because we already shuffle it manually
    self.history = self.model.fit(self.lstm.final_train_x, self.lstm.train_y_cat, epochs=self.epochs, batch_size=self.batch, shuffle=False, class_weight = self.weights)


  def predict(self):
    #
    self.predictions = self.model.predict(self.lstm.final_test_x, batch_size=self.batch)


  def prepareEER(self):
    # The easiest way to explain this is by using an example
    # The model will predict a probability percentage for a certain sequence to belong to a certain class, for each sequence and class
    # Therefore, the predictions variable will be a 2-dimension array (sequencesxclasses)
    # Say prediction that sequences 1 belongs to class A is 0.8. It means the model is 80% sure that is the case
    # When calculating the EER, if indeed the right class is A, the prediction will be a TP, until our threshold value is bigger than the prediction value. When that happens, the prediction becomes a FN.
    # If the right class happens to be B, the prediction would be a FP while (again) the threshold is smaller than the prediction value. When the prediction value is smaller than the threshold, then it turns into a TN.
    for pr in range(self.lstm.new_len_test / self.lstm.new_batch):
      # For each prediction made in the testing phase with the testing data, with as many predictions as sequences
      for cl in range(self.lstm.number_classes):
        # For each class
        for er in range(len(self.threshold)):
          # For each threshold value
          # If the class (cl) being evaluated is the same as the correct class for a sequence (set in test_y), then the prediction can only be either a True Positive or a False Negative:
          if (cl == self.lstm.test_y[pr*self.lstm.new_batch]) :
            # If the prediction value for a class and sequence is bigger than the threshold, increase number of TP. Otherwise, increase number of FN.
            if (self.predictions[pr*self.lstm.new_batch][cl] >= self.threshold[er]) : self.values[cl][er]["tp"] += 1
            else : self.values[cl][er]["fn"] += 1
          
          else :
            # If the prediction value for a sequence and the wrong class is bigger than the threshold, increase number of FP. Otherwise, increase number of TN.
            if (self.predictions[pr*self.lstm.new_batch][cl] >= self.threshold[er]) : self.values[cl][er]["fp"] += 1
            else : self.values[cl][er]["tn"] += 1


  def evaluate(self):
    # Gets accuracy and loss values
    return self.model.evaluate(self.lstm.final_test_x, self.lstm.test_y_cat, batch_size=self.batch)
  
  
  def countClassInstances(self, myclass):
    # Count the number of instances or appearances from the class myclass
    count = 0
    for pos in range(0, len(self.lstm.test_y), self.lstm.new_batch) :
      if (self.lstm.test_y[pos] == myclass) : count += 1
    return count


  def calculateEER(self):
    # Calculates the EER for a dataset testing phase
    # pot_eer adds each class EER and then calculate the average
    # eer stores the EER for each class
    # num_appearances counts and saves the number of appearances of each class in the testing dataset
    pot_eer = 0.0
    eer = numpy.zeros((self.lstm.number_classes))
    num_appearances = numpy.zeros((self.lstm.number_classes))
    
    for cl in range(self.lstm.number_classes) :
      # For each class, find the intersection point between the FPR and FNR
      # The variable intersection is used as a boolean (0, 1) to know if there has been an intersection or not.
      diff = []
      num_appearances[cl] = self.countClassInstances(cl)
      intersection = 0
      
      #mydata += "\nUSER " + str(i) + ": " + str(int(num_appearances[i]))
#      print "Class " + str(cl) + ":  "

      for e in range(len(self.threshold)) :
        # Calculate the EER rates (TPR, TNR, FPR, FNR) for each threshold
        self.values[cl][e]["tpr"] = float(self.values[cl][e]["tp"])/(self.values[cl][e]["tp"] + self.values[cl][e]["fn"])
        self.values[cl][e]["fnr"] = float(self.values[cl][e]["fn"])/(self.values[cl][e]["tp"] + self.values[cl][e]["fn"])
        self.values[cl][e]["tnr"] = float(self.values[cl][e]["tn"])/(self.values[cl][e]["fp"] + self.values[cl][e]["tn"])
        self.values[cl][e]["fpr"] = float(self.values[cl][e]["fp"])/(self.values[cl][e]["fp"] + self.values[cl][e]["tn"])
        
        # Considering the FNR will always be initially bigger than the FPR, when the FPR >= FNR it means that it changed, and there has been an intersection
        if (self.values[cl][e]["fpr"] >= self.values[cl][e]["fnr"]):
          intersection = 1
        
#        print "FPR: " + str("%5.3f%%" % (100*float(self.values[cl][e]["fpr"]))),
#        print " | FNR: " + str("%5.3f%%" % (100*float(self.values[cl][e]["fnr"])))

        # Append the absolute difference between the FPR and the FNR value for each threshold
        diff.append(abs(self.values[cl][e]["fpr"] - self.values[cl][e]["fnr"]))
        
        # Unused storing to mydata
#        mydata += "\nEER(" + str(e+1) + "): "
#        mydata += "TPR: " + str("%5.3f%%" % (100*float(self.values[cl][e]["tpr"])))
#        mydata += " | TNR: " + str("%5.3f%%" % (100*float(self.values[cl][e]["tnr"])))
#        mydata += " | FPR: " + str("%5.3f%%" % (100*float(self.values[cl][e]["fpr"])))
#        mydata += " | FNR: " + str("%5.3f%%" % (100*float(self.values[cl][e]["fnr"])))
      
      # If there is no intersection, the EER for class 'cl' will be 0. Otherwise, we approximate the EER intersection by finding the minimum absolute difference point stored in the 'diff' variable, and calculating the average between the FPR and FNR at that particular point
      if (intersection == 0) : eer[cl] = 0.0
      else :
#        print diff
        min_pos = diff.index(min(diff))
        eer[cl] = (self.values[cl][min_pos]["fpr"] + self.values[cl][min_pos]["fnr"]) / 2
      
#      print "EER for class:  " + str("%5.3f%%" % (100*float(eer[cl])))
      #mydata += "\nEER for class:  " + str("%5.3f%%" % (100*float(eer[i]))) + "\n"

    # Add each eer for all the classes together and then calculate the average
    for ee in eer :
      pot_eer += ee

    pot_eer = 100*(pot_eer/self.lstm.number_classes)
    self.eers.append(pot_eer)


  def finalCalculation(self):
    # Calculate the final average accuracy, loss and eer for the experiment, adding it to the accuracy, loss and eers arrays in the last position
    self.accuracy.append(0.0)
    self.loss.append(0.0)
    self.eers.append(0.0)

    for kf in range(self.lstm.kfold) :
      self.accuracy[self.lstm.kfold] += self.accuracy[kf]
      self.loss[self.lstm.kfold] += self.loss[kf]
      self.eers[self.lstm.kfold] += self.eers[kf]

    self.accuracy[self.lstm.kfold] /= self.lstm.kfold
    self.loss[self.lstm.kfold] /= self.lstm.kfold
    self.eers[self.lstm.kfold] /= self.lstm.kfold


def saveHistory(history) :
  # Function to manage and save the model history, to be used later on to put into a file and save
  myh = ""

  for i in history.history['acc'] :
    myh += "\n" + str(i)

  myh += "\n\n"

  for i in history.history['loss']:
    myh += "\n" + str(i)

  myh += "\n\n"

  return myh


def saveMyFiles(mydata, myhistory):
  #Function to save both mydata and myhistory. The goal of saving myhistory is to plot it (future work) to see the evolution of the training phase
  file = open("/home/epiferre/top-eer-kf-ts6-bs12.txt","a+")
  file.write(mydata)
  file.close()

  file = open("/home/epiferre/top-eer-kf-ts6-bs12-plot.txt","a+")
  myhistory += "\n---------------------------"
  myhistory += "\n---------------------------"
  file.write(myhistory)
  file.close()


def main():
  # Create my lstm
  # Params: classes=25, batch=12, steps=6, features=12, kfolding=1
  print ("Creating my LSTM...")
  mylstm = MyLSTM(25, 12, 6, 12, 4)
  
  # Create my model using mylstm
  # Params: myepochs=1, mybatch=1, myneurons=256, mydropout=0.2, mylstm=MyLSTM()
  print ("Creating my Model...")
  mymodel = MyModel(1, 1, 256, 0.2, mylstm)
  
  # Create a variable "mydata" that will store all the important information we want to keep
  mydata = "\n-----------------------------------------------------------"
  mydata += datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
  myhistory = "TS: " + str(mylstm.timesteps) + "  BS: " + str(mylstm.batch_size)
  myhistory += "\n---------------------------"

  # Start cross-validation loop
  for kf in range(mymodel.lstm.kfold):

    print " ---KFOLD " + str(kf+1) + " out of " + str(mymodel.lstm.kfold)
    print "Putting data into arrays..."

    #Initialize variables train_x and test_x
    mymodel.lstm.startDataset()

    # Set up the sequence randomly for training and testing
    mymodel.lstm.setupDataset(kf)

    print "Transforming data for LSTMs..."

    # Reorganize the dataset
    mymodel.lstm.setupNewDataset(kf)
      
    mydata += "\n-----------------------------------------------------------"
    mydata += "\n\nOur trainx / trainy / testx / testy shapes are:\n" + str(mymodel.lstm.final_train_x.shape) + " " + str(mymodel.lstm.train_y.shape) + " " + str(mymodel.lstm.final_test_x.shape) + " " + str(mymodel.lstm.test_y.shape) + "\n"

    print "Creating the model..."

    # Set up my model batch size, before start compiling and training my model
    mymodel.setBatch(mymodel.lstm.new_batch * mymodel.lstm.timesteps * mymodel.lstm.features)

    mymodel.createModel()

    # Variables 'learn' and 'adam_opt' are not used so far
    learn = 0.001
    adam_opt = keras.optimizers.Adam(lr=learn, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mymodel.compile()
    
    # Set up again the target to fulfill the categorical predictions format
    mymodel.lstm.setupY()
  
    mymodel.setupWeights()
    print "Weights are set up:\n{}".format(mymodel.weights)

    print "Fitting the model..."

    mymodel.fit()

    mymodel.predict()

    myhistory += saveHistory(mymodel.history)

    # Save some information of the test into mydata variable to print it later
    mydata += "\nThe model :\n -LSTM " + str(mymodel.neurons)
    mydata += ", with input shape " + str(mymodel.lstm.final_train_x.shape[1]) + "," + str(mymodel.lstm.final_train_x.shape[2])
    mydata += "\n -Dropout " + str(mymodel.dropout)
    mydata += "\n -Activation relu\n -LSTM " + str(mymodel.neurons)
    mydata += "\n -Dropout " + str(mymodel.dropout+0.1)
    mydata += "\n -Activation relu\n -Dense " + str(mymodel.lstm.number_classes)
    mydata += ", with activation=softmax"
    mydata += "\nOptimizer = adam, lr = default\nloss = categorical_crossentropy"
    mydata += "\nEpochs = " + str(mymodel.epochs)
    mydata += ", Batch = " + str(mymodel.batch)
    mydata += ", #Classes = " + str(mymodel.lstm.number_classes) + "\n"

    print "Calculating FP, FN, for each THRESHOLD, CLASS, PREDICTION..."

    mymodel.prepareEER()
    
    # Evaluate the model, gets accuracy and prediction
    scores = mymodel.evaluate()
    print scores
    mymodel.accuracy.append(scores[1])
    mymodel.loss.append(scores[0])
    mydata += "\nScores: " + str(scores) + "\n"

    print "Calculating EER..."
    mydata += "\nCalculating EER...\n"

    mymodel.calculateEER()

    print "Potential EER:  " + str("%5.3f%%" % (float(mymodel.eers[kf])))
    mydata += "\nEER for FOLD: " + str("%5.3f%%" % (mymodel.eers[kf])) + "\n\n"
                                       
  mymodel.finalCalculation()
                                       
  mydata += "\n------------------\n------------------\n------------------\n"
  mydata += "\nAccuracy avg: " + str("%5.3f%%" % (100*mymodel.accuracy[mymodel.lstm.kfold]))
  mydata += "\nLoss avg: " + str("%5.3f" % (mymodel.loss[mymodel.lstm.kfold]))
  mydata += "\nEER avg: " + str("%5.3f%%" % (mymodel.eers[mymodel.lstm.kfold])) + "\n\n"

  print mydata
                                       
  saveMyFiles(mydata, myhistory)


if __name__ == "__main__":
  main()

