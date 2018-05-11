#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import random

from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Conv2D
#from keras.layers import Flatten
#from keras.layers import MaxPooling2D
from keras.regularizers import l2, l1
from keras.models import load_model
from keras.callbacks import Callback

from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

class PlotCallback(Callback):
    
#    def __init__(self):
#        Callback.__init()
#        self.acc_history = []
        
    def on_train_begin(self, logs={}):
        self.ls_history = []
        self.as_history = []
 
    def on_train_end(self, logs={}):
        results = zip(self.ls_history, self.as_history)
        print("Loss \t Accuracy")
        for l, a in results:
            print("{} \t {}".format(l, a))
# 
#    def on_epoch_begin(self, logs={}):
#        return
 
    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        acc = logs.get('acc')
        self.ls_history.append(loss)
        self.as_history.append(acc)
        ls = self.ls_history
        acs = self.as_history
        es = np.arange(1, epoch+2)
        plt.figure()
        plt.plot(es, ls)
        plt.plot(es, acs)
        plt.show()
        return
 
#    def on_batch_begin(self, batch, logs={}):
#        return
# 
#    def on_batch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss'))
#        return
    
datafolder = 'data'
xs = np.genfromtxt('{}/dataX.csv'.format(datafolder), delimiter=',')
ys = np.genfromtxt('{}/dataY.csv'.format(datafolder), delimiter=',')

feature_names = ['PFTIds',
         'MODIS.MOD11A2.MODLST_Day_1km_QA1.values',
         'MODIS.MOD11A2.MODLST_Night_1km_QA1.values',
         'MODIS.MCD43A4.MODNDWI.values',
         'MODIS.MOD11A2.MODLST_Day_1km_QA1_MSC.Max',
         'MODIS.MOD11A2.MODNDVIRg.values',
         'MODIS.MOD13Q1.MOD250m_16_days_EVI_QA1_MSC.Amp',
         'MODIS.MOD13Q1.MOD250m_16_days_MIR_reflectance_QA1_MSC.Amp',
         'MODIS.MOD15A2.MODLai_1km_QA1_MSCmatch',
         'MODIS.MCD43A4.MODEVILST_MSCmatch',
         'MODIS.MCD43A4.MODFPARRg_MSC.Max',
         'MODIS.MOD11A2.MODEVILST.values_ano',
         'MODIS.MOD11A2.MODLST_Night_1km_QA1.values_ano',
         'Rg',
         'Rpot',
         'oriLongTermMeteoData.Rg_all_MSC.Min']

target_names = ['GPP',
                'NEE',
                'TER',
                'LE',
                'Rn',
                'H']

# normalization of features

a, b = 0, 1
for i in range(len(feature_names)):
    xmin = np.min(xs[:,i])
    xmax = np.max(xs[:,i])
    xs[:,i] = (b-a)*(xs[:,i] - xmin)/(xmax-xmin)+a
    
for i in range(len(target_names)):
    ymin = np.min(ys[:,i])
    ymax = np.max(ys[:,i])
    ys[:,i] = (b-a)*(ys[:,i] - ymin)/(ymax-ymin)+a

#features = range(0, 16)
features = (3, 1, 4, 2, 5, 6, 7, 8, 0)
targets = (3,)

model = Sequential()

input_dim = len(features)
hidden_layers = (50, 30, 20, 10)
output_dim = len(targets)

rel = lambda: l2(0.2)

model.add(Dense(hidden_layers[0], 
                input_dim = input_dim, 
                kernel_initializer="normal", 
                activation="sigmoid", 
                kernel_regularizer=rel()))
for neurons in hidden_layers[1:]:
    model.add(Dense(neurons, 
                    kernel_initializer="normal", 
                    activation="sigmoid", 
                    kernel_regularizer=rel()))
model.add(Dense(output_dim, 
                kernel_initializer="normal", 
                activation="sigmoid",
                kernel_regularizer=rel()))
                        
model.compile(loss='mean_squared_error', 
              optimizer="adam", metrics=['accuracy'])

print("Training with ", hidden_layers)

model.fit(xs[:,features], ys[:,targets], 
          batch_size=5,
          epochs = 5,
          verbose=1,
          callbacks=[PlotCallback()])

          
          