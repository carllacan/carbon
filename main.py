#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, l1
#from keras.models import load_model
from keras.callbacks import Callback

from keras import backend as K

class PlotCallback(Callback):
    # Plots accuracy and loss after each epoch
    def on_train_begin(self, logs={}):
        self.ls_history = []
        self.as_history = []

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
     
def mean_error(y_true, y_pred):
    return K.mean(y_pred - y_true)
   
def root_mean_squared_error(y_true, y_pred):
    # keras.backend has a square function, but with it the the calculated
    # error is less accurate than with python's ** operator. Go figure.
    return K.sqrt(K.mean((y_pred - y_true)**2))



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

metrics=[mean_error, 
         root_mean_squared_error, 
#         'mean_squared_error', 
         'mean_absolute_error']
metric_abbs = {'mean_squared_error':'MSE',
               'mean_absolute_error':'MAE',
               'mean_absolute_percentage_error':'MAPE',
               mean_error:'ME',
               root_mean_squared_error:'RMSE'}
metric_keys = {'mean_squared_error':'val_mean_squared_error',
               'mean_absolute_error':'val_mean_absolute_error',
               'mean_absolute_percentage_error':'val_mean_absolute_percentage_error',
               mean_error:'val_mean_error',
               root_mean_squared_error:'val_root_mean_squared_error'}

# normalization of features

val_range = 0, 1
a, b = val_range
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
targets = (0,)


input_dim = len(features)
hidden_layers = (50, 30, 20, 10)
output_dim = len(targets)

reg_type = "l2"
reg_v = 0.15
reg = {"l1":l1,"l2":l2}[reg_type](reg_v)

model = Sequential()
model.add(Dense(hidden_layers[0], 
                input_dim = input_dim, 
                kernel_initializer="normal", 
                activation='sigmoid', 
                kernel_regularizer=reg))
for neurons in hidden_layers[1:]:
    model.add(Dense(neurons, 
                    kernel_initializer="normal", 
                    activation='sigmoid', 
                    kernel_regularizer=reg))
model.add(Dense(output_dim, 
                kernel_initializer="normal", 
                activation='sigmoid',
                kernel_regularizer=reg))
                        
model.compile(loss='mean_squared_error', 
              optimizer="adam", 
              metrics=metrics)

print("Training with ", hidden_layers)

batch_size = 50
epochs = 5
validation_split = 0.2
fit_history = model.fit(xs[:,features], ys[:,targets], 
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose=1,
                        shuffle=True,
                        validation_split=validation_split,
#                        callbacks=[PlotCallback()]
                        )
print('')
header = ''

for m in metrics:
    abb = metric_abbs[m]
    header += abb + '\t\t'
print(header)
    
for e in range(epochs):
    l = ''
    for m in metrics:
        v = fit_history.history[metric_keys[m]][e]
        l +='{:f} \t'.format(v)
    print(l)
            



          
          