#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.regularizers import l2, l1
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
     
# Create some custom metrics
        
def mean_error(y_true, y_pred):
    return K.mean(y_pred - y_true)
   
def root_mean_squared_error(y_true, y_pred):
    # keras.backend has a square function, but with it the the calculated
    # error is less accurate than with python's ** operator. Go figure.
    return K.sqrt(K.mean((y_pred - y_true)**2))

# Load data
    
datafolder = 'data'
xs = np.genfromtxt('{}/dataX.csv'.format(datafolder), delimiter=',')
ys = np.genfromtxt('{}/dataY.csv'.format(datafolder), delimiter=',')
val_folds = xs[:,0]
# TODO: load the val inds from where they are
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

# Decide the metrics

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
# Choose params

params = {'neurons':'linear',
#          'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
          'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
          'targets' : (0,),
          'hidden_layers' : (30, 20, 10),
          'dropout':0.0,
          'reg_type' : 'l2',
          'reg_v' : 0.1,
          'batch_size' : 5,
          'epochs' : 10,
          'validation_split' : 0.05}

# normalization of training data
if params['neurons'] == 'sigmoid':
    a, b = 0, 1
elif params['neurons'] == 'tanh':
    a, b = -1, 1
elif params['neurons'] == 'linear':
    a, b = -1, 1
elif params['neurons'] == 'relu':
    a, b = 0, 1
    
for i in range(1, len(feature_names)):
#    xs[:,i] -= np.mean(xs[:,i])
    xmin = np.min(xs[:,i])
    xmax = np.max(xs[:,i])
    xs[:,i] = (b-a)*(xs[:,i] - xmin)/(xmax-xmin)+a
    
#for i in range(len(target_names)):
#    ys[:,i] -= np.mean(ys[:,i])
#    ymin = np.min(ys[:,i])
#    ymax = np.max(ys[:,i])
#    ys[:,i] = (b-a)*(ys[:,i] - ymin)/(ymax-ymin)+a

# Network architechture
    
#features = range(1, 16)
features = params['features']
targets = params['targets']

input_dim = len(features)
hidden_layers = params['hidden_layers']
output_dim = len(targets)

# Regularization
reg_type = params['reg_type']
reg_v = params['reg_v']
reg = {"l1":l1,"l2":l2}[reg_type](reg_v)

nfolds = 2 #update thisssssss
results = np.zeros((3, nfolds))
for fold in range(1, nfolds+1):
    # Model definition
    
    model = Sequential()
    model.add(Dense(hidden_layers[0], 
                    input_dim = input_dim, 
                    kernel_initializer="normal", 
                    activation=params['neurons'], 
                    kernel_regularizer=reg))
    for neurons in hidden_layers[1:]:
        model.add(Dense(neurons, 
                        kernel_initializer="normal", 
                        activation=params['neurons'], 
                        kernel_regularizer=reg))
        model.add(Dropout(params['dropout']))  
    model.add(Dense(output_dim, 
                    kernel_initializer="normal", 
                    activation=params['neurons'],
                    kernel_regularizer=reg))
                            
    model.compile(loss='mse', 
                  optimizer="adam", 
                  metrics=metrics)
    
    # Training
    
    xs_train = xs[val_folds != fold,:][:,features]
    xs_val = xs[val_folds == fold,:][:,features] 
    ys_train = ys[val_folds != fold,:][:,targets]
    ys_val = ys[val_folds == fold,:][:,targets]
    
    batch_size = params['batch_size']
    epochs = params['epochs']
    validation_split = params['validation_split']
    
    print('Fold {}'.format(fold))
    fit_history = model.fit(xs_train, ys_train, 
                            batch_size = batch_size,
                            epochs = epochs,
                            verbose=1,
                            shuffle=False,
#                            validation_split=0,
    #                        callbacks=[PlotCallback()]
                            )
#    results = fit_history.history
    
    # Validation
    
    # TODO: write normalize and denormalize function for matrices
    
    
    #for i in range(len(feature_names)):
    #    xs[:,i] = (xmax-xmin)*(xs[:,i] - a)/(b-a)+xmin
        
#    inds_val = tuple(range(0, 100))
#    xs_val = xs[3000:4500,features]
#    ys_val = ys[3000:4500,targets]
    ys_pred = model.predict(xs_val)
        
    # De-normalize
    #for i in range(len(targets)):
    #    ys_val =  (ymax-ymin)*(ys_val  - a)/(b-a)+ymin
    #    ys_pred = (ymax-ymin)*(ys_pred - a)/(b-a)+ymin
    
        
    me = np.mean(ys_val-ys_pred)
    rmse = np.sqrt(np.mean((ys_val-ys_pred)**2))
    mae = np.mean(np.abs(ys_val-ys_pred))
    
    results[0,fold-1] = me
    results[1,fold-1] = rmse
    results[2,fold-1] = mae

# Print results

#print('')
#header = ''
#
#for m in metrics:
#    abb = metric_abbs[m]
#    header += abb + '\t\t'
#print(header)
#    
#for e in range(epochs):
#    l = ''
#    for m in metrics:
#        v = results[metric_keys[m]][e]
#        l +='{:f} \t'.format(v)
#    print(l)

# Print results

print('')

print('ME \t\t RMSE \t\t MAE')
for fold in range(0, nfolds):
    l = ''
    for m in (0, 1, 2):
        v = results[m,fold]
        l +='{:f} \t'.format(v)
    print(l)

            
# Log results
    
param_names = params.keys()
delimiter = ','

fname = 'results2.csv'
try:
    results_file = open(fname, 'r+')
    results_file.read()
except:
    results_file = open(fname, 'w')
    header = ''
    for p in param_names:
        header += p + delimiter
#    for m in metrics:
#        header += metric_abbs[m] + delimiter
    for m in  ('ME', 'RMSE', 'MAE'):
        header += m + delimiter
    results_file.write(header + '\n')
    
row = ''
for p in param_names:
    v = params[p]
    if type(v) == tuple:
        v = str(v)
        v = v.replace(' ', '')
        v = v.replace('(', '')
        v = v.replace(')', '')
        v = v.replace(',', '-')
    else:
        v = str(v)
    row += v + delimiter
    
for m in (0,1,2):
    v = np.mean(results[m,:])
    row +='{:f}'.format(v) + delimiter
#
#for m in metrics:
#    v = results[metric_keys[m]][-1]
#
#    row += '{:f}'.format(v) + delimiter
results_file.write(row + '\n')

results_file.close()

# TODO: log normalization range
          
          