#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2, l1
from keras.callbacks import Callback

# TODO: try  normalized values with  biases initialized to zero

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

random.seed(137)

# Load data
    
datafolder = 'data'
xs = np.genfromtxt(datafolder + '/dataX.csv', delimiter=',')
ys = np.genfromtxt(datafolder + '/dataY.csv', delimiter=',')
vs =  np.genfromtxt(datafolder + '/inds_crossval.csv')
# MSC-day GPP
feature_names = ['PFTIds',
         'MODIS.MOD11A2.MODLST_Day_1km_QA1.values', #LST day GPP LE
         'MODIS.MOD11A2.MODLST_Night_1km_QA1.values', # LST night GPP LE
         'MODIS.MCD43A4.MODNDWI.values', #NDWI GPP
         'MODIS.MOD11A2.MODLST_Day_1km_QA1_MSC.Max', # MSC-day GPP
         'MODIS.MOD11A2.MODNDVIRg.values', # NDVI times Rg GPP
         'MODIS.MOD13Q1.MOD250m_16_days_EVI_QA1_MSC.Amp', # EVI GPP
         'MODIS.MOD13Q1.MOD250m_16_days_MIR_reflectance_QA1_MSC.Amp', # MIR GPP
         'MODIS.MOD15A2.MODLai_1km_QA1_MSCmatch', # LAI GPP
         'MODIS.MCD43A4.MODEVILST_MSCmatch', # EVIxLST LE
         'MODIS.MCD43A4.MODFPARRg_MSC.Max', # fAPARxRg LE
         'MODIS.MOD11A2.MODEVILST.values_ano', #EVIxLSTday? LE
         'MODIS.MOD11A2.MODLST_Night_1km_QA1.values_ano', # LST-night yearly?
         'Rg', # Rg LE
         'Rpot', # Rpot LE
         'oriLongTermMeteoData.Rg_all_MSC.Min' # Rg-MSC-min LE
         ]

target_names = ['GPP',
                'NEE',
                'TER',
                'LE',
                'Rn',
                'H']

all_features = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
gpp_features = (1, 2, 3, 4, 5, 6, 7, 8)
le_features = (1, 2, 9, 10, 11, 13, 14, 15)

# Choose params

runs = []

# GPP

runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0,),
              'hidden_layers' : (20, 15, 10),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.01,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0,),
              'hidden_layers' : (30, 20, 15),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0,),
              'hidden_layers' : (40, 30, 20),
              'dropout':0.01,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0,),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.03,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0,),
              'hidden_layers' : (50, 50, 50, 40, 40, 30),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' :10,
              'epochs' : 15,
              'validation_split' : 0.0,
              'note':''})
    
# LE
    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3,),
              'hidden_layers' : (20, 15, 10),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.01,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3,),
              'hidden_layers' : (30, 20, 15),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3,),
              'hidden_layers' : (40, 30, 20),
              'dropout':0.01,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3,),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.03,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
#    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3,),
              'hidden_layers' : (80, 60, 40, 20),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' :10,
              'epochs' : 15,
              'validation_split' : 0.0,
              'note':''})
    
# ALL CARBON

runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0, 1, 2),
              'hidden_layers' : (20, 15, 10),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.01,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0, 1, 2),
              'hidden_layers' : (30, 20, 15),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0, 1, 2),
              'hidden_layers' : (40, 30, 20),
              'dropout':0.01,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0, 1, 2),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.03,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : gpp_features, 
              'targets' : (0, 1, 2),
              'hidden_layers' : (50, 50, 50, 40, 40, 30),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' :10,
              'epochs' : 15,
              'validation_split' : 0.0,
              'note':''})
    
    
# ALL ENERGY

runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3, 4, 5),
              'hidden_layers' : (20, 15, 10),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.01,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3, 4, 5),
              'hidden_layers' : (30, 20, 15),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3, 4, 5),
              'hidden_layers' : (40, 30, 20),
              'dropout':0.01,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3, 4, 5),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.03,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : le_features, 
              'targets' : (3, 4, 5),
              'hidden_layers' : (80, 60, 40, 20),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' :10,
              'epochs' : 15,
              'validation_split' : 0.0,
              'note':''})
    
# ALL 
    
runs.append({'neurons':'linear',
              'features' : all_features, 
              'targets' : (0, 1, 2, 3, 4, 5),
              'hidden_layers' : (20, 15, 10),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.01,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : all_features, 
              'targets' : (0, 1, 2, 3, 4, 5),
              'hidden_layers' : (30, 20, 15),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : all_features, 
              'targets' : (0, 1, 2, 3, 4, 5),
              'hidden_layers' : (40, 30, 20),
              'dropout':0.01,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : all_features, 
              'targets' : (0, 1, 2, 3, 4, 5),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.03,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
    
runs.append({'neurons':'linear',
              'features' : all_features, 
              'targets' : (0, 1, 2, 3, 4, 5),
              'hidden_layers' : (50, 50, 50, 40, 40, 30),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' :10,
              'epochs' : 15,
              'validation_split' : 0.0,
              'note':''})
    
    
    
# normalization of training data
    
for i in range(0, len(feature_names)):
    xs[:,i] -= np.mean(xs[:,i])
    xs[:,i] /= np.std(xs[:,i])
    
    
for i in range(0, len(target_names)):
    ys[:,i] -= np.mean(ys[:,i])
#    ys[:,i] /= np.std(ys[:,i])
    
for r, params in enumerate(runs):
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
    
    nfolds = np.unique(vs).size
    results = np.zeros((3, nfolds+1, len(targets)))
    print('')
    for fold in range(1, nfolds+1):
        # Model definition
        
        model = Sequential()
        model.add(Dense(hidden_layers[0], 
                        input_dim = input_dim, 
                        bias_initializer="zeros", 
                        kernel_initializer="normal", 
                        activation=params['neurons'], 
                        kernel_regularizer=reg))
        for neurons in hidden_layers[1:]:
            model.add(Dense(neurons, 
                            bias_initializer="zeros", 
                            kernel_initializer="normal", 
                            activation=params['neurons'], 
                            kernel_regularizer=reg))
            model.add(Dropout(params['dropout']))  
        model.add(Dense(output_dim, 
                        bias_initializer="zeros", 
                        kernel_initializer="normal", 
                        activation=params['neurons'],
                        kernel_regularizer=reg))
                                
        model.compile(loss='mse', 
                      optimizer="adam", 
                      metrics=[])
        
        # Training
        
        xs_train = xs[vs != fold,:][:,features]
        xs_val = xs[vs == fold,:][:,features] 
        ys_train = ys[vs != fold,:][:,targets]
        ys_val = ys[vs == fold,:][:,targets]
        
        batch_size = params['batch_size']
        epochs = params['epochs']
        validation_split = params['validation_split']
        
        print('Run {}/{}, split {}/{}'.format(r+1, len(runs), fold, nfolds))
        fit_history = model.fit(xs_train, ys_train, 
                                batch_size = batch_size,
                                epochs = epochs,
                                verbose=1,
    #                            shuffle=False,
    #                            validation_split=0,
    #                            callbacks=[PlotCallback()]
                                )
        
        # Validation
        
        for t, tar in enumerate(targets):
            ys_pred = model.predict(xs_val)
                     
            me = np.mean(ys_val-ys_pred)
            rmse = np.sqrt(np.mean((ys_val-ys_pred)**2))
            mae = np.mean(np.abs(ys_val-ys_pred))
            
            results[0, fold-1, t] = me
            results[1, fold-1, t] = rmse
            results[2, fold-1, t] = mae
        
    for t, tar in enumerate(targets):
        results[0,fold,t] = np.mean(results[0,:,t])
        results[1,fold,t] = np.mean(results[1,:,t])
        results[2,fold,t] = np.mean(results[2,:,t])
        
    # Print results
    
    print('')
    
    for t, tar in enumerate(targets):
        print('ME_{0} \t\t RMSE_{0} \t MAE_{0}'.format(tar))
        for fold in range(0, nfolds):
            l = ''
            for m in (0, 1, 2):
                v = results[m,fold,t]
                l +='{:f} \t'.format(v)
            print(l)
        
                
    # Log results
        
    param_names = params.keys()
    delimiter = ','
    
#    fname = 'resultsSO_6d1.csv'
#    fname = 'resultsMO_2d3.csv'
    fname = 'results_fixed_seed.csv'
    try:
        results_file = open(fname, 'r+')
        results_file.read()
    except:
        results_file = open(fname, 'w')
        header = ''
        for p in param_names:
            header += p + delimiter
        
        for t, tar in enumerate(targets):
            for m in  ('ME', 'RMSE', 'MAE'):
                header += m + '_' + str(tar) + delimiter # TODO: use format()
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
        
    for t, tar in enumerate(targets):
        for m in (0,1,2):
            v = np.mean(results[m,:,t])
            row +='{:f}'.format(v) + delimiter
        
    
    results_file.write(row + '\n')
    
    results_file.close()
          
          