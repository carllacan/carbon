#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Load data
    
datafolder = 'data'
xs = np.genfromtxt(datafolder + '/dataX.csv', delimiter=',')
ys = np.genfromtxt(datafolder + '/dataY.csv', delimiter=',')
vs =  np.genfromtxt(datafolder + '/inds_crossval.csv')

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


# Choose params

runs = []
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (0,),
              'hidden_layers' : (40, 30, 20),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 5,
              'epochs' : 12,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (0,),
              'hidden_layers' : (15, 12, 8),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.03,
              'batch_size' : 5,
              'epochs' : 8,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (0,),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.5,
              'reg_type' : 'l2',
              'reg_v' : 0.1,
              'batch_size' : 5,
              'epochs' : 20,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (0,),
              'hidden_layers' : (30, 20, 10),
              'dropout':0.03,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 5,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : range(0, 16),
#             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (0,),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.1,
              'batch_size' : 5,
              'epochs' : 1,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : range(0, 16),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (0,),
              'hidden_layers' : (30, 20, 10),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 5,
              'epochs' : 1,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (0,),
              'hidden_layers' : (40, 10),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 12,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : range(0, 16),
              'targets' : (0,),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 12,
              'validation_split' : 0.0,
              'note':''})

runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (3,),
              'hidden_layers' : (40, 30, 20),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 5,
              'epochs' : 12,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (3,),
              'hidden_layers' : (15, 12, 8),
              'dropout':0.00,
              'reg_type' : 'l2',
              'reg_v' : 0.03,
              'batch_size' : 5,
              'epochs' : 8,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (3,),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.5,
              'reg_type' : 'l2',
              'reg_v' : 0.1,
              'batch_size' : 5,
              'epochs' : 20,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (3,),
              'hidden_layers' : (30, 20, 10),
              'dropout':0.03,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 5,
              'epochs' : 10,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : range(0, 16),
#             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (3,),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.1,
              'batch_size' : 5,
              'epochs' : 1,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : range(0, 16),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (3,),
              'hidden_layers' : (30, 20, 10),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 5,
              'epochs' : 1,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : (3, 1, 4, 2, 5, 6, 7, 8, 0),
#            'features' : (4, 2, 5, 3, 6, 7, 8, 9, 1),
              'targets' : (3,),
              'hidden_layers' : (40, 10),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 12,
              'validation_split' : 0.0,
              'note':''})
runs.append({'neurons':'linear',
             'features' : range(0, 16),
              'targets' : (3,),
              'hidden_layers' : (60, 40, 20),
              'dropout':0.05,
              'reg_type' : 'l2',
              'reg_v' : 0.05,
              'batch_size' : 10,
              'epochs' : 12,
              'validation_split' : 0.0,
              'note':''})

# normalization of training data
#a, b = 0, np.max(xs[:,params['features']])
    
for i in range(0, len(feature_names)):
    xs[:,i] -= np.mean(xs[:,i])
    xs[:,i] /= np.std(xs[:,i])
#    xmin = np.min(xs[:,i])
#    xmax = np.max(xs[:,i])
#    xs[:,i] = (b-a)*(xs[:,i] - xmin)/(xmax-xmin)+a
    
#for i in range(len(target_names)):
#    ys[:,i] -= np.mean(ys[:,i])
#    params['note'] = 'mean substracted from ys'
#    ymin = np.min(ys[:,i])
#    ymax = np.max(ys[:,i])
#    ys[:,i] = (b-a)*(ys[:,i] - ymin)/(ymax-ymin)+a

for params in runs:
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
    results = np.zeros((3, nfolds+1))
    
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
                      metrics=[])
        
        # Training
        
        xs_train = xs[vs != fold,:][:,features]
        xs_val = xs[vs == fold,:][:,features] 
        ys_train = ys[vs != fold,:][:,targets]
        ys_val = ys[vs == fold,:][:,targets]
        
        batch_size = params['batch_size']
        epochs = params['epochs']
        validation_split = params['validation_split']
        
        print('Fold {}'.format(fold))
        fit_history = model.fit(xs_train, ys_train, 
                                batch_size = batch_size,
                                epochs = epochs,
                                verbose=1,
    #                            shuffle=False,
    #                            validation_split=0,
    #                            callbacks=[PlotCallback()]
                                )
        
        # Validation
        
        ys_pred = model.predict(xs_val)
                 
        me = np.mean(ys_val-ys_pred)
        rmse = np.sqrt(np.mean((ys_val-ys_pred)**2))
        mae = np.mean(np.abs(ys_val-ys_pred))
        
        results[0,fold-1] = me
        results[1,fold-1] = rmse
        results[2,fold-1] = mae
    
    
    results[0,fold] = np.mean(results[0,:])
    results[1,fold] = np.mean(results[1,:])
    results[2,fold] = np.mean(results[2,:])
    
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
    
    fname = 'results4.csv'
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
    
    
    results_file.write(row + '\n')
    
    results_file.close()

# TODO: log normalization range?
          
          