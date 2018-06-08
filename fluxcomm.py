#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import random
import numpy as np

import utils

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2, l1

# TODO: try  normalized values with  biases initialized to zero



# Load data
    
datafolder = 'data'
xs = np.genfromtxt(datafolder + '/dataX.csv', delimiter=',')
ys = np.genfromtxt(datafolder + '/dataY.csv', delimiter=',')
vs =  np.genfromtxt(datafolder + '/inds_crossval.csv') -1

# MSC-day GPP?
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

# Set folder
folder = 'results/results_test'
start_at = 1

runs_filename = folder + '/runs.csv'
results_filename = folder + '/results.csv'

# Load the parameters of the runs
runs = utils.load_runs(runs_filename)

nfolds = 2#np.unique(vs).size
    
# Normalization of training data
    
for i in range(0, len(feature_names)):
    xs[:,i] -= np.mean(xs[:,i])
    xs[:,i] /= np.std(xs[:,i])
    
    
for i in range(0, len(target_names)):
    ys[:,i] -= np.mean(ys[:,i])
#    ys[:,i] /= np.std(ys[:,i])
    
for r, params in enumerate(runs[start_at-1:], start=start_at-1):
    # Network architecture
    features = params['features']
    targets = params['targets']
    
    input_dim = len(features)
    hidden_layers = params['hidden_layers']
    output_dim = len(targets)
    
    # Regularization
    reg_type = params['reg_type']
    reg_v = params['reg_v']
    reg = {"l1":l1,"l2":l2}[reg_type](reg_v)
            
    batch_size = params['batch_size']
    epochs = params['epochs']
        
    optimizer = "adam"
    
    
    results = np.zeros((4, nfolds+1, len(targets)))
    print('')
    for fold in range(0, nfolds):
        # Model definition
        
        model = Sequential()
        model.add(Dense(hidden_layers[0], 
                        input_dim = input_dim, 
                        bias_initializer="zeros", 
                        kernel_initializer="normal", 
                        activation='linear', 
                        kernel_regularizer=reg))
        for neurons in hidden_layers[1:]:
            model.add(Dense(neurons, 
                            bias_initializer="zeros", 
                            kernel_initializer="normal", 
                            activation='linear', 
                            kernel_regularizer=reg))
            model.add(Dropout(params['dropout']))  
        model.add(Dense(output_dim, 
                        bias_initializer="zeros", 
                        kernel_initializer="normal", 
                        activation='linear',
                        kernel_regularizer=reg))
                                
        model.compile(loss='mse', 
                      optimizer=optimizer, 
                      metrics=[])
        
        # Training
        
        xs_train = xs[vs != fold,:][:,features]
        xs_val = xs[vs == fold,:][:,features] 
        ys_train = ys[vs != fold,:][:,targets]
        ys_val = ys[vs == fold,:][:,targets]

        
        print('Run {}/{}, split {}/{}'.format(r+1, len(runs), fold, nfolds))
        fit_history = model.fit(xs_train, ys_train, 
                                batch_size = batch_size,
                                epochs = epochs,
                                verbose=1,
                                )
        
        # Validation
        
        ys_pred = model.predict(xs_val)
        for t, tar in enumerate(targets):
            ys_val_t = ys_val[:,t]      
            ys_pred_t = ys_pred[:,t]      
            me = np.mean(ys_val_t-ys_pred_t)
            rmse = np.sqrt(np.mean((ys_val_t-ys_pred_t)**2))
            mae = np.mean(np.abs(ys_val_t-ys_pred_t))
            pearson = np.cov((ys_val_t, ys_pred_t))[1,0]/(
                    ys_val_t.std()*ys_pred_t.std())
            
            results[0, fold, t] = me
            results[1, fold, t] = rmse
            results[2, fold, t] = mae
            results[3, fold, t] = pearson
            
    # Train and save final model
    print('Run {}/{}, final model'.format(r+1, len(runs)))
    xs_train = xs[:,features]
    ys_train = ys[:,targets]
    fit_history = model.fit(xs_train, ys_train, 
                            batch_size = batch_size,
                            epochs = epochs,
                            verbose=1,
                            )
    model.save(folder + '/run{}.h5'.format(r+1))
    
    # Record mean errors
    for t, tar in enumerate(targets):
        for e in range(4):
            results[e,nfolds,t] = np.mean(results[e,:-1,t])
        
    # Print results
    
    print('\n Run {}/{} results \n'.format(r+1, len(runs)))
    header = 'Fold \t ME_{0} \t\t RMSE_{0} \t MAE_{0} \t\t Pearson_{0}'
    for t, tar in enumerate(targets):
        print(header.format(tar))
        for fold in range(nfolds+1):
            l = '{} \t'.format(fold if fold != nfolds else 'Mean')
            for e in (0, 1, 2, 3):
                v = results[e,fold,t]
                l +='{:f} \t'.format(v)
            print(l)
            
    utils.weight_hist(model)
        
                
    # Log results
        
    param_names = params.keys()
    delimiter = ','
    
    try:
        results_file = open(results_filename, 'r+')
        results_file.read()
    except:
        results_file = open(results_filename, 'w')
        header = ''
        for p in param_names:
            header += p + delimiter
        
        for t, tar in enumerate(targets):
            for m in  ('ME', 'RMSE', 'MAE', 'Pearson'):
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