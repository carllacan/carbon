#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#import random
import utils
import time

np.random.seed(1729)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2, l1

# Load data
    
datafolder = 'data'
xs = np.genfromtxt(datafolder + '/dataX.csv', delimiter=',')
ys = np.genfromtxt(datafolder + '/dataY.csv', delimiter=',')
vs =  np.genfromtxt(datafolder + '/inds_crossval.csv') -1


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

nfolds = 1#np.unique(vs).size
    
# Normalization of training data
    
for i in range(16):
    xs[:,i] -= np.mean(xs[:,i])
    xs[:,i] /= np.std(xs[:,i])
    
    
for i in range(6):
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
        # Model creation
        
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
            me, rmse, mae, p = utils.evaluate_model(ys_val[:,t], ys_pred[:,t])
            results[0, fold, t] = me
            results[1, fold, t] = rmse
            results[2, fold, t] = mae
            results[3, fold, t] = p
            
    # Train and save final model
    print('Run {}/{}, final model training'.format(r+1, len(runs)))
    xs_train = xs[:,features]
    ys_train = ys[:,targets]
    t0 = time.time()
    fit_history = model.fit(xs_train, ys_train, 
                            batch_size = batch_size,
                            epochs = epochs,
                            verbose=1,
                            )
    train_dt = time.time() - t0
    t0 = time.time()
    model.predict(xs_train)
    test_dt = time.time() - t0
    
    model.save(folder + '/run{}.h5'.format(r+1))
    
    # Record mean errors
    
    for t, tar in enumerate(targets):
        for e in range(4):
            results[e,nfolds,t] = np.mean(results[e,:-1,t])
        
    # Print results
    
    print('\n Run {}/{} results \n'.format(r+1, len(runs)))
    for t, tar in enumerate(targets):
        print("Target {}".format(tar))
        for fold in range(nfolds+1):
            rowname = 'Fold {}'.format(fold) if fold != nfolds else 'Mean'
            utils.print_results(results[:,fold,t], rowname)
      
    # Log results
        
    delimiter = ','
    try:
        results_file = open(results_filename, 'r+')
        results_file.read()
    except:
        results_file = open(results_filename, 'w')
        colnames = ('Run', 'Target', 'ME', 'RMSE', 'MAE', 
                    'Pearson', 'Train time', 'Test time')
        header = delimiter.join(colnames)
        results_file.write(header + '\n')
        
    temp = '{},{},{r[0]:.5f},{r[1]:.5f},{r[2]:.5f},{r[3]:.5f},{:.4f},{:.4f}'
    for t, tar in enumerate(targets):
        row = temp.format(r, tar, train_dt, test_dt, r=results[:,-1,t])
        
        results_file.write(row + '\n')
    
    results_file.close()