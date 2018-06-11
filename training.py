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
from keras import backend as K

datafolder = 'data'
resultsfolder = 'results/results1'
start_at = 52

# Load and normalize data
xs, ys, vs = utils.load_data(datafolder)
xs = utils.normalize_data(xs)
ys = utils.normalize_data(ys)

# Set filenames
runs_filename = resultsfolder + '/runs.csv'
results_filename = resultsfolder + '/results.csv'

# Load the parameters of the runs
runs = utils.load_runs(runs_filename)

nfolds = np.unique(vs).size
    
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
    
    results = np.zeros((nfolds+1, len(targets), 4))
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

        print('Run {}/{}, split {}/{}'.format(r+1, len(runs), fold+1, nfolds))
        model.fit(xs_train, ys_train, 
                  batch_size = batch_size,
                  epochs = epochs,
                  verbose=1,
                  )
        
        # Validation
        results[fold] = utils.evaluate_model(model, xs_val, ys_val)

        del(model)
        K.clear_session()
    
            
    # Train and save final model
    print('Run {}/{}, final model training'.format(r+1, len(runs)))

    # TODO: make a create_model function, or make it so that this code doesn't get repeated

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

    xs_train = xs[:,features]
    ys_train = ys[:,targets]
    t0 = time.time()
    model.fit(xs_train, ys_train, 
              batch_size = batch_size,
              epochs = epochs,
              verbose=1,
              )
    train_dt = time.time() - t0
    t0 = time.time()
    model.predict(xs_train)
    test_dt = time.time() - t0
    
    model.save(resultsfolder + '/run{}.h5'.format(r+1))
    
    del(model)
    K.clear_session()

    # Record mean errors
    
    for t, tar in enumerate(targets):
        for e in range(4):
            results[nfolds,t,e] = np.mean(results[:-1,t,e])
        
    # Print results
    
    print('\n Run {}/{} results \n'.format(r+1, len(runs)))
    utils.print_all_results(results[fold], targets)

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
        row = temp.format(r, tar, train_dt, test_dt, r=results[-1,t,:])
        
        results_file.write(row + '\n')
    
    results_file.close()

