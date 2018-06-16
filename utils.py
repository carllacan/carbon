#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

def load_data(datafolder):
    delimiter=','
    xs = np.genfromtxt(datafolder + '/dataX.csv', delimiter=delimiter)
    ys = np.genfromtxt(datafolder + '/dataY.csv', delimiter=delimiter)
    vs =  np.genfromtxt(datafolder + '/inds_crossval.csv') - 1
    return xs, ys, vs

def normalize_data(xs, ys):
    for i in range(xs.shape[1]):
        xs[:,i] -= np.mean(xs[:,i])
        xs[:,i] /= np.std(xs[:,i])
    for i in range(ys.shape[1]):
        ys[:,i] -= np.mean(ys[:,i])
#        ys[:,i] /= np.std(ys[:,i])
    return xs, ys


#def normalize_data(xs): # has a bug, will delete when re-training
#    for i in range(6):
#        xs[:,i] -= np.mean(xs[:,i])
#        xs[:,i] /= np.std(xs[:,i])
#    return xs

def load_runs(filename):
    delimiter = '\t'
    f = open(filename, 'r')
    lines = f.readlines()
    runs = []
    for l in lines[1:]:
        values = l.strip('\n').replace(',','.').split(delimiter)
        params = {}
        params['features'] = [int(i) for i in values[0].split('-')]
        params['targets'] = [int(i) for i in values[1].split('-')]
        params['hidden_layers'] = [int(i) for i in values[2].split('-')]
        params['dropout'] = float(values[3])
        params['reg_type'] = values[4]
        params['reg_v'] = float(values[5])
        params['batch_size'] = int(values[6])
        params['epochs'] = int(values[7])
        runs.append(params)

    return runs

def get_neurons(model, remove_zeros = True):
    weights = []
    layers = model.get_weights()
    for i, l in enumerate(layers):
        for j, ws in enumerate(l):
            if type(ws) is np.float32:
                w = layers[i][j]
                if w != 0.0 or (w == 0.0 and not remove_zeros):
                    weights.append(w)
            else:
                for k, w in enumerate(ws):
                    w = layers[i][j][k]
                    if w != 0.0 or (w == 0.0 and not remove_zeros):
                        weights.append(w)
    return weights
                    
def weight_hist(model, remove_zeros = True):
    ws = get_neurons(model, remove_zeros)
    plt.figure()
    plt.hist(ws, bins=100)
    plt.show()

def evaluate_model(model, xs_val, ys_val):
    num_targets = model.output_shape[1]
    ys_pred = model.predict(xs_val)
    results = np.zeros((num_targets, 4))
    for t in range(num_targets):            
        ys_val_t = ys_val[:,t]
        ys_pred_t = ys_pred[:,t]
        me = np.mean(ys_val_t-ys_pred_t)
        rmse = np.sqrt(np.mean((ys_val_t-ys_pred_t)**2))
        mae = np.mean(np.abs(ys_val_t-ys_pred_t))
        pearson = np.cov((ys_val_t, ys_pred_t))[1,0]/(
                          ys_val_t.std()*ys_pred_t.std())
        results[t, 0] = me
        results[t, 1] = rmse
        results[t, 2] = mae
        results[t, 3] = pearson
    return results

def print_results(results, rowname):
    header = '\tME \t\tRMSE \t\tMAE \t\tPearson'
    print(header)
    l = '{} \t'.format(rowname)
    for e in results:
        l +='{:f} \t'.format(e)
    print(l)
    
def print_all_results(results, targets):
    header = '\tME \t\tRMSE \t\tMAE \t\tPearson'
    print(header)
    
    for t, tar in enumerate(targets):
        l = 'T{} \t'.format(tar)
        for e in (0, 1, 2, 3):
            l +='{:f}, \t'.format(results[t,e])
        print(l)
