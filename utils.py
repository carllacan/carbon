#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import Callback

#class PlotCallback(Callback):
#    # Plots accuracy and loss after each epoch
#    def on_train_begin(self, logs={}):
#        self.ls_history = []
#        self.as_history = []
#
#    def on_epoch_end(self, epoch, logs={}):
#        loss = logs.get('loss')
#        acc = logs.get('acc')
#        self.ls_history.append(loss)
#        self.as_history.append(acc)
#        ls = self.ls_history
#        acs = self.as_history
#        es = np.arange(1, epoch+2)
#        plt.figure()
#        plt.plot(es, ls)
#        plt.plot(es, acs)
#        plt.show()

def load_data(datafolder):
    delimiter=','
    xs = np.genfromtxt(datafolder + '/dataX.csv', delimiter=delimiter)
    ys = np.genfromtxt(datafolder + '/dataY.csv', delimiter=delimiter)
    vs =  np.genfromtxt(datafolder + '/inds_crossval.csv') - 1
    return xs, ys, vs

def normalize_data(ds):
    for i in range(6):
        ds[:,i] -= np.mean(ds[:,i])
        ds[:,i] /= np.std(ds[:,i])
    return ds

def load_runs(filename):
    delimiter = ','
    int_params = ('batch_size', 'epochs')
    float_params = ('dropout', 'reg_v')
    list_params = ('features','targets','hidden_layers')
    f = open(filename, 'r')
    lines = f.readlines()
    colnames = lines[0].strip('\n').split(delimiter)
    runs = []
    for l in lines[1:]:
        runs.append({})
        values = l.strip('\n').split(delimiter)
        for param, value in zip(colnames, values):
            if param in int_params:
                runs[-1][param] = int(value)
            elif param in float_params:
                runs[-1][param] = float(value)
            elif param in list_params:
                runs[-1][param] = []
                for e in value.split('-'):
                    if e != '':
                        runs[-1][param].append(int(e))
            else:
                runs[-1][param] = value

    # TODO: can I one-line this for?
    return runs

def get_neurons(model):
#    return np.hstack([l.flatten() for l in model.get_weights()])
    weights = []
    layers = model.get_weights()
    for i, l in enumerate(layers):
        for j, ws in enumerate(l):
            if type(ws) is np.float32:
                weights.append(layers[i][j])
            else:
                for k, w in enumerate(ws):
                    weights.append(w)
    return weights
                    
                    

def weight_hist(model):
    ws = get_neurons(model)
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

# TODO: make this function print by target too
def print_results(results, rowname):
    header = '\tME \t\tRMSE \t\tMAE \t\tPearson'
    print(header)
    l = '{} \t'.format(rowname)
    for e in results:
        l +='{:f} \t'.format(e)
    print(l)