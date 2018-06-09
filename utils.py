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
    
def evaluate_model(ys_val, ys_pred):
    
    me = np.mean(ys_val-ys_pred)
    rmse = np.sqrt(np.mean((ys_val-ys_pred)**2))
    mae = np.mean(np.abs(ys_val-ys_pred))
    pearson = np.cov((ys_val, ys_pred))[1,0]/(
                      ys_val.std()*ys_pred.std())
    
    return me, rmse, mae, pearson

def print_results(results, rowname):
#    header = '\tME_{0} \t\tRMSE_{0} \t\tMAE_{0} \t\tPearson_{0}'
    header = '\tME \t\tRMSE \t\tMAE \t\tPearson'
    print(header)
    l = '{} \t'.format(rowname)
    for e in results:
#            v = results[e,fold,t]
        l +='{:f} \t'.format(e)
    print(l)