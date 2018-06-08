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

def weight_hist(model):
    ws = np.hstack([l.flatten() for l in model.get_weights()])
    plt.figure()
    plt.hist(ws, bins=100)

runs = load_runs('testruns.csv')