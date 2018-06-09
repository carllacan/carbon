#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
import utils 

datafolder = 'data'
resultsfolder = 'results/results_test'

runs_filename = resultsfolder + '/runs.csv'
results_filename = resultsfolder + '/results.csv'

folder = 'results/results_test'
numruns = 3

xs, ys, _ = utils.load_data(datafolder)
xs = utils.normalize_data(xs)
ys = utils.normalize_data(ys)
runs = utils.load_runs(runs_filename)

pruning = 15 # prune 10% of neurons

for r in range(numruns):
    model = load_model(folder + '/run{}.h5'.format(r+1))
    utils.weight_hist(model)
    
    features = runs[r]['features']
    targets = runs[r]['targets']
    xs_val = xs[:,features]
    ys_val = ys[:,targets]
    results = utils.evaluate_model(model, xs_val, ys_val)
    rowname = "Run {}".format(r)
    for t, tar in enumerate(targets):
        print("Target {}".format(tar))
        utils.print_results(results[t,:], rowname)
    
    ns = np.abs( utils.get_neurons(model))
    threshold = np.percentile(ns, pruning)
    layers = model.get_weights()
    p = 0.0
    for i, l in enumerate(layers):
        for j, ws in enumerate(l):
            if type(ws) is np.float32:
                if np.abs(ws) <= threshold:
                    layers[i][j] = 0
                    p += 1
            else:
                for k, w in enumerate(ws):
                    if np.abs(w) <= threshold:
                        layers[i][j][k] = 0
                        p += 1
    print("{:2.2f}% neurons prunned".format(100*p/len(ns)))
    model.set_weights(layers)
    utils.weight_hist(model)
    
    features = runs[r]['features']
    targets = runs[r]['targets']
    xs_val = xs[:,features]
    ys_val = ys[:,targets]
    results = utils.evaluate_model(model, xs_val, ys_val)
    rowname = "Run {}".format(r)
    for t, tar in enumerate(targets):
        print("Target {}".format(tar))
        utils.print_results(results[t,:], rowname)