#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
import utils 

folder = 'results/results_test'
numruns = 3

models = []
for r in range(numruns):
    model = load_model(folder + '/run{}.h5'.format(r+1))
    models.append(model)
    utils.weight_hist(model)
#threshold = 0.02
pruning = 10 # % prune 10% of neurons

for m in models:
    ns = np.abs( utils.get_neurons(m))
#    threshold = ns[int(np.floor(len(ns)*pruning))]
    threshold = np.percentile(ns, pruning)
    layers = m.get_weights()
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
    m.set_weights(layers)
    utils.weight_hist(m)
# TODO: evaluate before and after