#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
import utils 

from sklearn.neighbors import KernelDensity

datafolder = 'data'
resultsfolder = 'results/results1_repeat'

runs_filename = resultsfolder + '/runs.csv'
results_filename = resultsfolder + '/results.csv'
sparsityfolder = resultsfolder + '/sparsityanalysis'

#runlist = (4,16,27,36,51,63,75,87,99)
#runlist = (10,22,35,46,57,70,82,94,99)
#runlist = (4, 15, 24, 36, 51, 62, 77, 84, 96)
#runlist = (9, 22, 34, 43, 55, 71, 81, 93, 96)
runlist=(4,)
xs, ys, _ = utils.load_data(datafolder)
xs, ys = utils.normalize_data(xs, ys)
runs = utils.load_runs(runs_filename)

pruning = 10 # prune 10% of neurons

norms = np.zeros((9, 3))

for m, r in enumerate(runlist):
    model = load_model(resultsfolder + '/run{}.h5'.format(r+1))
    fig = utils.weight_hist(model)
#    fig.savefig(sparsityfolder + '/run{}histogram'.format(i))
    weights = utils.get_neurons(model, False)
##    utils.ksdensity(np.array(ws).reshape(-1,1))
    
#    features = runs[r]['features']
#    targets = runs[r]['targets']
#    xs_val = xs[:,features]
#    ys_val = ys[:,targets]
    
#    results, _ = utils.evaluate_model(model, xs_val, ys_val)
#    print("\nRun {}".format(r))
#    utils.print_all_results(results, targets)
    
    # Decide pruning threshold
#    ns = np.abs( utils.get_neurons(model))
#    threshold = np.percentile(ns, pruning)
#    layers = model.get_weights()
#    p = 0.0 # pruned neuron counter
#    for i, l in enumerate(layers):
#        for j, ws in enumerate(l):
#            if type(ws) is np.float32:
#                if np.abs(ws) <= threshold:
#                    layers[i][j] = 0.0
#                    p += 1
#            else:
#                for k, w in enumerate(ws):
#                    if np.abs(w) <= threshold:
#                        layers[i][j][k] = 0.0
#                        p += 1
#    model.set_weights(layers)
#    
#    results, _ = utils.evaluate_model(model, xs_val, ys_val)
#    print("\nRun {}, {:2.2f}% neurons prunned".format(r, 100*p/len(ns)))
#    utils.print_all_results(results, targets)
    
    # TODO: traure histograma de la funció, posar-ho aci i au
    
    # TODO: calcular L0, L1, L2
    L0 = np.count_nonzero(np.abs(weights) > 0.01)
    L1 = np.linalg.norm(weights, 1)
    L2 = np.linalg.norm(weights, 2)
    norms[m] = [L0, L1, L2]
    
    # TODO: save prunning results. TODO: export_model function
    
    # TODO: performance vs prunning level plot for each model?
    
    # TODO: what's more important, the neurons or the weights? 
    # I could try calculating the average or rms weight of the incoming 
    # connections to each neuron to find out how important it is, and then
    # redefine the model without those neurons that are not important.
    
    # TODO: prun until 16-feature models are comparatively big to 8-feature
    # models, and compare performances.