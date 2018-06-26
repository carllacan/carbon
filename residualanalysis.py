#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import utils 

datafolder = 'data'
resultsfolder = 'results/results1_repeat'

runs_filename = resultsfolder + '/runs.csv'
results_filename = resultsfolder + '/results.csv'
residualfolder = resultsfolder + '/residuals'

#runlist = (4,16,27,36,51,63,75,87,99)
#runlist = (10,22,35,46,57,70,82,94,99)
#runlist = (4, 15, 24, 36, 51, 62, 77, 84, 96)
runlist = (9, 22, 34, 43, 55, 71, 81, 93, 96)
#runlist = (4,)

xs, ys, _ = utils.load_data(datafolder)
xs, ys = utils.normalize_data(xs, ys)
runs = utils.load_runs(runs_filename)

for i, r in enumerate(runlist):
    model = load_model(resultsfolder + '/run{}.h5'.format(r+1))
    
    features = runs[r]['features']
    targets = runs[r]['targets']
    xs_val = xs[:,features]
    ys_val = ys[:,targets]
    
    results, rs, p = utils.evaluate_model(model, xs_val, ys_val)
    print("\nRun {}".format(r))
    utils.print_all_results(results, targets)
    
    # Residual histogram
    plt.figure()
    plt.hist(rs, bins=100)
    plt.savefig(residualfolder + '/run{}residuals'.format(i))
    plt.show()
    
# TODO: matriu correlació de residus entre ells
# TODO: m d c entre residus i ys
# TODO: matri de correlació de residus i x
# traure normes de cada matriu
# exportar normes a latex
# comparar en latex

    
     # Correlation of residuals and features/targets
     
    corr_res = np.corrcoef(rs, rs)
     
     
     
     
     
     
     
     
     
#    
#    if len(features) == 8:
#        nrow, ncol = 2, 4
#    if len(features) == 15:
#        nrow, ncol = 5, 3
#    fig, axs = plt.subplots(nrow, ncol)
#    axs = np.array(axs).flatten()
#    for f, fea in enumerate(features):
#        axs[f].plot(rs, xs_val[:,f], '.', label=fea)
#        axs[f].set_title('Feature {}'.format(fea))
#        print('Feature {}: \t{:2.3f}'.format(
#                fea, np.correlate(rs, xs_val[:,f], mode="valid")[0]))
#    plt.savefig(residualfolder + '/run{}scatter'.format(i))
#    plt.show()
#    
#    if len(targets) == 1:
#        nrow, ncol = 1, 1
#    if len(targets) == 3:
#        nrow, ncol = 1, 3
#    if len(targets) == 6:
#        nrow, ncol = 2, 3
#    fig, axs = plt.subplots(nrow, ncol)
#    axs = np.array(axs).flatten()
#    for t, tar in enumerate(targets):
#        axs[t].plot(rs, ys_val[:,t], '.', label=tar)
#        axs[t].set_title('Target {}'.format(tar))
#        print('Target {}: \t{:2.3f}'.format(
#                tar, np.correlate(rs, ys_val[:,t], mode="valid")[0]))
#    plt.show()
#    