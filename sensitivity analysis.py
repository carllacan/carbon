#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import utils 

datafolder = 'data'
resultsfolder = 'results/results1'

runs_filename = resultsfolder + '/runs.csv'
results_filename = resultsfolder + '/results.csv'

runlist = (4,16,27,36,51,63,75,87,99)
#runlist = (10,22,35,46,57,70,82,94,99)
runlist = (75,)

xs, ys, _ = utils.load_data(datafolder)
xs = utils.normalize_data(xs)
ys = utils.normalize_data(ys)
runs = utils.load_runs(runs_filename)

num_bins = 10
#pdfs = [] # probability density functions. Discrete name?

#for f in range(15):
#    pdfs.append(np.histogram(xs[:,f], bins=num_bins, density=True))
#    # "Note that the sum of the histogram values will not be equal to 1 unless 
#    # bins of unity width are chosen; it is not a probability mass function."
    
    
    
for r in runlist:
    model = load_model(resultsfolder + '/run{}.h5'.format(r+1))
    
    features = runs[r]['features']
    targets = runs[r]['targets']
    xs_val = xs[:,features]
    ys_val = ys[:,targets]
    
    print("Run {}".format(r))
    ss = []
    for i, f in enumerate(features):
        s = 0
        steps = list(range(0, xs.shape[0], int(xs.shape[0]/10)))
        xs_sorted = xs_val[xs_val[:,i].argsort()][steps]
        ys_pred = model.predict(xs_sorted)
        
        for j, x in enumerate(xs_sorted):
            if j == 0:
                continue
            dx = (xs_sorted[j, i] - xs_sorted[j-1, i])**2
            dy = np.sum((ys_pred[j] - ys_pred[j-1])**2)
            s += dy/dx
        s /= len(steps)
        ss.append(s)
        print("Sensibility of feature {}: {:2.7f}".format(f, s))
    ranking = np.array(features)[np.array(ss).argsort()]
    print("Ranking: {}".format(ranking))
    
    # Really inconsistent method, resulting ranking depends on number of steps
    # perhaps proper normalization (without the bug) solves it?
            
    # TODO: calculate sensitivity for many values of delta x, then average over them.
    # TODO: try leave-one-oout sensitivity analysis. Clamping to mean.
    # TODO: try leaveoneout s. a. with retraining of models
    # TODO: try leaving out all possible combinations
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Scatter plot analysis
#    ys_pred = model.predict(xs_val)
#    for i, f in enumerate(features):
#        plt.figure()
#        for j, t in enumerate(targets):
#            plt.plot(xs_val[:,i], ys_pred[:,j], '.')
#        plt.show()
#        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        