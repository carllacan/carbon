#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import load_model
import utils 

folder = 'results/results_test'
numruns = 3
models = []

for r in range(numruns):
    model = load_model(folder + '/run{}.h5'.format(r+1))
    models.append(model)
    utils.weight_hist(model)

