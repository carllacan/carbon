#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

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

def get_neurons(model):
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
                    
def ksdensity(values):

    plt.show()

def weight_hist(model):
    ws = np.array(get_neurons(model))
    fig = plt.figure()
    bins = plt.hist(ws, bins=100, normed=True)[1]
#    values = np.array(ws).reshape(-1,1) #TODO: make this prettier
    mu = ws.mean()
    sigma = ws.std()
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
             linewidth=2, color='r')
#    Vecpoints=np.linspace(values.min(),values.max(),100)[:,None]
#    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(values)
#    logkde = kde.score_samples(bins)
#    plt.plot(bins,np.exp(logkde), linewidth=2, color='g')
    plt.show()
    
    return fig

def histfit(ax, values, title=""):
    lorentz = lambda p0, p1, p2,x: (1/(1+(x/p0 - 1)**4*p1**2))*p2
    gaussian = lambda mu, sigma, xs: (1/(sigma * np.sqrt(2 * np.pi)) *
                 np.exp( - (xs - mu)**2 / (2 * sigma**2) ))

    bins = ax.hist(values, bins=100, normed=True)[1]
    mu = values.mean()
    sigma = values.std()
    ax.plot(bins, gaussian(mu, sigma, bins),
             linewidth=2, color='r')
    # TODO: fit a lorentzian
#    p = 0.1, 0.2, 0.3
#    axs[i].plot(bins, lorentz(p[0], p[1], p[2], bins),
#             linewidth=2, color='g')
    
    ax.set_title(title)
    
def evaluate_model(model, xs_val, ys_val):
    num_targets = model.output_shape[1]
    ys_pred = model.predict(xs_val)
    results = np.zeros((num_targets, 4))
    rs = np.zeros(ys_val.shape)
    for t in range(num_targets):            
        ys_val_t = ys_val[:,t]
        ys_pred_t = ys_pred[:,t]
        residuals = ys_val_t-ys_pred_t
        me = np.mean(residuals)
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        pearson = np.cov((ys_val_t, ys_pred_t))[1,0]/(
                          ys_val_t.std()*ys_pred_t.std())
        # TODO: consider np.corrcoef
        results[t, 0] = me
        results[t, 1] = rmse
        results[t, 2] = mae
        results[t, 3] = pearson
        rs[:,t:t+1] = residuals.reshape((9724,1))
    return results, rs, ys_pred

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
