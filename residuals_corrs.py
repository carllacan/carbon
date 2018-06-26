#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

xs, ys, _ = utils.load_data(datafolder)
xs, ys = utils.normalize_data(xs, ys)
#runs = utils.load_runs(runs_filename)

norm_ord = 1

# Correlations matrix

corrs = np.zeros((3,3))
lorentz = lambda p0, p1, p2,x: (1/(1+(x/p0 - 1)**4*p1**2))*p2
gaussian = lambda mu, sigma, xs: (1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (xs - mu)**2 / (2 * sigma**2) ))

# Single-output analysis
ys_pred = np.zeros(ys.shape)
rs = np.zeros(ys.shape)
fig, axs = plt.subplots(2, 3)
axs = np.array(axs).flatten()
for i, r in enumerate(runlist[0:6]):
    model = load_model(resultsfolder + '/run{}.h5'.format(r+1))
    
    features = runs[r]['features']
    targets = runs[r]['targets']
    xs_val = xs[:,features]
    ys_val = ys[:,targets]
    results, residues, prediction = utils.evaluate_model(model, xs_val, ys_val)
    
    ys_pred[:,i:i+1] = prediction
    rs[:,i:i+1] = residues
    
    print("\nRun {}".format(r))
    utils.print_all_results(results, targets)
    
    # Residual histogram
    targetname = "Target whatever"
#    bins = axs[i].hist(residues, bins=100, normed=True)[1]
#    mu = residues.mean()
#    sigma = residues.std()
#    axs[i].plot(bins, gaussian(mu, sigma, bins),
#             linewidth=2, color='r')
#    # TODO: fit a lorentzian
##    p = 0.1, 0.2, 0.3
##    axs[i].plot(bins, lorentz(p[0], p[1], p[2], bins),
##             linewidth=2, color='g')
#    
#    axs[i].set_title(targetname)
    utils.histfit(axs[i], residues)
    
plt.savefig(residualfolder + '/singleoutput'.format(i))
plt.show()
    
    
# Correlation of residuals and features/targets

corr_xs = np.corrcoef(xs, rs, rowvar=False)
corr_res = np.corrcoef(rs, rowvar=False)
corr_ys = np.corrcoef(ys, ys_pred, rowvar=False)

corrs[0,0] = np.linalg.norm(corr_xs, ord=norm_ord)
corrs[0,1] = np.linalg.norm(corr_res, ord=norm_ord)
corrs[0,2] = np.linalg.norm(corr_ys, ord=norm_ord)

##############################################

## Multiple-output analysis
#ys_pred = np.zeros(ys.shape)
#rs = np.zeros(ys.shape)
#fig, axs = plt.subplots(2, 3)
#axs = np.array(axs).flatten()
#for i, r in enumerate(runlist[6:8]):
#    model = load_model(resultsfolder + '/run{}.h5'.format(r+1))
#    
#    features = runs[r]['features']
#    targets = runs[r]['targets']
#    xs_val = xs[:,features]
#    ys_val = ys[:,targets]
#    results, residues, prediction = utils.evaluate_model(model, xs_val, ys_val)
#    
#    ys_pred[:,i*3:i*3+3] = prediction
#    rs[:,i*3:i*3+3] = residues
#    
#    print("\nRun {}".format(r))
#    utils.print_all_results(results, targets)
#    
#    # Residual histogram
#    targetname = "Target whatever"
#    bins = axs[i].hist(residues, bins=100, normed=True)[1]
#    mu = residues.mean()
#    sigma = residues.std()
#    axs[i].plot(bins, gaussian(mu, sigma, bins),
#             linewidth=2, color='r')
#    # TODO: fit a lorentzian
##    p = 0.1, 0.2, 0.3
##    axs[i].plot(bins, lorentz(p[0], p[1], p[2], bins),
##             linewidth=2, color='g')
#    
#    axs[i].set_title(targetname)
#    
#plt.savefig(residualfolder + '/singleoutput'.format(i))
#plt.show()
#
#    
## Correlation of residuals and features/targets
#
#corr_xs = np.corrcoef(xs, rs, rowvar=False)
#corr_res = np.corrcoef(rs, rowvar=False)
#corr_ys = np.corrcoef(ys, ys_pred, rowvar=False)
#
#corrs[0,0] = np.linalg.norm(corr_xs, ord=norm_ord)
#corrs[0,1] = np.linalg.norm(corr_res, ord=norm_ord)
#corrs[0,2] = np.linalg.norm(corr_ys, ord=norm_ord)
#
#
#print(corrs)
