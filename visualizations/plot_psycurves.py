"""
This script can be used to plot the model psychometric curves alongside
the human data.
Since we are running the models on different noise instances than they
have been trained on, this takes a while.

@author: Lynn Schmittwilken, June 2024
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.insert(1, '../')
import psignifit as ps

sys.path.insert(1, '../simulations')
from functions import create_filter_outputs, compute_performance, create_edge, \
    create_noises, plotPsych, load_all_data, reformat_data, load_params

sys.path.insert(1, '../experiment')
from params import stim_params as sparams

np.random.seed(0)

# Get model psychometric functions for the following contrasts:
n_levels = 30
xlim = [1e-05, 0.05]
contrasts = np.linspace(xlim[0], xlim[1], n_levels)

nInstances = 30  # number of noise instances used for testing

edge_conds = sparams["edge_widths"]
noise_conds = sparams["noise_types"]


def get_performance(params, mparams, e, noiseList, c):
    beta, eta, kappa = params["beta"], params["eta"], params["kappa"]
    alphas = [value for key, value in params.items() if "alpha" in key.lower()]
    
    edge = create_edge(c, e, sparams)
    
    pc = np.zeros(mparams["n_trials"])
    for t in range(mparams["n_trials"]):
        mout1, mout2 = create_filter_outputs(edge, noiseList[t], noiseList[t-1], mparams)
        pc[t] = compute_performance(mout1, mout2, mparams, alphas, beta, eta, kappa)
    return pc.mean()


def get_pcs(params, mparams, e, n):
    # Append performances
    pcs = []
    for c in contrasts:
        pcs.append(get_performance(params, mparams, e, n, c))
    return pcs


def plotHuman(axes):
    vps = ["ls", "mm", "jv", "ga", "sg", "fd"]
    datadir = "../experiment/results/"
    data = load_all_data(datadir, vps)

    # Set parameters for psignifit
    options = {"sigmoidName": "norm", "expType": "2AFC"}
    colors = ["C2", "C1", "C0"]
    
    for ni, n in enumerate(noise_conds):
        for ei, e in enumerate(edge_conds[::-1]):
            x, n_correct, n_trials = reformat_data(data, n, e)
            res = ps.psignifit(np.array([x, n_correct, n_trials]).transpose(), optionsIn=options)
            plotPsych(res, color=colors[ei], axisHandle=axes[ni, ei], plotAsymptote=True, plotCI=True, xlim=xlim)


def plotModel(results_file, axes, ltype, color, noiseDict):
    best_params, mparams = load_params(results_file)
    mparams["n_trials"] = nInstances
    for ni, n in enumerate(noise_conds):
        for ei, e in enumerate(edge_conds[::-1]):
            pcs = get_pcs(best_params, mparams, e, noiseDict[n])
            axes[ni, ei].plot(contrasts, pcs, ltype, color=color, linewidth=1)


fig, axes = plt.subplots(len(noise_conds), len(edge_conds), figsize=(6, 8), sharey=True, sharex=True)
fig.subplots_adjust(wspace=0.001, hspace=0.001)

# Plot human data
plotHuman(axes)

# Plot pooled curves
print()
noiseDict = create_noises(sparams, nInstances)
plotModel("../simulations/results_multi_5.pickle", axes, "--", "darkgray", noiseDict)
plotModel("../simulations/results_multi.pickle", axes, "-.", "k", noiseDict)

axes[3, 0].set(ylabel="Percent correct")
axes[len(noise_conds)-1, 1].set(xlabel="Edge contrast [rms]")
plt.savefig('psychocurves_multi-multi5.png', dpi=300)
plt.show()
