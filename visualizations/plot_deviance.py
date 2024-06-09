#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:55:57 2024

@author: lynnschmittwilken
"""

import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from functions import run_model, naka_rushton, decoder_dprime, \
    create_edge, load_all_data, crop_outputs, calc_deviance_residual
from stimupy.noises.whites import white as create_whitenoise
from stimupy.noises.narrowbands import narrowband as create_narrownoise
from stimupy.noises.naturals import one_over_f as create_pinknoise

sys.path.append('../experiment')
from params import stim_params as sparams

np.random.seed(0)

edge_conds = sparams["edge_widths"]
noise_conds = sparams["noise_types"]

n_edges = len(edge_conds)
n_noises = len(noise_conds)
nTrials = 30


def reformat_data(data, n, e, vp='all', session=None):
    if session != None and vp != "all":
        data = data[data["session"] == session]

    # Get data from one condition
    data_cond = data[(data["noise"] == n) & (data["edge_width"] == e)]

    # Get data from all or selected observer
    if vp == 'all':
        data_vp = data_cond
    else:
        data_vp = data_cond[data_cond["vp"] == vp]
    
    for s in range(2):
        contrasts = np.unique(data_vp["edge_contrast"])
        n_correct = np.zeros(len(contrasts))
        n_trials = np.zeros(len(contrasts))
        for i in range(len(contrasts)):
            data_correct = data_vp[data_vp["edge_contrast"] == contrasts[i]]["correct"]
            n_correct[i] = np.sum(data_correct)
            n_trials[i] = len(data_correct)
    return np.array([contrasts, n_correct, n_trials]).transpose()


def load_params(results_file):
    # Load data from pickle:
    with open(results_file, 'rb') as handle:
        data_pickle = pickle.load(handle)
    
    best_params = data_pickle["best_params_auto"]
    best_loss = data_pickle["best_loss_auto"]
    model_params = data_pickle["model_params"]
    print("Best loaded loss:", best_loss)

    # Optionally update model params
    model_params["n_trials"] = nTrials
    return best_params, model_params


def create_noise(n, sparams):
    sp = sparams
    ssize = sp["stim_size"]
    ppd = sp["ppd"]
    rms = sp["noise_contrast"] * sp["mean_lum"]
    
    if n == "none":
        noise = np.zeros([int(ssize*ppd), int(ssize*ppd)])
    elif n == "white":
        noise = create_whitenoise(visual_size=ssize, ppd=ppd, pseudo_noise=True)["img"]
    elif n == "pink1":
        noise = create_pinknoise(visual_size=ssize, ppd=ppd, exponent=1., pseudo_noise=True)["img"]
    elif n == "pink2":
        noise = create_pinknoise(visual_size=ssize, ppd=ppd, exponent=2., pseudo_noise=True)["img"]
    elif n == "narrow0.5":
        noise = create_narrownoise(visual_size=ssize, ppd=ppd, center_frequency=0.5, bandwidth=1., pseudo_noise=True)["img"]
    elif n == "narrow3":
        noise = create_narrownoise(visual_size=ssize, ppd=ppd, center_frequency=3, bandwidth=1., pseudo_noise=True)["img"]
    elif n == "narrow9":
        noise = create_narrownoise(visual_size=ssize, ppd=ppd, center_frequency=9, bandwidth=1., pseudo_noise=True)["img"]

    if not n=="none":
        noise = noise - noise.mean()
        noise = noise / noise.std() * rms
    return noise


def create_noises(sparams, nTrials):
    noiseDict = {}
    for ni, n in enumerate(noise_conds):
        noiseList = []
        for t in range(nTrials):
            noiseList.append(create_noise(n, sparams))
        noiseDict[n] = noiseList
    return noiseDict


def get_performance(params, mparams, e, noiseList, c, weighted):
    try:
        gain = mparams["gain"]
    except:
        gain = None

    if type(params) is dict:
        beta, eta, kappa = params["beta"], params["eta"], params["kappa"]
        alphas = [value for key, value in params.items() if "alpha" in key.lower()]
    
    edge = create_edge(c, e, sparams)
    sweight = np.abs(edge - edge.mean())
    
    pc = np.zeros(mparams["n_trials"])
    for t in range(mparams["n_trials"]):
        # Run models
        mout1 = run_model(edge+noiseList[t], mparams)
        if mparams["sameNoise"]:
            mout2 = run_model(noiseList[t]+sparams["mean_lum"], mparams)
        else:
            mout2 = run_model(noiseList[t-1]+sparams["mean_lum"], mparams)
        
        # Weight by stimulus profile
        if weighted:
            mout1 = mout1 * np.expand_dims(sweight/sweight.max(), -1)
            mout2 = mout2 * np.expand_dims(sweight/sweight.max(), -1)
        
        # For different instances, we need to crop model outputs
        mout1 = crop_outputs(mout1, mparams["cfac"])
        mout2 = crop_outputs(mout2, mparams["cfac"])

        # Naka-rushton
        mout1 = naka_rushton(mout1, alphas, beta, eta, kappa, gain)
        mout2 = naka_rushton(mout2, alphas, beta, eta, kappa, gain)
        
        # Get performance
        pc[t] = decoder_dprime(mout1, mout2)
    return pc.mean()


def getHuman():
    vps = ["ls", "mm", "jv", "ga", "sg", "fd"]
    datadir = "../experiment/results/"
    data = load_all_data(datadir, vps)

    # Get pooled / individual experimental data
    dfListPool = []; dfListInd = []
    for ni, n in enumerate(noise_conds):
        for ei, e in enumerate(edge_conds):
            refData = reformat_data(data, n, e, "all")
            dfListPool.append(
                pd.DataFrame({
                    "noise": [n,]*len(refData[:,0]),
                    "edge": [e,]*len(refData[:,0]),
                    "contrasts": refData[:,0],
                    "ncorrect": refData[:,1],
                    "ntrials": refData[:,2],
                    })
                )
            for v, vp in enumerate(vps):
                refData = reformat_data(data, n, e, v)
                dfListInd.append(
                    pd.DataFrame({
                        "noise": [n,]*len(refData[:,0]),
                        "edge": [e,]*len(refData[:,0]),
                        "contrasts": refData[:,0],
                        "ncorrect": refData[:,1],
                        "ntrials": refData[:,2],
                        "vp": [v,]*len(refData[:,0])
                        })
                    )
    dfPool = pd.concat(dfListPool).reset_index(drop=True)
    dfInd = pd.concat(dfListInd).reset_index(drop=True)
    return dfPool, dfInd


def bin_residuals(residuals, bound, bins):
    # Stack residuals
    residuals_stack = np.copy(residuals)
    residuals_stack[residuals_stack < -bound] = -bound
    residuals_stack[residuals_stack > bound] = bound
    
    # Binning
    c, b = np.histogram(residuals_stack, bins)
    
    # Vectors have different length, so repeat first element for our plotting routine
    c = np.append(c[0], c)
    
    # Normalize area under the curve to 1
    c = c / np.trapz(c, dx=b[1]-b[0])
    return c, b


def getModel(results_file, dfPool, dfInd, noiseDict, weighted=False, plotting=False):
    best_params, mparams = load_params(results_file)
    vps = np.unique(dfInd["vp"])
    
    # Initialize arrays for deviances per stimulus condition
    devModel = np.zeros([n_noises, n_edges])
    devHuman = np.zeros([n_noises, n_edges])
    
    for ni, n in enumerate(noise_conds):
        for ei, e in enumerate(edge_conds[::-1]):
            # Initialize list for deviance residuals per stimulus condition
            devResModel = []; devResHuman = []
    
            # Get performance of average observer (all contrasts)
            dfTemp = dfPool[(dfPool["noise"]==n) & (dfPool["edge"]==e)]
            pc_human = dfTemp["ncorrect"].to_numpy() / dfTemp["ntrials"].to_numpy()
    
            # Loop through all contrasts
            for i, c in enumerate(dfTemp["contrasts"].to_numpy()):
                pc_model = get_performance(best_params, mparams, e, noiseDict[n], c, weighted)
    
                # Calculate deviance residuals for each observer and append
                for v, vp in enumerate(vps):
                    dfVP = dfInd[(dfInd["noise"]==n) & (dfInd["edge"]==e) & (dfInd["vp"]==v)]
                    nCorrect = dfVP["ncorrect"].to_numpy()
                    nTrials = dfVP["ntrials"].to_numpy()
    
                    devResModel.append(calc_deviance_residual(y=nCorrect[i], n=nTrials[i], p=pc_model))
                    devResHuman.append(calc_deviance_residual(y=nCorrect[i], n=nTrials[i], p=pc_human[i]))
    
            # Calculate deviance per condition (=sum of squared deviance residuals)
            devModel[ni, ei] = (np.array(devResModel)**2.).sum()
            devHuman[ni, ei] = (np.array(devResHuman)**2.).sum()
            
            # Plot deviance residuals
            if plotting:
                # Stack residuals larger than 5 / smaller than -5 + Binning + Normalizung
                c1, b1 = bin_residuals(devResModel, 5, 6)
                c_emp, b_emp = bin_residuals(devResHuman, 5, 6)
        
                # Get x and y for normal distribution + Normalize
                xnorm = np.linspace(-5, 5, 1000)
                ynorm = norm.pdf(xnorm, 0, 1)
                ynorm = ynorm / np.trapz(ynorm, dx=xnorm[1]-xnorm[0])
        
                # Calculate mean residuals (=bias)
                biasModel = np.array(devResModel).mean()
        
                # Plotting: Single-scale model + empirical
                axes[ni, ei].plot(b2, c2, "gray", drawstyle="steps", label="single")
                axes[ni, ei].fill_between(b2, c2, step="pre", alpha=0.8, color="darkgray")
                axes[ni, ei].plot(xnorm, ynorm, "gray")
                axes[ni, ei].plot((biasSingle, biasSingle), (0, 0.9), "gray", linestyle="dashed")
                axes[ni, ei].plot(b_emp, c_emp, colors[ei], drawstyle="steps", lw=2)
                
                # Plotting: Multi-scale model + empirical
                axes[ni, ei].plot(b1, -c1, "k", drawstyle="steps", label="multi")
                axes[ni, ei].fill_between(b1, -c1, step="pre", alpha=0.5, color="k")
                axes[ni, ei].plot(xnorm, -ynorm, "k")
                axes[ni, ei].plot((biasMulti, biasMulti), (0, -0.9), "k", linestyle="dashed")
                axes[ni, ei].plot(b_emp, -c_emp, colors[ei], drawstyle="steps", label="empirical", lw=2)
    return devModel, devHuman



# Plot human data
dfPool, dfInd = getHuman()

# Plot pooled curves
print()
noiseDict = create_noises(sparams, nTrials)
dev1, devHuman = getModel("./multi_dprime_30_crop.pickle", dfPool, dfInd, noiseDict, weighted=False, plotting=True)
dev2, devHuman = getModel("./multi_dprime_50_dif-weighted.pickle", dfPool, dfInd, noiseDict, weighted=True, plotting=True)
# dev3, devHuman = getModel("./multi_dprime_30_dif_crop_local.pickle", dfPool, dfInd, noiseDict)

plt.figure(figsize=(12, 4))
m = 2; nc = np.arange(0, n_noises); ec = np.arange(0, n_edges); ecl = ["LSF", "MSF", "HSF"]
# plt.subplot(1, n, 1); plt.imshow(devHuman,      "gray", vmin=1); plt.colorbar(); plt.title("Human")
plt.subplot(1, m, 1); plt.imshow(dev1/devHuman, "gray", vmin=1, vmax=4.5); plt.title("Old approach - main"), plt.yticks(nc, noise_conds); plt.xticks(ec, ecl); plt.colorbar()
plt.subplot(1, m, 2); plt.imshow(dev2/devHuman, "gray", vmin=1, vmax=4.5); plt.title("New approach - main"), plt.yticks(nc, noise_conds); plt.xticks(ec, ecl); plt.colorbar()
# plt.subplot(1, m, 3); plt.imshow(dev3/devHuman, "gray", vmin=1, vmax=6); plt.title("New approach - best"), plt.yticks(nc, noise_conds); plt.xticks(ec, ecl); plt.colorbar()

plt.savefig('deviances_models_weighted.png', dpi=300)

