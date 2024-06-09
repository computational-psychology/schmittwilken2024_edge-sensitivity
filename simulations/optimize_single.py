"""
Script to optimize performance of single-scale / csf edge model

@author: Lynn Schmittwilken
Last update: June 2024
"""

import numpy as np
import pandas as pd
import pickle
from time import time
import sys
from scipy.optimize import minimize
from functions import create_loggabors, compute_performance, get_lapse_rate, \
    grid_optimize, load_all_data, log_likelihood, reformat_data, \
    save_filter_outputs, load_filter_outputs

sys.path.append('../experiment')
from params import stim_params as sparams

np.random.seed(23) # random seed for reproducibility


####################################
#           Parameters             #
####################################
results_file = "results_single.pickle"

# Model params
fos = [2.64553923,]              # fitted to CSF
sigma_fo = 0.49796013            # fitted to CSF
sigma_angleo = 0.2965            # from Schütt & Wichmann (2017)
n_trials = 50                    # average performance over n-trials
noiseVar = 1.                    # magnitude of internal noise

gain = None                      # None, global, channel, local, spatial
sameNoise = True                 # use same or different noise instances?
outDir = "./outputs_single"      # directory to save filter outputs

n_filters = len(fos)
ppd = sparams["ppd"]
fac = int(ppd*2)                 # padding to avoid border artefacts
sparams["n_masks"] = n_trials    # use same noise masks everytime


####################################
#            Read data             #
####################################
print("Loading psychophysical data and computing lapse rates ... (takes a moment)")
vps = ["ls", "mm", "jv", "ga", "sg", "fd"]
n_vps = len(vps)

# Load data
datadir = "../experiment/results/"
data = load_all_data(datadir, vps)

# Reformat data
noise_conds = np.unique(data["noise"])
edge_conds = np.unique(data["edge_width"])

df_list = []
for n in noise_conds:
    for e in edge_conds:
        contrasts, ncorrect, ntrials = reformat_data(data, n, e)
        lamb = get_lapse_rate(contrasts, ncorrect, ntrials)

        df = pd.DataFrame({
            "noise": [n,]*len(contrasts),
            "edge": [e,]*len(contrasts),
            "contrasts": contrasts,
            "ncorrect": ncorrect,
            "ntrials": ntrials,
            "lambda": [lamb,]*len(contrasts),
            })
        df_list.append(df)
df = pd.concat(df_list).reset_index(drop=True)


####################################
#           Preparations           #
####################################
# Calculate spatial frequency axis in cpd:
nX = int(sparams["stim_size"] * ppd)
fs = np.fft.fftshift(np.fft.fftfreq(int(nX), d=1./ppd))
fx, fy = np.meshgrid(fs, fs)

# Create loggabor filters
loggabors = create_loggabors(fx, fy, fos, sigma_fo, 0., sigma_angleo)

# Constant model params
mparams = {"n_filters": n_filters,
           "fos": fos,
           "sigma_fo": sigma_fo,
           "sigma_angleo": sigma_angleo,
           "loggabors": loggabors,
           "fac": fac,
           "nX": nX,
           "n_trials": n_trials,
           "gain": gain,
           "outDir": outDir,
           "sameNoise": sameNoise,
           "noiseVar": noiseVar,
           }

adict = {"model_params": mparams,
          "stim_params": sparams}


def get_loss(params):
    # Read params from dict / list
    if type(params) is dict:
        beta, eta, kappa = params["beta"], params["eta"], params["kappa"]
        alphas = [value for key, value in params.items() if "alpha" in key.lower()]

    else:
        beta, eta, kappa = params[0], params[1], params[2]
        alphas = params[3::]

    # Infinite loss if all alphas are zero
    if sum(alphas) == 0:
        return np.inf
    
    # Infinite loss if alphas are negative
    if any(x < 0 for x in alphas):
        return np.inf
    if any(x < 0 for x in [beta, eta, kappa]):
        return np.inf
    if kappa > 6:
        return np.inf
    if beta == 0:
        beta = 1e-10  # beta should not be 0 to avoid division by zero in Naka-Rushton

    # Run model for each contrast in each condition
    LLs = []
    for n in noise_conds:
        for e in edge_conds:
            df_cond = df[(df["noise"]==n) & (df["edge"]==e)]
            contrasts = df_cond["contrasts"].to_numpy()
            ncorrect = df_cond["ncorrect"].to_numpy()
            ntrials = df_cond["ntrials"].to_numpy()
            lamb = np.unique(df_cond["lambda"].to_numpy())[0]

            for i, c in enumerate(contrasts):
                pc = np.zeros(n_trials)
                
                for t in range(n_trials):
                    # Load filter outputs
                    name = outDir + "/%s_%.3f_%i_%i.pickle" % (n, e, i, t)
                    mout1, mout2 = load_filter_outputs(name)
                    pc[t] = compute_performance(mout1, mout2, mparams, alphas, beta, eta, kappa, lamb)

                # Compute log-likelihood
                LLs.append(log_likelihood(y=ncorrect[i], n=ntrials[i], p=pc.mean()))
    return -sum(LLs)


####################################
#           Simulations            #
####################################
start = time()

# Filter outputs dont change with model params, so save+load to reduce runtime drastically
print('------------------------------------------------')
print('Creating all filter outputs and save them to memory for computational efficiency')
save_filter_outputs(sparams, mparams, df, outDir)

# Initial parameter guesses for grid search
params_dict = {
    "beta": np.linspace(0.5, 2.5, 5),
    "eta": np.linspace(0., 1., 5),
    "kappa": np.linspace(0., 4., 5),
    "alpha":  np.linspace(0., 5., 5),
    }

# Run / Continue optimization
print('------------------------------------------------')
print('Starting optimization:', results_file)
print('------------------------------------------------')
bparams, bloss = grid_optimize(results_file, params_dict, get_loss, adict)

print()
print("Best params:", bparams)
print("Best loss:", bloss)
print('------------------------------------------------')
print('Elapsed time: %.2f minutes' % ((time()-start) / 60.))

# Use best params from grid search for initial guess for optimizer
automatic_grid = True
if automatic_grid:
    start = time()
    res = minimize(get_loss,
                   list(bparams.values()),
                   method='Nelder-Mead',      # Nelder-Mead (=Simplex)
                   options={"maxiter": 1500},
                   )

    # Save final results to pickle
    with open(results_file, 'rb') as handle:
        data_pickle = pickle.load(handle)

    data_pickle["best_loss_auto"] = res.fun
    data_pickle["best_params_auto"] = {
        "beta": res.x[0],
        "eta": res.x[1],
        "kappa": res.x[2],
        "alpha": res.x[3],
        }
    data_pickle["overview_auto"] = res
        
    with open(results_file, 'wb') as handle:
        pickle.dump(data_pickle, handle)

    print('------------------------------------------------')
    print(res)
    print('------------------------------------------------')
    print('Elapsed time: %.2f minutes' % ((time()-start) / 60.))
