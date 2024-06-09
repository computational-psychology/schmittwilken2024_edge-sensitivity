"""
These functions are the basis for the modeling part

@author: Lynn Schmittwilken
Last update: June 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm, binom
from scipy.signal import fftconvolve
import os
import pickle
import itertools
import psignifit as ps
from stimupy.noises.whites import white as create_whitenoise
from stimupy.noises.narrowbands import narrowband as create_narrownoise
from stimupy.noises.naturals import one_over_f as create_pinknoise

sys.path.append('../experiment')
from stimulus_functions import cornsweet_illusion
from helper_functions import load_mask


# %%
###############################
#       Helper functions      #
###############################
def print_progress(count, total):
    """Helper function to print progress.

    Parameters
    ----------
    count
        Current iteration count.
    total
        Total number of iterations.

    """
    percent_complete = float(count) / float(total)
    msg = "\rProgress: {0:.1%}".format(percent_complete)
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def create_directory(dir_path, askPermission=False):
    if os.path.exists(dir_path) == 1:
        if askPermission:
           out = input("Directory already exists. Override content? (y / n)")
           if not (out == "y" or out == "yes"): raise SystemExit(0)
    else:
        os.makedirs(dir_path)


def remove_padding(array, rb):
    array_shape = array.shape
    array = array[:, rb:array_shape[1]-rb]
    return array


def add_padding(array, rb, val):
    h, w = array.shape
    new_array = np.ones([h, w+rb*2]) * val
    new_array[:, rb:w+rb] = array
    return new_array


def load_params(results_file):
    # Load data from pickle:
    with open(results_file, 'rb') as handle:
        data_pickle = pickle.load(handle)
    
    best_params = data_pickle["best_params_auto"]
    best_loss = data_pickle["best_loss_auto"]
    model_params = data_pickle["model_params"]
    print("Best loaded loss:", best_loss)
    return best_params, model_params


# %%
###############################
#       Stimulus-related      #
###############################
def create_edge(contrast, width, stim_params):
    edge = cornsweet_illusion(stim_params["stim_size"],
                              stim_params["ppd"],
                              'rms',
                              contrast,
                              width,
                              stim_params["edge_exponent"],
                              stim_params["mean_lum"])
    return edge


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


def create_noises(sparams, nInstances):
    noiseDict = {}
    for ni, n in enumerate(sparams["noise_types"]):
        noiseList = []
        for t in range(nInstances):
            noiseList.append(create_noise(n, sparams))
        noiseDict[n] = noiseList
    return noiseDict


def pull_noise_mask(noise_type, sparams, trial, mask_path="./noise_masks/"):
    # maskID = np.random.randint(0, sparams["n_masks"])
    maskID = trial
    if noise_type == 'none':
        nX = int(sparams["stim_size"]*sparams["ppd"])
        noise = np.zeros([nX, nX])
    elif noise_type == 'white':
        noise, _ = load_mask(mask_path + "white/" + str(maskID) + ".pickle")
    elif noise_type == 'pink1':
        noise, _ = load_mask(mask_path + "pink1/" + str(maskID) + ".pickle")
    elif noise_type == 'pink2':
        noise, _ = load_mask(mask_path + "pink2/" + str(maskID) + ".pickle")
    elif noise_type == 'narrow0.5':
        noise, _ = load_mask(mask_path + "narrow0.5/" + str(maskID) + ".pickle")
    elif noise_type == 'narrow3':
        noise, _ = load_mask(mask_path + "narrow3/" + str(maskID) + ".pickle")
    elif noise_type == 'narrow9':
        noise, _ = load_mask(mask_path + "narrow9/" + str(maskID) + ".pickle")
    else:
        raise ValueError("noise_type unknown")
    return noise


# %%
###############################
#        Model-related        #
###############################
def create_lowpass(fx, fy, radius, sharpness):
    # Calculate the distance of each frequency from requested frequency
    distance = radius - np.sqrt(fx**2. + fy**2.)
    distance[distance > 0] = 0
    distance = np.abs(distance)

    # Create bandpass filter:
    lowpass = 1. / (np.sqrt(2.*np.pi) * sharpness) * np.exp(-(distance**2.) / (2.*sharpness**2.))
    lowpass = lowpass / lowpass.max()
    return lowpass


def create_loggabor_fft(fx, fy, fo, sigma_fo, angleo, sigma_angleo):
    nY, nX = fx.shape
    fr = np.sqrt(fx**2. + fy**2.)
    fr[int(nY/2), int(nX/2)] = 1.

    # Calculate radial component of the filter:
    radial = np.exp((-(np.log(fr/fo))**2.) / (2. * np.log(sigma_fo)**2.))
    radial[int(nY/2), int(nX/2)] = 0.  # Undo radius fudge
    fr[int(nY/2), int(nX/2)] = 0.      # Undo radius fudge

    # Multiply radial part with lowpass filter to achieve even coverage in corners
    # Lowpass filter will limit the maximum frequency
    lowpass = create_lowpass(fx, fy, radius=fx.max(), sharpness=1.)
    radial = radial * lowpass

    # Calculate angular component of log-Gabor filter
    theta = np.arctan2(fy, fx)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # For each point in the polar-angle-matrix, calculate the angular distance
    # from the filter orientation
    ds = sintheta * np.cos(angleo) - costheta * np.sin(angleo)  # difference in sin
    dc = costheta * np.cos(angleo) + sintheta * np.sin(angleo)  # difference in cos
    dtheta = np.abs(np.arctan2(ds, dc))                         # absolute angular distance
    angular = np.exp((-dtheta**2.) / (2. * sigma_angleo**2.))   # calculate angular filter component
    return angular * radial  # loggabor is multiplication of both


def create_loggabor(fx, fy, fo, sigma_fo, angleo, sigma_angleo):
    # Create loggabor and ifft
    loggabor_fft = create_loggabor_fft(fx, fy, fo, sigma_fo, angleo, sigma_angleo)
    loggabor = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(loggabor_fft)))
    loggabor_even = loggabor.real          # real part = even-symmetric filter
    loggabor_odd = np.real(loggabor.imag)  # imag part = odd-symmetric filter
    return loggabor_even, loggabor_odd


def create_loggabors(fx, fy, fos, sigma_fo, angleo, sigma_angleo):
    n_filters = len(fos)
    loggabors = []
    for f in range(n_filters):
        _, loggabor_odd = create_loggabor(fx, fy, fos[f], sigma_fo, 0., sigma_angleo)
        loggabors.append(loggabor_odd)
    return loggabors


def create_loggabors_fft(fx, fy, fos, sigma_fo, angleo, sigma_angleo):
    n_filters = len(fos)
    loggabors = []
    for f in range(n_filters):
        loggabor = create_loggabor_fft(fx, fy, fos[f], sigma_fo, 0., sigma_angleo)
        loggabors.append(loggabor)
    return loggabors


# Naka Rushton adapted from Felix thesis
def naka_rushton(inpt, alpha, beta, eta, kappa, gain=None):
    alpha = np.expand_dims(np.array(alpha), axis=(0, 1))
    denom = inpt**kappa
    
    # Additional gain control mechanisms
    if gain == "global": # norm by global activity
        denom = denom.mean()
    elif gain == "channel": # norm by global activity within channels
        denom = denom / denom.mean((0, 1))
    elif gain == "local": # norm by local activity across channels
        denom = denom / np.expand_dims(denom.mean(2), -1)
    elif gain == "spatial": # norm by neighboring activity within channels
        lgsize = [94, 16, 4]
        for i in range(len(lgsize)):
            convf = np.ones([lgsize[i], lgsize[i]]) / lgsize[i]**2.
            denom[:,:,i] = fftconvolve(denom[:,:,i], convf, 'same')
    return alpha * inpt**(eta+kappa) / (denom + beta)


def apply_filters(stim, mparams):
    stim = add_padding(stim, mparams["fac"], stim.mean())  # Padding

    out = np.zeros([mparams["nX"], mparams["nX"], mparams["n_filters"]])
    for fil in range(mparams["n_filters"]):
        outTemp = np.abs(fftconvolve(stim, mparams["loggabors"][fil], 'same'))
        outTemp = remove_padding(outTemp, mparams["fac"])  # Remove padding
        out[:, :, fil] = outTemp
    return out


def create_filter_outputs(edge, noise1, noise2, mparams):
    mout1 = apply_filters(edge+noise1, mparams)
    
    if mparams["sameNoise"]:
        mout2 = apply_filters(noise1+edge.mean(), mparams)
    else:
        mout2 = apply_filters(noise2+edge.mean(), mparams)
        
        # if different noises are used, weight by edge profile
        sweight = np.abs(edge - edge.mean())
        mout1 = mout1 * np.expand_dims(sweight/sweight.max(), -1)
        mout2 = mout2 * np.expand_dims(sweight/sweight.max(), -1)
    return mout1, mout2


def compute_dprime(r1, r2, lamb=0.005, noiseVar=1.):
    dprime = (r1 - r2).sum() / np.sqrt( noiseVar*np.ones(r1.shape).sum())
    pc = norm.cdf(dprime)
    return lamb + (1. - 2.*lamb) * pc  # consider lapses


def compute_performance(mout1, mout2, mparams, alphas, beta, eta, kappa, lamb=0.005):
    # Apply Naka-rushton to filter outputs
    mout1 = naka_rushton(mout1, alphas, beta, eta, kappa, mparams["gain"])
    mout2 = naka_rushton(mout2, alphas, beta, eta, kappa, mparams["gain"])
    
    # Compute dprime
    pc = compute_dprime(mout1, mout2, lamb, mparams["noiseVar"])
    return pc


def save_filter_outputs(sparams, mparams, df, outDir):
    create_directory(outDir, True)   # Create directory
    
    for n in sparams["noise_types"]:
        for e in sparams["edge_widths"]:
            df_cond = df[(df["noise"]==n) & (df["edge"]==e)]
            contrasts = df_cond["contrasts"].to_numpy()
            for i, c in enumerate(contrasts):
                for t in range(mparams["n_trials"]):
                    # Create stims
                    noise1 = pull_noise_mask(n, sparams, t)
                    noise2 = pull_noise_mask(n, sparams, (t+1)%mparams["n_trials"])
                    edge = create_edge(c, e, sparams)
                    mout1, mout2 = create_filter_outputs(edge, noise1, noise2, mparams)
                    
                    results_file = outDir + "/%s_%.3f_%i_%i.pickle" % (n, e, i, t)
                    print(results_file)
                    save_dict = {"mout1": mout1, "mout2": mout2}
                    
                    with open(results_file, 'wb') as handle:
                        pickle.dump(save_dict, handle)


def load_filter_outputs(inDir):
    with open(inDir, 'rb') as handle:
        data_pickle = pickle.load(handle)
    mout1 = data_pickle["mout1"]
    mout2 = data_pickle['mout2']
    return mout1, mout2


# %%
###############################
#         Read data           #
###############################
def load_data(datadir):
    all_data = []

    # Read experimental data from all the subfolders
    filenames = os.listdir(datadir)
    for filename in filenames:
        if filename.startswith("experiment_"):
            data = pd.read_csv(os.path.join(datadir, filename), delimiter='\t')
            all_data.append(data)
    all_data = pd.concat(all_data) # Concat dataframes

    # Reset the indices
    all_data = all_data.reset_index(drop=True)
    return all_data


def load_data_two_sessions(datadir1, datadir2):
    df1 = load_data(datadir1)
    df1['session'] = 0
    try:
        df2 = load_data(datadir2)
        df2['session'] = 1
        df = pd.concat([df1, df2])
    except:
        df = df1
    df = df.reset_index(drop=True)
    return df


def load_all_data(folder_dir, subject_dirs):
    n_sub = len(subject_dirs)
    data_all = []

    for i in range(n_sub):
        datadir1 = folder_dir + subject_dirs[i] + '/experiment'
        datadir2 = folder_dir + subject_dirs[i] + '2/experiment'
        data_sub = load_data_two_sessions(datadir1, datadir2)

        # Add a column for the subject ID
        vp_id = np.repeat(i, data_sub.shape[0])
        data_sub['vp'] = vp_id

        # Add data of individuals to the full dataset
        data_all.append(data_sub)

    # Concat dataframes of all subjects:
    data_all = pd.concat(data_all)
    return data_all


def reformat_data(data, noise_cond, edge_cond):
    # Get data from one condition
    data = data[(data["noise"] == noise_cond) &
                (data["edge_width"] == edge_cond)]
    
    # For each contrast, get number of (correct) trials
    contrasts = np.unique(data["edge_contrast"])
    n_correct = np.zeros(len(contrasts))
    n_trials = np.zeros(len(contrasts))
    for i in range(len(contrasts)):
        data_correct = data[data["edge_contrast"] == contrasts[i]]["correct"]
        n_correct[i] = np.sum(data_correct)
        n_trials[i] = len(data_correct)
    return contrasts, n_correct, n_trials


def get_lapse_rate(contrasts, ncorrect, ntrials):
    psignifit_data = np.array([contrasts, ncorrect, ntrials]).transpose()
    
    options = {
        "sigmoidName": "norm",
        "expType": "2AFC",
        }

    sys.stdout = open(os.devnull, 'w')                # suppress print
    results = ps.psignifit(psignifit_data, options)
    lamb = results["Fit"][2]
    sys.stdout = sys.__stdout__                       # enable print
    return lamb


# %%
###############################
#         Optimization        #
###############################
# Function which computes log-likelihood
def log_likelihood(y, n, p):
    # y: hits, n: trials, p: model performance
    return binom.logpmf(y, n, p).sum()


def calc_deviance_residual(y, n, p, eta=1e-08):
    # Calculate deviance residuals as explained in Wichmann & Hill (2001)
    # y: hits, n: trials, p: model performance
    phuman = y/n
    p1 = n * phuman * np.log(eta + phuman/p)
    p2 = n * (1 - phuman) * np.log( eta + (1-phuman) / (1-p+eta) )
    out = np.sign(phuman-p) * np.sqrt(2*(p1 + p2))
    return out


def grid_optimize(results_file, params_dict, loss_func, additional_dict=None):
    # Check whether pickle exists. If so, load data from pickle file and continue optimization
    if os.path.isfile(results_file):
        print("-----------------------------------------------------------")
        print('Results file already exists')
        print('Loading existing optimization data from pickle...')
        print("-----------------------------------------------------------")

        # Load data from pickle:
        with open(results_file, 'rb') as handle:
            data_pickle = pickle.load(handle)

        params_dict = data_pickle["params_dict"]
        best_loss = data_pickle['best_loss']
        best_params = data_pickle['best_params']
        last_set = data_pickle['last_set']

    else:
        print("-----------------------------------------------------------")
        print("No results file found")
        print('Initiating optimization from scratch...')
        print("-----------------------------------------------------------")
        best_loss = np.inf
        best_params = []
        last_set = 0

    # Prepare list of params to run:
    n_params = len(params_dict.keys())
    print("Specified " + str(n_params) + " parameters.")
    print(params_dict.keys())
    print("-----------------------------------------------------------")
    print("Best starting loss", best_loss)
    print("Best starting params", best_params)
    print("-----------------------------------------------------------")

    # Get all individual parameter lists
    pind = []
    for key in params_dict.keys():
        pind.append(params_dict[key])

    # Get list with all parameter combinations
    pall = list(itertools.product(*pind))
    total = len(pall)

    for i in range(last_set, total):
        # Print progress and get relevant params:
        print_progress(count=i+1, total=total)
        p = {}
        for j, key in enumerate(params_dict.keys()):
            p[key] = pall[i][j]
        loss = loss_func(p)

        if loss < best_loss:
            best_loss = loss
            best_params = p
            print()
            print('New best loss:', best_loss)
            print('Parameters: ', best_params)
            print("-----------------------------------------------------------")

        # Save params and results in pickle:
        save_dict = {'params_dict': params_dict,
                     'last_set': i+1,
                     'best_loss': best_loss,
                     'best_params': best_params,
                     'total': total,
                     }

        if additional_dict is not None:
            save_dict.update(**additional_dict)

        with open(results_file, 'wb') as handle:
            pickle.dump(save_dict, handle)
    return best_params, best_loss


# %%
###############################
#        Visualizations       #
###############################
def plotPsych(result,
              color          = [0, 0, 0],
              alpha          = 1,
              plotData       = True,
              lineWidth      = 1,
              plotAsymptote  = False,
              extrapolLength = .2,
              dataSize       = 0,
              axisHandle     = None,
              showImediate   = False,
              plotCI         = False,
              xlim           = None,
              ):
    """
    This function plots the fitted psychometric function alongside 
    the data. Adapted from the Python-Psignifit plotPsych-function

    Author: Lynn Schmittwilken
    """
    
    fit, data, options = result['Fit'], result['data'], result['options']
    if np.isnan(fit[3]): fit[3] = fit[2]
    if data.size == 0: return
    if dataSize == 0: dataSize = 10000. / np.sum(data[:,2])
    ax = plt.gca() if axisHandle == None else axisHandle
    
    # PLOT DATA
    ymin = 1. / options['expN']
    ymin = min([ymin, min(data[:,1] / data[:,2])])
    xData = data[:,0]
    if plotData:
        yData = data[:,1] / data[:,2]
        ax.plot(xData, yData, '.', c=color, clip_on=False, alpha=alpha)
    
    # PLOT FITTED FUNCTION
    xMin = min(xData)
    xMax = max(xData)
    xLength = xMax - xMin
    x       = np.linspace(xMin, xMax, num=1000)
    xLow    = np.linspace(xMin - extrapolLength*xLength, xMin, num=100)
    xHigh   = np.linspace(xMax, xMax + extrapolLength*xLength, num=100)
    
    fitValuesLow  = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xLow,  fit[0], fit[1]) + fit[3]
    fitValuesHigh = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](xHigh, fit[0], fit[1]) + fit[3]
    fitValues     = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x,     fit[0], fit[1]) + fit[3]
    
    ax.plot(x,     fitValues,           c=color, lw=lineWidth, clip_on=True, alpha=alpha)
    ax.plot(xLow,  fitValuesLow,  '--', c=color, lw=lineWidth, clip_on=True, alpha=alpha)
    ax.plot(xHigh, fitValuesHigh, '--', c=color, lw=lineWidth, clip_on=True, alpha=alpha)
    
    if xlim is None:
        xmin, xmax = xLow.min(), xHigh.max()
        xlim = (xmin, xmax)
    
    # Add dashed line at 50% and 100% - lapse-rate
    if plotAsymptote:
        ax.hlines(y=1./options['expN'], xmin=xlim[0], xmax=xlim[1], ls='--', lw=0.8, colors="k")
        ax.hlines(y=1.0-fit[2], xmin=xlim[0], xmax=xlim[1], ls='--', lw=0.8, colors="k")
    ax.set(xlim=xlim)
    
    if plotCI:
        CIl = []
        CIu = []
        ys = (1 - fit[2] - fit[3]) * options['sigmoidHandle'](x, fit[0], fit[1]) + fit[3]
        for i in range(len(ys)):
            [threshold, CI] = ps.getThreshold(result, ys[i])  # res, pc
            CIl.append(CI[2, 0])  # 68% CI
            CIu.append(CI[2, 1])
        ax.fill_betweenx(ys, CIl, CIu, color=color, alpha=0.2)

    if showImediate:
        plt.show()
    return axisHandle
