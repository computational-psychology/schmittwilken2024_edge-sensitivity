"""
This script creates the noise masks needed to optimize the models.
This scripts needs to be run before optimizing the models.

@author: Lynn Schmittwilken
Last update: June 2024
"""

import numpy as np
import sys
from functions import create_directory, create_noise

sys.path.append('../experiment')
from params import stim_params
# from stimulus_functions import create_whitenoise, create_pinknoise, create_narrownoise
from helper_functions import save_mask

n_masks = 50                     # How many noise masks to create?
file_path = "./noise_masks/"     # Where to save them?

if __name__ == "__main__":
    np.random.seed(23)

    sp = stim_params
    stim_size = sp["stim_size"]
    ppd = sp["ppd"]
    rms = sp["noise_contrast"] * sp["mean_lum"]
    create_directory(file_path, True)

    # Create and save white noise
    noise_dir = file_path + sp["noise_types"][1] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        # noise = create_whitenoise(size=stim_size*ppd, rms_contrast=rms)
        noise = create_noise(sp["noise_types"][1], sp)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create pink1 noise
    noise_dir = file_path + sp["noise_types"][2] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        # noise = create_pinknoise(size=stim_size*ppd, ppd=ppd, rms_contrast=rms, exponent=1.)
        noise = create_noise(sp["noise_types"][2], sp)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create pink2 / brown noise
    noise_dir = file_path + sp["noise_types"][3] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        # noise = create_pinknoise(size=stim_size*ppd, ppd=ppd, rms_contrast=rms, exponent=2.)
        noise = create_noise(sp["noise_types"][3], sp)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create narrowband noise with center frequency of 0.5 cpd
    noise_dir = file_path + sp["noise_types"][4] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        # noise = create_narrownoise(size=stim_size*ppd, noisefreq=0.5, ppd=ppd, rms_contrast=rms)
        noise = create_noise(sp["noise_types"][4], sp)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create narrowband noise with center frequency of 3. cpd
    noise_dir = file_path + sp["noise_types"][5] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        # noise = create_narrownoise(size=stim_size*ppd, noisefreq=3., ppd=ppd, rms_contrast=rms)
        noise = create_noise(sp["noise_types"][5], sp)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")

    # Create narrowband noise with center frequency of 9. cpd
    noise_dir = file_path + sp["noise_types"][6] + "/"
    create_directory(noise_dir)
    for i in range(n_masks):
        # noise = create_narrownoise(size=stim_size*ppd, noisefreq=9., ppd=ppd, rms_contrast=rms)
        noise = create_noise(sp["noise_types"][6], sp)
        save_mask(noise, sp, noise_dir + str(i) + ".pickle")
