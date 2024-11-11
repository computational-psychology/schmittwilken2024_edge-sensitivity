This is the code used to produce the results and visualizations published in

Schmittwilken, L., Wichmann, F. A., & Maertens, M. (2023). Standard models of spatial vision mispredict edge sensitivity at low spatial frequencies. Vision Research, 222. [doi:10.1016/j.visres.2024.108450](https://doi.org/10.1016/j.visres.2024.108450)

## Description
The repository contains the following:

- The code for empirically testing edge sensitivity in noise (with [hrl](https://github.com/computational-psychology/hrl)) and the psychophyical data: [experiment](experiment)

- The code to set up and optimize all the variations of the standard spatial vision model as described in the paper: [simulations](simulations). To create the noise masks for the simulation, run [create_noises.py](simulations/create_noises.py). To optimize the single-scale model, run [optimize_single.py](simulations/optimize_single.py). To optimize the multi-scale model, run [optimize_multi.py](simulations/optimize_multi.py). Since all variable parameters are part of the normalization-step, both scripts will first run and save and model outputs to disc to save compute time.

- Code to create the visualizations from the manuscript and a model walkthrough: [visualizations](visualizations). In order to re-create the visualizations, first run the simulations to produce the respective results.

## Authors and acknowledgment
Code written by Lynn Schmittwilken (l.schmittwilken@tu-berlin.de)
