This is the code used to produce the results and visualizations published in

Schmittwilken, L., Wichmann, F. A., & Maertens, M. (2023). Standard models of spatial vision mispredict edge sensitivity at low spatial frequencies. Vision Research, 222. doi:[10.1016/j.visres.2024.108450](https://doi.org/10.1016/j.visres.2024.108450)

## Setup

Install all the libraries in  `requirements.txt`.
```bash
pip install -r requirements.txt
```
Note: we have used an older version of `python-psignifit` here, which is not available anymore. Therefore, we decided to add it to the repo directly in the folder [psignifit](psignifit). You can find information on the newest version of psignifit [here](https://github.com/wichmann-lab/python-psignifit).

## Description
The repository contains the following:

- Code for empirically testing edge sensitivity in noise and the psychophyical data: [experiment](experiment). If you want to run the experiment, you need to install the `HRL` library. For this, follow the instructions [here](https://github.com/computational-psychology/hrl).

- Code to set up and optimize all the variations of the standard spatial vision model as described in the paper: [simulations](simulations). To create the noise masks for the simulation, run [create_noises.py](simulations/create_noises.py). To optimize the single-scale model, run [optimize_single.py](simulations/optimize_single.py). To optimize the multi-scale model, run [optimize_multi.py](simulations/optimize_multi.py). Since all variable parameters are part of the normalization-step, both scripts will first run and save and model outputs to disc to save compute time.

- Code to create the visualizations from the manuscript and explore the empirical data and the model: [visualizations](visualizations). In order to re-create the visualizations, first run the simulations to produce the respective results.
- An old version of `python-psignifit`: [psignifit](psignifit)

## Authors and acknowledgment
Code written by Lynn Schmittwilken (l.schmittwilken@tu-berlin.de)
