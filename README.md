# Waveome

<!-- ![waveome logo](./figures/hex-waveome.png) -->
<!-- <img src="./figures/hex-waveome.png" width="200" height="200"> -->

## Overview
<img style="float: right;" width="200" height="200" src="./figures/hex-waveome.png">

This repository houses code for the _Waveome_ package - an easy to use and powerful Python library that analyzes longitudinal data using Gaussian processes. 

## General Usage
Please see the `waveome_overview.ipynb` for an example of the package modeling simulated data.

## Installation
To install the package it is suggested that you create a new conda environment   
`conda create -n waveome python=3.8.12`   
`conda activate waveome`   
and then pip install _waveome_ using    
`pip install git+https://github.com/omicsEye/waveome.git`.

Warning: If you are using an M1 Mac then before `pip install` you should run `conda install -c conda-forge grpcio`!

## Jupyter Notebook
If you would like to run `waveome_overview.ipynb` then you should also set up a Jupyter kernel for the new waveome environment. This can be done with `conda install ipykernel` and then `python -m ipykernel install --user --name=waveome`.