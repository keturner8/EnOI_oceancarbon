# EnOI_oceancarbon
Code and mask data for reconstructing ocean carbon content using Ensemble Optimal Interpolation

Code prepared for submission to Biogeosciences, associated with the manuscript "Reconstructing ocean carbon storage with CMIP6 models and synthetic Argo observations" by
Katherine Turner, Doug Smith, Anna Katavouta, and Ric Williams

Code folders and files are as follows:

** removing drifts from CMIP6 output ***

dedrifting_pp : folder of preprocessing scripts to remove drifts from CMIP6 T,S, and DIC fields. Here, the drift is approximated as the linear trend at each point (x,y,z) calculated within piControl output. Realisations of the models are assumed to have the same drift, i.e. only one drift is calculated for each model (such as ACCESS or IPSL)

** datasets/functions/preprocessing for ENOI ***

annual_avgs : folder of zsh scripts used to calculate annual average ocean fields from monthly average fields provided by CMIP6-ESGF

argo_mask.nc : NC file that contains number of months with observations for each 1 degree horizontal bin in a given year, starting at year 2002 and ending at year 2015

mask_ens.npy : land/sea/numerical issue mask for each model once they had been regridded from their native grids to the standard 1x1 degree horizontal grid

multimodel_ens.py : creation of a standardised "super-timeseries" created by concatenating all model ensemble runs, in order to calculate covariance/correlation fields

regression_fcts.py : set of regression functions used in python scripts

regridding.py : script for regridding annual average model output onto a 1x1 horizontal grid using xESMF

*** ENOI calculation and figure creation ***

calc_optimal_coefficients.ipynb : jupyter notebook for calculating optimal coefficients (using co-located observations) for different depth levels and saving output as netcdf files

coefficients_depth.ipynb : jupyter notebook producing figure with optimal coefficients for different interior layers

reconstruction_100m_{dense,argo}.ipynb : jupyter notebook producing figures for reconstructing top 100m carbon using colocated T and S profiles (dense) and year 2015 T and S Argo-style profiles (argo)

reconstruction_depth.ipynb : jupyter notebook producing figures for reconstructing interior carbon using colocated T and S profiles

reconstruction_NorESM.ipynb : jupyter notebook producing figures for NorESM test case using time-varying coverage consistent with existing Argo profiles
