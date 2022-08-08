# EnOI_oceancarbon
Code and mask data for reconstructing ocean carbon content using Ensemble Optimal Interpolation

Code prepared for submission to Biogeosciences, associated with the manuscript "Reconstructing ocean carbon storage with CMIP6 models and
synthetic Argo observations" by
Katherine Turner, Doug Smith, Anna Katavouta, and Ric Williams

Code files are as follows:

annual_avgs : folder of zsh scripts used to calculate annual average ocean fields from monthly average fields provided by CMIP6-ESGF

argo_mask.nc : NC file that contains number of months with observations for each 1 degree horizontal bin in a given year, starting at year 2002 and ending at year 2015

argo_reconstruct_method2.py : python script for creating reconstructions using co-located temperature and salinity profiles. Includes scripts used to run sensitivity tests

man_argo_plots.py : python script for creating reconstructions using temperature and salinity profiles at Argo locations

mask_ens.npy : land/sea/numerical issue mask for each model once they had been regridded from their native grids to the standard 1x1 degree horizontal grid

multimodel_ens.py : creation of a standardised "super-timeseries" created by concatenating all model ensemble runs, in order to calculate covariance/correlation fields

recon_global_fig.py : python script to create reconstruction figures using the Norwegian ESM

regression_fcts.py : set of regression functions used in python scripts

regridding.py : script for regridding annual average model output onto a 1x1 horizontal grid using xESMF
