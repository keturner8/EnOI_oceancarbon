 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:54:28 2021

@author: keturner
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import xesmf as xe
from xesmf.backend import (esmf_grid, esmf_regrid_build,
                           esmf_regrid_apply, esmf_regrid_finalize)
from pathlib import Path
import glob

fdir = '/Users/keturner/nc_files/CMIP6/'

regrid_dirs = ['mACCESS/', 'mCanESM/', 'mIPSL/', 'mMPI/', 'mUKESM/', 'mCESM2/']
regrid_dirs = ['mMPI/']
variable = 'dissic_Oyr'
v = 'dissic'
output_dir = 'regrid_horiz/'

ncfiles = []
for moddir in regrid_dirs:
    for file in glob.glob(fdir + moddir+ variable + "_*.nc", recursive=True):
        file_split = file.split(variable)
        file_regrid = fdir+output_dir+variable+file_split[1][:-3]+'_1x1r.nc'
        if Path(file_regrid).exists():
                print(file+' has already been regridded')
        else:
                ncfiles.append(file)         

for file in ncfiles:
    ds = xr.open_dataset(file)

    if ('ACCESS' in file) or ('CanESM' in file) or ('NorESM' in file) or ('UKESM' in file) or ('MRI' in file) or ('MPI' in file):
        ds = ds.rename({'longitude':'lon', 'latitude':'lat'})
    if 'IPSL' in file:
        ds = ds.rename({'nav_lat':'lat', 'nav_lon':'lon', 'olevel': 'lev', 'olevel_bounds':'lev_bnds'})

    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.75, 90, .5)), 'lon': (['lon'], np.arange(0.25, 360, .5)), 
                         'lat_b': (['lat_b'], np.arange(-90, 90.5,.5)), 'lon_b': (['lon_b'], np.arange(0,360.5,.5))})

    regridder = xe.Regridder(ds, ds_out, 'bilinear', ignore_degenerate=True, periodic=True)

    dr = ds[v][-60:,:,:,:]
    dr_regrid = regridder(dr)
    var_r = dr_regrid.where(dr_regrid != 0)
    
    merged_regrid = xr.merge([var_r, ds['lev_bnds']])
    
    file_split = file.split(variable)
    file_regrid = fdir+output_dir+variable+file_split[1][:-3]+'_05x05r.nc'
    merged_regrid.to_netcdf(file_regrid)
    print(variable+file_split[1]+ ' has been regridded')
    
## Testing to see if we can see a difference bertween bilinear and conservative regridding - seems to be OK to use bilinear

file05 = '/Users/keturner/nc_files/CMIP6/regrid_horiz/dissic_Oyr_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_1950-2014_05x05r.nc'

ds = xr.open_dataset(file05)
lat_bg = np.arange(-90, 90.5,.5)
lon_bg = np.arange(0,360.5,.5)

ds_in = xe.util.grid_global(.5, .5)  # input grid
ds_coarse = xe.util.grid_global(1, 1)

ds_in["data1"] = ds.dissic[1,0,:,:]-ds.dissic[0,0,:,:]

def regrid(ds_in, ds_out, dr_in, method):
    """Convenience function for one-time regridding"""
    regridder = xe.Regridder(ds_in, ds_out, method, periodic=True)
    dr_out = regridder(dr_in)
    return dr_out

ds_bil = regrid(ds_in, ds_coarse, ds_in['data1'], 'bilinear')
ds_con = regrid(ds_in, ds_coarse, ds_in['data1'], 'conservative')

plt.pcolormesh(ds_in.data1,vmin=-.1, vmax=.1, cmap="RdBu_r")
plt.colorbar()

plt.pcolormesh(ds_bil,vmin=-.1, vmax=.1, cmap="RdBu_r")
plt.colorbar()

plt.pcolormesh(ds_con,vmin=-.1, vmax=.1, cmap="RdBu_r")
plt.colorbar()

plt.pcolormesh(ds_bil[:,:180]-ds_con[:,180:], vmin=-.001, vmax=.001, cmap='RdBu')
plt.colorbar()

plt.pcolormesh(ds_bil[:,180:]-ds_con[:,:180], vmin=-.001, vmax=.001, cmap='RdBu')
plt.colorbar()