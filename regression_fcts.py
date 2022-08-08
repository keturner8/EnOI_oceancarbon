#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:18:51 2021

@author: keturner
"""
import numpy as np
import scipy.signal
import xarray as xr
import matplotlib.pyplot as plt

def load_model():
    fdir = '/Users/keturner/nc_files/CMIP6/'

    access_d = {'lat':'latitude', 'lon':'longitude', 'lev':'lev', 'lev_bnds':'lev_bnds'}
    canesm_d = {'lat':'latitude', 'lon':'longitude', 'lev':'lev', 'lev_bnds':'lev_bnds'}
    cesm_d = {'lat':'lat', 'lon':'lon', 'lev':'lev', 'lev_bnds':'lev_bnds'}
    gfdl_d = {'lat':'lat', 'lon':'lon', 'lev':'lev', 'lev_bnds':'lev_bnds'}
    ipsl_d = {'lat':'nav_lat', 'lon':'nav_lon', 'lev':'olevel', 'lev_bnds':'olevel_bounds'}
    noresm_d = {'lat':'latitude', 'lon':'longitude', 'lev':'lev', 'lev_bnds':'lev_bnds'}
    
    fpath_acce= fdir + 'dissic_Oyr_ACCESS-ESM1-5_historical_r1i1p1f1_gn_1950-2014.nc'
    fpath_cane= fdir + 'dissic_Oyr_CanESM5_historical_r1i1p1f1_gn_1850-2014.nc'
    fpath_cesm= fdir + 'dissic_Oyr_CESM2_historical_r1i1p1f1_gn_1850-2014.nc'
    fpath_gfdl= fdir + 'dissic_Oyr_GFDL-ESM4_historical_r1i1p1f1_gr_1950-2014.nc'
    fpath_ipsl= fdir + 'IPSL/dissic_Oyr_IPSL-CM6A-LR_historical_r32i1p1f1_gn_1850-2014.nc'
    fpath_nore= fdir + 'dissic_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014.nc'
    
    model = input('Choose model to calculate covariance matrices:\n 1 - ACCESS\n 2 - CanESM \n 3 - CESM2 \n 4 - GFDL \n 5 - IPSL \n 6 - NorESM \nPut number here: ')
    
    if model == '1': 
        print('Loading ACCESS model')
        fpath = fpath_acce
        mdict = access_d
        fname = 'ACCESS'
    elif model == '2':
        print('Loading CanESM model')
        fpath = fpath_cane
        mdict = canesm_d
        fname = 'CanESM'
    elif model == '3':
        print('Loading CESM2 model')
        fpath = fpath_cesm
        mdict = cesm_d
        fname = 'CESM2'
    elif model == '4':
        print('Loading GFDL model')
        fpath = fpath_gfdl
        mdict = gfdl_d
        fname = 'GFDL'
    elif model == '5':
        print('Loading IPSL model')
        fpath = fpath_ipsl
        mdict = ipsl_d
        fname = 'IPSL'
    elif model == '6':
        print('Loading NorESM model')
        fpath = fpath_nore
        mdict = noresm_d
        fname = 'NorESM'
        
    return fpath, mdict, fname

def csat(T, S, DIC, TA):
    S2 = S**2
    invT = 1/T
    T100 = T/100
    
    #K0 from Weiss 1974 via jml matlab script...
    k0 = np.exp(93.4517/T100 - 60.2409 + 23.3585 * np.log(T100) + \
               S * (0.023517 - 0.023656*T100 + 0.0047036*T100**2));
    
    #coefficient algorithms from OCMIP2 protocols
    #K1, K2 from Millero (1995) using Mehrbach data
    k1 = 10**(-1*(3670.7 * invT - 62.008 + 9.7944 * np.log(T) - 0.0118*S + 0.000116*S2))
    k2 = 10**(-1*(1394.7 * invT + 4.777 - 0.0184 * S + 0.000118 * S2)) 
    
        
    #as first cut we assume that the carbonate and total alkalinities are appox
    #the same -- keeps us from having to use phosphate and silicate in calculation
    
    gamm = DIC/TA #should be umol/kg but since we are just taking a fraciton it doesnt matter right now...
    Hplus = np.empty(shape = T.shape)
    
    for i in np.arange(T.shape[0]):
        dummy = (1-gamm[i,:,:])**2*(k1[i,:,:]**2)-4 * k1[i,:,:] * k2[i,:,:] * (1-2 * gamm[i,:,:])
        Hplus[i,:,:] = 0.5 *((gamm[i,:,:]-1)*k1[i,:,:]+np.sqrt(dummy))
    
    dHplus = xr.DataArray(
        data=Hplus,
        dims=["time", "y", "x"],
        coords=dict(
            nav_lat=(["y","x"], T.nav_lat.data),
            nav_lon=(["y","x"], T.nav_lat.data),
            time=T.time.data
        ),
        attrs=dict(description="mole concentration of H+ ions"))
    
    return k0, k1, k2, dHplus

def ts_pulldepths(ds, loc_y, loc_x, loc_z):   
    """
    This function interpolates model data to specific depths for time series
    """
    if np.size(loc_z) > 1:
        ts = np.empty(shape=(np.shape(ds)[0],np.size(loc_x),np.size(loc_z)))*np.nan
        for i in np.arange(np.size(loc_x)):
            ts_loc_full = find_loc_timeseries(ds,loc_y[i],loc_x[i])
            dummy = np.squeeze(interp_z(ts_loc_full, loc_z).data)
            ts[:,i,:] = dummy
        ts_final = np.reshape(ts, [np.shape(ts)[0], np.shape(ts)[1]*np.shape(ts)[2]])
        
    else:
        ts_final = np.empty(shape=(np.shape(ds)[0],np.size(loc_x)))*np.nan
        for i in np.arange(np.size(loc_x)):
            ts_loc_full = find_loc_timeseries(ds,loc_y[i],loc_x[i])
            dummy = np.squeeze(interp_z(ts_loc_full, loc_z).data)
            ts_final[:,i] = dummy       
    return ts_final
            
def ts_addlag(ts, lag_num):
    orig_timelength = np.shape(ts)[0]
    new_timelength = orig_timelength - lag_num
    
    ts_new = np.empty(shape=(new_timelength,np.shape(ts)[1],lag_num+1))*np.nan
    
    ts_new[:,:,0] = ts[lag_num:,:]
    for i in np.arange(lag_num):
        idx = i+1
        ts_new[:,:,idx] = ts[lag_num-idx:-idx,:]
    ts_final = np.reshape(ts_new, [np.shape(ts_new)[0], np.shape(ts_new)[1]*np.shape(ts_new)[2]])   
    return ts_final

def find_loc_indices(ds, lat, lon, mdict):
    # function uses nearest neighbour and euclidian distances
    # need to add something if lon is from 0 to 360 rather than -180 to 180
    if lon < 0:
        lon_min = np.min(ds[mdict['lon']].data)
        if lon_min >= 0:
            print('Converting longitude to positive value to match model')
            lon_final = 360 + lon
        else:
            lon_final = lon
    else:
        lon_final = lon
    
    if np.shape(ds[mdict['lon']].data) == np.shape(ds[mdict['lat']].data):
        D = (ds[mdict['lon']].data-lon_final)**2+(ds[mdict['lat']].data-lat)**2
        [idx_y, idx_x] = np.where(D == np.min(D))
        print(idx_y, idx_x)
    elif np.ndim(ds[mdict['lon']].data) == 1:
        Dy = (ds[mdict['lat']].data-lat)**2
        Dx = (ds[mdict['lon']].data-lon_final)**2
        idx_y = np.where(Dy == np.min(Dy))
        idx_x = np.where(Dx == np.min(Dx))
        print(idx_y, idx_x)
    return idx_y, idx_x
    
def extract_timeseries(ds, idx_y, idx_x):
    if np.ndim(ds) == 3:
        ts = ds[:,idx_y[0],idx_x[0]]
    elif np.ndim(ds) == 4:
        ts = ds[:,:,idx_y[0],idx_x[0]]
    else:
        print('Error: check grid input')
        return
    return ts

def interp_z(ds, loc_z):
    ds_d = ds.interp(depth=loc_z)
    return ds_d

def integ_layer(f, fd, limit_z, years_used, mdict):
    """
    Parameters
    ----------
    f : Full netcdf file
    fd : Netcdf file with variable 'dissic'/'thetao'/etc
    limit_z : Depth limit of integration- all integrals are taken from the surface, 
        so to create a integration within the interior two of these functions will
        need to be called
    years_used : Length of time series desired -- cuts down on computation time
    mdict : Dictionary of specific model used

    Returns
    -------
    fd_int : A (time) x (lat-ish) x (lon-ish) matrix of integrated DIC, with units
        mol C m^-2
    mask : A (lat-ish) x (lon-ish) matrix of 0s and 1s that isolate ocean regions
        with depth >= limit_z
    """
    idx_yr = -1*np.abs(years_used)
    thickness = find_depth_weights(limit_z, f[mdict['lev_bnds']])
    f_weighted = fd[idx_yr:,:,:,:].weighted(thickness)
    f_int = f_weighted.sum(dim=mdict['lev'])
    
    #create mask for depths so we arent concerned about the shelf seas
    idx_mask = np.sum(thickness>0) - 1
    mask = ~np.isnan(fd[0,idx_mask,:,:])

    return f_int, mask

def find_depth_weights(lower_limit, edges):
    thickness = edges[:,1] - edges[:,0]
    
    idx_end = np.sum(np.cumsum(thickness) < lower_limit).data
    partial_depth = lower_limit - np.sum(thickness[:idx_end]).data
    
    thickness[idx_end] = partial_depth
    thickness[idx_end+1:] = 0
    return thickness

def create_lags(ts, lag_num):
    time_length = np.size(ts) #will need to use shorter timeseries because of lag
    output_length = time_length-lag_num   
    out_ts = np.empty(shape=(output_length,lag_num+1))*np.nan  
    out_ts[:,0] = ts[-1*output_length:]
    for i in np.arange(lag_num):
        out_ts[:,i+1] = ts[-1*output_length-(i+1):-1-i]
    return out_ts
        
def multivar_reg(y, x, flag_method):
    #first check that first dimensions of y and x are the same
    if np.shape(y)[0] != np.shape(x)[0]:
        print('ERROR: make sure 1st dimension is time and has same length for input.')
        return
    
    if flag_method == 1:
        print('Using model climatology as background state')
        y_firstguess = np.nanmean(y.fillna(0), axis=0)
        y_anomaly = y.fillna(0) - y_firstguess
        x_anomaly = x - np.nanmean(x, axis=0)
    elif flag_method == 2:
        print('Using model trend as background state')
        x_anomaly = scipy.signal.detrend(x, axis=0)
        y_anomaly = scipy.signal.detrend(y.fillna(-999), axis=0)
        y_firstguess = y - y_anomaly
    
    err = np.zeros(shape=((np.shape(y)[1], np.shape(y)[2])))
    if np.ndim(x)>1:
        coeff_mat = np.zeros(shape=((np.shape(x)[1], np.shape(y)[1], np.shape(y)[2])))
        for i in np.arange(np.shape(y)[2]):
            y_subsamp = y_anomaly[:,:,i]
            dummy = scipy.linalg.lstsq(x_anomaly,y_subsamp)
            coeff_mat[:,:,i] = dummy[0]
            err[:,i] = dummy[1]
    else:
        coeff_mat = np.zeros(shape=((np.shape(y)[1], np.shape(y)[2])))
        x_anomaly_2d = np.expand_dims(x_anomaly, axis=1)
        for i in np.arange(np.shape(y)[2]):
            y_subsamp = y_anomaly[:,:,i]
            dummy = scipy.linalg.lstsq(x_anomaly_2d,y_subsamp)
            coeff_mat[:,i] = dummy[0]
            err[:,i] = dummy[1]
    return coeff_mat, err, y_firstguess

def regression_rmse(fd, ts, coeff_mat, first_guess, flag_method):
    if flag_method == 1:
        ts_anomaly = ts - np.nanmean(ts, axis=0)
    elif flag_method ==2:
        ts_anomaly = scipy.signal.detrend(ts, axis=0)
        
    if np.ndim(ts)>1:
        analysis_step = np.matmul(coeff_mat.T, ts_anomaly.T).T
        predicted = first_guess + analysis_step
        
    mse = np.nanmean((fd - predicted)**2, axis=0)
    rmse = np.sqrt(mse)
    
    return predicted, rmse