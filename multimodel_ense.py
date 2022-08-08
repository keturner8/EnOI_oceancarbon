#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:04:31 2021

@author: keturner
"""
import numpy as np
import scipy.signal
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import xarray as xr
import xesmf as xe
from xesmf.backend import (esmf_grid, esmf_regrid_build,
                           esmf_regrid_apply, esmf_regrid_finalize)
import sys
sys.path.append('/Users/keturner/CODE/carbon_timeseries')
import regression_fcts as rf
import glob

fdir = '/Users/keturner/nc_files/CMIP6/regrid_horiz/'

models =    ['ACCESS', 'CanESM', 'CESM', 'IPSL', 'MPI', 'UKESM']

ncfiles = []
for i in np.arange(np.shape(models)[0]):
    for file in glob.glob(fdir + "thetao_Oyr_" + models[i] + "*.nc"):
        ncfiles.append(file)
ncfiles.sort()

mdict = {'lat':'lat', 'lon':'lon', 'lev':'lev', 'lev_bnds':'lev_bnds'}

# open up pco2 time series - here it is the global average as first cut
years_used = -60
limit_z = 100
m0 = xr.open_dataset('/Users/keturner/nc_files/CMIP6/forcings/mol1.nc', decode_times=False)
pco2 = m0.mole_fraction_of_carbon_dioxide_in_air[years_used:,0]
pco2_extended = np.tile(pco2.data - np.nanmean(pco2.data), np.shape(ncfiles)[0])

def change_var(filelist, oldvar, newvar):
    newfilelist = []
    for i in np.arange(np.shape(ncfiles)[0]):
        file2 = ncfiles[i].replace(oldvar, newvar)
        newfilelist.append(file2)
    return newfilelist

def dic_ENOI_member():
    for i in np.arange(np.shape(ncfiles)[0]):
        f = xr.open_dataset(ncfiles[i])
        fd = f.dissic
        fd_int, mask = rf.integ_layer(f, fd, limit_z, years_used, mdict)
        fd_anom = fd_int - np.nanmean(fd_int, axis=0)
        
        BATS_ts = fd_anom[:,121,295]
        HOT_ts = fd_anom[:,112,201]

        # first take coefficients with regards to pco2
        pco2_coef,_,_ = rf.multivar_reg(fd_anom,pco2, 1)
        # then calculate residuals from pco2 regression
        dic_pco2resid = fd_anom - np.tile(pco2_coef,(np.abs(years_used),1,1))*np.tile(pco2 - pco2.mean(), (359,180,1)).T
        BATS_resid = dic_pco2resid[:,121,295]
        HOT_resid = dic_pco2resid[:,112,201]

        BATS_coef,_,_ = rf.multivar_reg(dic_pco2resid,BATS_resid, 1)
        HOT_coef,_,_ = rf.multivar_reg(dic_pco2resid,HOT_resid, 1)
    
        BATS_rstd = BATS_resid.std(axis=0).data
        HOT_rstd = HOT_resid.std(axis=0).data
        f_rstd = dic_pco2resid.std(axis=0)
    
        BATS_corr_resid = BATS_coef * BATS_rstd/f_rstd
        HOT_corr_resid = HOT_coef * HOT_rstd/f_rstd
        
        output_folder = '/Users/keturner/carbon_timeseries/pseudo_assimilation/TEST_ensemble_pco2/indivs/'
        
        plt.subplots()
        plt.contourf(BATS_corr_resid*mask, np.arange(-1,1.2,.2), cmap=plt.get_cmap('RdBu_r'))
        plt.colorbar()
        plt.title(ncfiles[i][55:-21] + ' BATS correlations')
        plt.savefig(output_folder+'BATS_'+ncfiles[i][55:-21]+'.eps', format='eps')
        plt.close()

        plt.subplots()
        plt.contourf(HOT_corr_resid*mask, np.arange(-1,1.2,.2), cmap=plt.get_cmap('RdBu_r'))
        plt.colorbar()
        plt.title(ncfiles[i][55:-21] + ' HOT correlations')
        plt.savefig(output_folder+'HOT_'+ncfiles[i][55:-21]+'.eps', format='eps')
        plt.close()
        
def create_ensemble(ncfiles, years_used, limit_z, mdict):
    ens_fd = np.empty(shape = (60*np.shape(ncfiles)[0], 180,359))*np.nan
    mask_ensemble = np.ones(shape=(180,359))

    for i in np.arange(np.size(ncfiles)):
    
        f = xr.open_dataset(ncfiles[i])
        fd = f.thetao
        fd_int, mask = rf.integ_layer(f, fd, limit_z, years_used, mdict)
    
        mask_ensemble = mask_ensemble * mask
    
        fd_int2 = fd_int * mask
    
        # here we have to create anomalies (minus climatology)
        # and arrange accordingly
        fd_anom = fd_int2 - np.nanmean(fd_int2, axis=0)
        ens_fd[i*60:(i+1)*60,:,:] = fd_anom
        print((i+1)/np.shape(ncfiles)[0]*100)
        
    return ens_fd, mask_ensemble

def corr_lengthscales():
    ens_fd, _ = create_ensemble(ncfiles, years_used, limit_z, mdict)
    
    ens_d =  xr.DataArray(data=ens_fd)
    # first take coefficients with regards to pco2
    pco2_coef,_,_ = rf.multivar_reg(ens_d,pco2_extended, 1)
    # then calculate residuals from pco2 regression
    var_pco2resid = ens_d - ens_d.mean(axis=0) - np.tile(pco2_coef,(60*np.shape(ncfiles)[0],1,1))*np.tile(pco2_extended - pco2_extended.mean(), (359,180,1)).T
    var_rstd = var_pco2resid.std(axis=0)
    
    idx_x = np.arange(359)
    idx_y = np.arange(180)
    
    LS_zon = np.zeros(shape = np.shape(ens_d)[1:])
    LS_mer = np.zeros(shape = np.shape(ens_d)[1:])
    for i in np.arange(100,120):
        for j in np.arange(270,350):
            #first check that there is data
            if var_rstd[i,j] > 0 and LS_zon[i,j] == 0:
                #calculate global correlations
                var_coef,_,_ = rf.multivar_reg(var_pco2resid,var_pco2resid[:,i,j], 1)
                local_corr = var_coef * var_rstd[i,j].data/var_rstd
                
                #now we try to look at meridional and zonal cross-sections for a first cut
                corr_zonal = local_corr[i,:]
                corr_merid = local_corr[:,j]
                
                idx_zonal = corr_zonal >= 0.8
                idx_merid = corr_merid >= 0.8
                
                zon_x = np.abs(j - idx_x[idx_zonal])
                zon_y = corr_zonal[idx_zonal]
                zon_model = LinearRegression().fit(zon_x.reshape(-1,1), zon_y)
                
                mer_x = np.abs(i - idx_y[idx_merid])
                mer_y = corr_merid[idx_merid]
                mer_model = LinearRegression().fit(mer_x.reshape(-1,1), mer_y)
                
                LS_zon[i,j] = zon_model.coef_
                LS_mer[i,j] = mer_model.coef_       
        print(i)
    

def dic_ENOI():
    ens_fd = np.empty(shape = (60*np.shape(ncfiles)[0], 180,359))*np.nan
    mask_ensemble = np.ones(shape=(180,359))

    for i in np.arange(np.size(ncfiles)):
    
        f = xr.open_dataset(ncfiles[i])
        fd = f.thetao
        fd_int, mask = rf.integ_layer(f, fd, limit_z, years_used, mdict)
    
        mask_ensemble = mask_ensemble * mask
    
        fd_int2 = fd_int * mask
    
        # here we have to create anomalies (minus climatology)
        # and arrange accordingly
        fd_anom = fd_int2 - np.nanmean(fd_int2, axis=0)
        ens_fd[i*60:(i+1)*60,:,:] = fd_anom
        print((i+1)/np.shape(ncfiles)[0]*100)
    
    ens_dic =  xr.DataArray(data=ens_fd)
    ens_std = np.std(ens_fd, axis=0)
    plt.subplots()
    plt.pcolormesh(ens_std*mask_ensemble)
    plt.colorbar()
    plt.title('Model Ensemble standard deviation,\n0-500m integrated (detrended time series)')

    pco2_extended = np.tile(pco2.data - np.nanmean(pco2.data), np.shape(ncfiles)[0])

    BATS_ts = ens_fd[:,121,295]
    HOT_ts = ens_fd[:,112,201]

    # first take coefficients with regards to pco2
    pco2_coef,_,_ = rf.multivar_reg(ens_dic,pco2_extended, 1)
    # then calculate residuals from pco2 regression
    dic_pco2resid = ens_dic - ens_dic.mean(axis=0) - np.tile(pco2_coef,(60*np.shape(ncfiles)[0],1,1))*np.tile(pco2_extended - pco2_extended.mean(), (359,180,1)).T
    BATS_resid = dic_pco2resid[:,121,295]
    HOT_resid = dic_pco2resid[:,112,201]

    BATS_coef,_,_ = rf.multivar_reg(dic_pco2resid,BATS_resid, 1)
    HOT_coef,_,_ = rf.multivar_reg(dic_pco2resid,HOT_resid, 1)
    
    BATS_c2,_,_ = rf.multivar_reg(ens_dic, BATS_ts, 1)
    HOT_c2,_,_ = rf.multivar_reg(ens_dic, HOT_ts, 1)
    
    BATS_std = BATS_ts.std(axis=0).data
    HOT_std = HOT_ts.std(axis=0).data
    BATS_rstd = BATS_resid.std(axis=0).data
    HOT_rstd = HOT_resid.std(axis=0).data
    f_rstd = dic_pco2resid.std(axis=0)
    
    BATS_corr = BATS_c2 * BATS_std/ens_std
    HOT_corr = HOT_c2 * HOT_std/ens_std
    
    BATS_corr_resid = BATS_coef * BATS_rstd/f_rstd
    HOT_corr_resid = HOT_coef * HOT_rstd/f_rstd

    fig, axs = plt.subplots(2,1)
    axs[0].plot(BATS_ts)
    axs[0].scatter(np.arange(0,25*60,60), np.zeros(25), c = 'g')
    axs[0].set_title('BATS DIC anomaly from climatology')

    axs[1].plot(BATS_resid)
    axs[1].scatter(np.arange(0,25*60,60), np.zeros(25), c = 'g')
    axs[1].set_title('BATS DIC minus atmospheric CO2 regression')

    fig, axs = plt.subplots(2,1)
    axs[0].plot(HOT_ts)
    axs[0].scatter(np.arange(0,25*60,60), np.zeros(25), c = 'g')
    axs[0].set_title('HOT DIC anomaly from climatology')

    axs[1].plot(HOT_resid)
    axs[1].scatter(np.arange(0,27*60,60), np.zeros(25), c = 'g')
    axs[1].set_title('HOT DIC minus atmospheric CO2 regression')

    plt.subplots()
    plt.contourf(BATS_corr_resid, np.arange(-1,1.2,.2), cmap=plt.get_cmap('RdBu_r'))
    
    plt.subplots()
    plt.contourf(BATS_corr, np.arange(-1,1.2,.2), cmap=plt.get_cmap('RdBu_r'))

    plt.subplots()
    plt.contourf(HOT_corr_resid, np.arange(-1,1.2,.2), cmap=plt.get_cmap('RdBu_r'))
    
    plt.subplots()
    for i in np.arange(np.size(ncfiles)):
        plt.plot(BATS_ts[i*60:(i+1)*60])
        
    # here we eliminate all trends from the timeseries within the T dataset to test if a linear 
    # approximation would be better
    T_detrend = np.zeros(shape=ens_theta.shape)
    for i in np.arange(0,ens_theta.shape[0],60):
        T_detrend[i:i+60,:,:] = scipy.signal.detrend(ens_theta[i:i+60,:,:], axis=0)
        print(i)
    
    ens_T_detrend = xr.DataArray(data=T_detrend)
    BATS_detrend = ens_T_detrend[:,112,201]
    BATS_coef_detrend,_,_ = rf.multivar_reg(ens_T_detrend, BATS_detrend, 1)
    
    BATS_corr_detrend = BATS_coef_detrend * BATS_detrend.std(axis=0).data/ens_T_detrend.std(axis=0).data
     
    plt.contourf(BATS_corr_detrend, np.arange(-1,1.2,.2), cmap=plt.get_cmap('RdBu_r'))
    plt.title('HOT T correlations, linear trend removed')
    plt.colorbar()
    plt.savefig('/Users/keturner/Desktop/destination_path2.eps', format='eps')
    

def length_scales(dic_pco2resid):
    dx = np.ones(dic_pco2resid.shape[1])
    dy = np.ones(dic_pco2resid.shape[2])
    for i in dic_pco2resid.shape[1]:
        for j in dic_pco2resid.shape[2]:
            #pull out time series
            dummy_ts = dic_pco2resid[:,i,j]
            if dummy_ts[0].data != 0: 
                dummy_sd = dummy_ts.std(axis=0).data
                dummy_coef,_,_ = rf.multivar_reg(dic_pco2resid,dummy_ts, 1)
                corr_zonal = dummy_coef[i,:]
                corr_meridional = dummy_coef[:,j]
                
                plt.plot(corr_zonal[1:] - corr_zonal[:-1])
                
def cross_corr():
    ens_dic =  xr.DataArray(data=ens_dic_full_100)
    ens_theta =  xr.DataArray(data=ens_thetao_full_100)
    ens_sal = xr.DataArray(data=ens_so_full_100)
    # first take coefficients with regards to pco2
    pco2_coef_T,_,_ = rf.multivar_reg(ens_theta,pco2_extended, 1)
    pco2_coef_C,_,_ = rf.multivar_reg(ens_dic,pco2_extended, 1)
    # then calculate residuals from pco2 regression
    dic_resid = ens_dic - ens_dic.mean(axis=0) - np.tile(pco2_coef_C,(60*np.shape(ncfiles)[0],1,1))*np.tile(pco2_extended - pco2_extended.mean(), (359,180,1)).T
    theta_resid = ens_theta - ens_theta.mean(axis=0) - np.tile(pco2_coef_T,(60*np.shape(ncfiles)[0],1,1))*np.tile(pco2_extended - pco2_extended.mean(), (359,180,1)).T
    
    BATS_resid = dic_resid[:,121,295]
    HOT_resid = dic_resid[:,112,201]
    
    dic_rstd = dic_resid.std(axis=0)
    theta_rstd = theta_resid.std(axis=0)
    sal_rstd = ens_sal.std(axis=0)
    
    BATS_rstd_dic = BATS_resid.std(axis=0).data
    BATS_xcoef,_,_ = rf.multivar_reg(theta_resid,BATS_resid, 1) 
    BATS_xcorr = BATS_xcoef * BATS_rstd_dic/theta_rstd
    
    HOT_rstd_dic = HOT_resid.std(axis=0).data
    HOT_xcoef,_,_ = rf.multivar_reg(theta_resid,HOT_resid, 1) 
    HOT_xcorr = HOT_xcoef * HOT_rstd_dic/theta_rstd

    plt.subplots()
    plt.contourf(BATS_xcorr, np.arange(-.5,.55,.05), cmap=plt.get_cmap('RdBu_r'))
    plt.title('T cross-correlations with BATS DIC')
    plt.colorbar()
    plt.savefig('/Users/keturner/Desktop/BATS_xcorr_100.eps', format='eps')
    
    plt.subplots()
    plt.contourf(HOT_xcorr, np.arange(-.5,.55,.05), cmap=plt.get_cmap('RdBu_r'))
    plt.title('T cross-correlations with HOT DIC')
    plt.colorbar()
    plt.savefig('/Users/keturner/Desktop/HOT_xcorr_100.eps', format='eps')
    
    dummy_CxT = np.zeros(shape=ens_dic.shape[1:])
    dummy_TxS = np.zeros(shape=ens_dic.shape[1:])
    dummy_CxS = np.zeros(shape=ens_dic.shape[1:])
    for i in np.arange(ens_dic.shape[1]):
        for j in np.arange(ens_dic.shape[2]):
            if dic_rstd[i,j] == 0:
                dummy_CxT[i,j] = np.nan
                dummy_TxS[i,j] = np.nan
                dummy_CxS[i,j] = np.nan
            if dic_rstd[i,j] > 0:
                dummy_CxT[i,j] = scipy.stats.pearsonr(dic_resid[:,i,j], theta_resid[:,i,j])[0]
                dummy_CxS[i,j] = scipy.stats.pearsonr(dic_resid[:,i,j], ens_sal[:,i,j])[0]
                dummy_TxS[i,j] = scipy.stats.pearsonr(theta_resid[:,i,j], ens_sal[:,i,j])[0]
        print(i/180*100)
    
    plt.subplots()
    plt.contourf(dummy_TxS, np.arange(-1,1.2,.2), cmap=plt.get_cmap('RdBu_r'))
    plt.colorbar()
    plt.title('Point-wise cross-correlations of residual T and salinity')
    plt.savefig('/Users/keturner/Desktop/ptwise_TxcorrS_100.eps', format='eps')
    
    fig, axs = plt.subplots(1,3)
    axs[0].hist(theta_resid[:,121,295])
    axs[0].set_title('T residuals')
    axs[1].hist(dic_resid[:,121,295])
    axs[1].set_title('DIC residuals')
    axs[2].hist(ens_sal[:,121,295])
    axs[2].set_title('S residuals')
    fig.tight_layout()
    plt.savefig('/Users/keturner/Desktop/BATS_resids.eps', format='eps')

ens_dic =  xr.DataArray(data=dissic_100_SORTED)
ens_theta =  xr.DataArray(data=thetao_100_SORTED)
ens_sal = xr.DataArray(data=so_100_SORTED)
# first take coefficients with regards to pco2
pco2_coef_T,_,_ = rf.multivar_reg(ens_theta,pco2_extended, 1)
pco2_coef_C,_,_ = rf.multivar_reg(ens_dic,pco2_extended, 1)
# then calculate residuals from pco2 regression
dic_resid = ens_dic - ens_dic.mean(axis=0) - np.tile(pco2_coef_C,(60*np.shape(ncfiles)[0],1,1))*np.tile(pco2_extended - pco2_extended.mean(), (359,180,1)).T
theta_resid = ens_theta - ens_theta.mean(axis=0) - np.tile(pco2_coef_T,(60*np.shape(ncfiles)[0],1,1))*np.tile(pco2_extended - pco2_extended.mean(), (359,180,1)).T

i = 4
ri = np.arange(i*5*60,(i+1)*5*60)
full = np.arange(30*60)

test_index = np.delete(full,ri)
coeff_UKESM = dic_MLR(dic_resid[test_index,:,:], theta_resid[test_index,:,:], ens_sal[test_index,:,:])

MPI_reconstruct = coeff_UKESM[0,:,:] * theta_resid[ri[0]:ri[0]+300,:,:] + coeff_UKESM[1,:,:] * (ens_sal[ri[0]:ri[0]+300,:,:] - np.nanmean(ens_sal[ri[0]:ri[0]+300,:,:])) 
rmse_MPI_fg = np.sqrt(np.nanmean((dic_resid[ri[0]:ri[0]+300,:,:].data)**2, axis=0))
rmse_MPI = np.sqrt(np.nanmean((dic_resid[ri[0]:ri[0]+300,:,:].data - MPI_reconstruct.data)**2, axis=0))

plt.subplots()
plt.plot(UKESM_reconstruct[:,121,295])
plt.plot(dic_resid[ri[0]:ri[0]+300, 121,295])

test = (rmse_ACCESS_fg - rmse_ACCESS)/rmse_ACCESS_fg + (rmse_CanESM_fg - rmse_CanESM)/rmse_CanESM_fg + \
    (rmse_CESM_fg - rmse_CESM)/rmse_CESM_fg + (rmse_IPSL_fg - rmse_IPSL)/rmse_IPSL_fg + \
        (rmse_MPI_fg - rmse_MPI)/rmse_MPI_fg + (rmse_UKESM_fg - rmse_UKESM)/rmse_UKESM_fg

plt.contourf((rmse_UKESM_fg - rmse_UKESM)/rmse_UKESM_fg, np.arange(-1,1.2, .2), cmap=plt.get_cmap('RdBu_r'))

plt.contourf(test/6, np.arange(-1,1.2,.1), cmap =plt.get_cmap('RdBu_r'))
plt.colorbar()
plt.title('Average error %age change from MLR')
plt.savefig('/Users/keturner/Desktop/avg_rmse_improv.eps', format='eps')

fig,axs = plt.subplots(1,3, figsize=(16,6))
fig0 = axs[0].contourf(rmse_UKESM_fg, np.arange(0,5.25,.25))
plt.colorbar(fig0, ax=axs[0])
fig1 = axs[1].contourf(rmse_UKESM, np.arange(0,5.25,.25))
plt.colorbar(fig1, ax=axs[1])
fig2 = axs[2].contourf((rmse_CanESM_fg - rmse_CanESM)/rmse_CanESM_fg, np.arange(-1,1.2, .2), cmap=plt.get_cmap('RdBu_r'))
plt.colorbar(fig2, ax=axs[2])
plt.tight_layout()
plt.savefig('/Users/keturner/Desktop/ACCESS_rmse.eps', format='eps')

def dic_MLR(DIC, T, S):
    output_var = np.zeros(shape = (2, 180, 359))
    for i in np.arange(DIC.shape[1]):
        for j in np.arange(DIC.shape[2]):
            if DIC[:,i,j].data[-1] == 0:
                output_var[:,i,j] = [np.nan, np.nan]

            if DIC[:,i,j].data[-1] != 0:
                dic_subsamp = DIC[60:,i,j]
                theta_subsamp = T[60:,i,j]
                sal_subsamp = S[60:,i,j]
        
                input_vars = [theta_subsamp, sal_subsamp]
                dummy = scipy.linalg.lstsq(np.transpose(input_vars), dic_subsamp)
                output_var[:,i,j] = dummy[0]
                
    return output_var
                
def Nor_test(mrl_coef, loc1, loc2, depth, years_used, mdict):
        f_dic = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/dissic_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
        nor_dic = rf.integ_layer(f_dic, f_dic.dissic, depth, years_used, mdict)[0]
        f_theta = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/thetao_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
        nor_theta = rf.integ_layer(f_theta, f_theta.thetao, depth, years_used, mdict)[0]
        f_so = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/so_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
        nor_so = rf.integ_layer(f_so, f_so.so, depth, years_used, mdict)[0]
        
        pco2_coef_dic,_,_ = rf.multivar_reg(nor_dic - np.nanmean(nor_dic, axis=0),pco2 - np.nanmean(pco2), 1)
        pco2_coef_theta,_,_ = rf.multivar_reg(nor_theta - np.nanmean(nor_theta, axis=0),pco2 - np.nanmean(pco2), 1)
        
        # then calculate residuals from pco2 regression
        nor_dicresid = nor_dic - np.nanmean(nor_dic, axis=0) - np.tile(pco2_coef_dic,(np.abs(years_used),1,1))*np.tile(pco2 - pco2.mean(), (359,180,1)).T
        nor_thetaresid = nor_theta - np.nanmean(nor_theta, axis=0) - np.tile(pco2_coef_theta,(np.abs(years_used),1,1))*np.tile(pco2 - pco2.mean(), (359,180,1)).T
        
        plt.plot(dic_resid[:60, loc1,loc2])
        plt.plot(nor_dicresid[:, loc1,loc2])
        
        plt.plot(theta_resid[:60,loc1,loc2])
        plt.plot(nor_thetaresid[:,loc1,loc2])
        
        rmse_resid = np.sqrt(np.nanmean(nor_dicresid.data**2, axis=0))
        
        rmse = np.zeros(nor_thetaresid.shape[1:])
        for i in np.arange(nor_thetaresid.shape[1]):
            for j in np.arange(nor_thetaresid.shape[2]):
        
                nor_reconstruct = mlr_coef[0,i,j] * nor_thetaresid[:,i,j] + mlr_coef[1,i,j] * (nor_so[:,i,j] - np.nanmean(nor_so[:,i,j]))
                rmse[i,j] = np.sqrt(np.nanmean((nor_dicresid[:,i,j].data - nor_reconstruct.data)**2, axis=0))
            print(i)
            
def argo_test(loc1, loc2, radius, depth, years_used, mdict):
        f_dic = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/dissic_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
        nor_dic = rf.integ_layer(f_dic, f_dic.dissic, depth, years_used, mdict)[0]
        f_theta = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/thetao_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
        nor_theta = rf.integ_layer(f_theta, f_theta.thetao, depth, years_used, mdict)[0]
        f_so = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/so_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
        nor_so = rf.integ_layer(f_so, f_so.so, depth, years_used, mdict)[0]
        
        pco2_coef_dic,_,_ = rf.multivar_reg(nor_dic - np.nanmean(nor_dic, axis=0),pco2 - np.nanmean(pco2), 1)
        pco2_coef_theta,_,_ = rf.multivar_reg(nor_theta - np.nanmean(nor_theta, axis=0),pco2 - np.nanmean(pco2), 1)
        
        # then calculate residuals from pco2 regression
        nor_dicresid = nor_dic - np.nanmean(nor_dic, axis=0) - np.tile(pco2_coef_dic,(np.abs(years_used),1,1))*np.tile(pco2 - pco2.mean(), (359,180,1)).T
        nor_thetaresid = nor_theta - np.nanmean(nor_theta, axis=0) - np.tile(pco2_coef_theta,(np.abs(years_used),1,1))*np.tile(pco2 - pco2.mean(), (359,180,1)).T
        
        true_dicresid = nor_dicresid[:60, loc1,loc2]
        true_thetaresid = nor_thetaresid[:60, loc1,loc2]
        
        avg_thetaresid = np.nanmean(nor_thetaresid[:60,loc1-radius:loc1+radius,
                                                   loc2-radius:loc2+radius])
    

fig, axs = plt.subplots(2)
i=121
j=295
axs[0].plot(nor_dicresid[:,i,j], label='Model DIC')
axs[0].plot(mlr_coef[0,i,j] * nor_thetaresid[:,i,j] + mlr_coef[1,i,j] * (nor_so[:,i,j] - np.nanmean(nor_so[:,i,j])), label='Reconstructed DIC')
axs[0].set_title('BATS model output and reconstruction, mol m$^{-2}$')
axs[0].legend()

i=112
j=201
axs[1].plot(nor_dicresid[:,i,j])
axs[1].plot(mlr_coef[0,i,j] * nor_thetaresid[:,i,j] + mlr_coef[1,i,j] * (nor_so[:,i,j] - np.nanmean(nor_so[:,i,j])))
axs[1].set_title('HOT model output and reconstruction, mol m$^{-2}$')
plt.tight_layout()
plt.savefig('/Users/keturner/Desktop/Nortest_ts.eps', format='eps')

fig, axs = plt.subplots(1,2)
fig1 = axs[0].contourf(rmse_resid, np.arange(0,2.75,.25))
axs[0].set_title('RMSE of pCO$_2$ LR only')
fig2 = axs[1].contourf(rmse, np.arange(0,2.75,.25))
axs[1].set_title('RMSE of pCO$_2$ LR and T/S MLR')
plt.colorbar(fig1, ax=axs[0])
plt.colorbar(fig2, ax=axs[1])
plt.tight_layout()
plt.savefig('/Users/keturner/Desktop/Nortest_rmse.eps', format='eps')

plt.pcolormesh((rmse_resid-rmse)/rmse_resid, vmin = -1, vmax = 1 , cmap=plt.get_cmap('RdBu_r'))
plt.colorbar()
plt.title('Fractional error reduction using T, S MLR on NorESM output')
plt.savefig('/Users/keturner/Desktop/Nortest_relimprov.eps', format='eps')

plt.plot(nor_dicresid[:,20,200])
plt.plot(mlr_coef[0,20,200] * nor_thetaresid[:,20,200] + mlr_coef[1,20,200] * (nor_so[:,20,200] - np.nanmean(nor_so[:,20,200])))

plt.pcolormesh(output_var[0,:,:], vmin=-.05, vmax=.05, cmap=plt.get_cmap('RdBu_r'))
plt.colorbar()

plt.pcolormesh(output_var[1,:,:], vmin=-.12, vmax=.12)
plt.colorbar()

plt.plot(dic_resid[:60,121,295], label='DIC residual')
plt.plot(output_var[0,121,295] * theta_resid[:60,121,295] + output_var[1,121,295] * ens_sal[:60,121,295], label='Reconstruction from ensemble T and S regression')
plt.legend()
plt.savefig('/Users/keturner/Desktop/BATS_test_TS.eps', format='eps')

    