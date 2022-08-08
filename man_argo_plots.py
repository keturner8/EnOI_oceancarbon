#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:05:44 2022

@author: keturner

Sections of this script should be run AFTER running argo_reconstruct_method2.py
so that the variables are loaded correctly
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs

argo_coverage= xr.open_dataset('/Users/keturner/Desktop/argo_mask.nc')
argo_mask = argo_coverage.argo_annualcoverage

fig = plt.figure(figsize=(8, 6))
ax0 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) 2002 Argo coverage')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, np.ma.masked_where(land_mask, argo_mask[0,:,:-1]), vmin=0, vmax=1,
                   transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=plt.get_cmap('Purples'),rasterized=True)

ax1 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(b) 2015 Argo coverage')
ax1.coastlines('50m')
im1 = ax1.pcolormesh(lon, lat, np.ma.masked_where(land_mask, argo_mask[-1,:,:-1]), vmin=0, vmax=1,
                   transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=plt.get_cmap('Purples'),rasterized=True)
plt.savefig('/Users/keturner/Desktop/argo_dist.eps', format='eps')

##############################################################################
ens_mask = np.load('/Users/keturner/carbon_timeseries/coeff_matrices/mask_ens.npy')
argo_2015_mask = np.ma.masked_where(ens_mask | (argo_mask[-1,:,1:]==0), argo_mask[-1,:,1:])
argo_2002_mask = np.ma.masked_where(ens_mask | (argo_mask[0,:,1:]==0), argo_mask[0,:,1:])
radii = 5

full = np.arange(30*60)
rmse_argo = np.ones((6,180,359)) * np.nan
rmse_orig = np.ones((6,180,359)) * np.nan
rmse_argo_dt = np.ones((6,180,359)) * np.nan
rmse_orig_dt = np.ones((6,180,359)) * np.nan

for i in np.arange(0,6):

    ri_i = np.arange(i*5*60,(i+1)*5*60)
    ri_ens = np.delete(full,ri_i)
    rmse_orig[i,:,:] = RMSE(dic_resid[ri_i,:,:],0,0)
    rmse_orig_dt[i,:,:] = RMSE(detrend_ens(dic_resid[ri_i,:,:],0,60),0,0)


    for la in np.arange(0+radii,180-1-radii):
        for lo in np.arange(0+radii,360-1-radii):
        
            if ens_mask[la,lo]==0: #check that we have DIC at the location in the first place
                loc_lat = np.arange(la-radii,la+radii+1)
                loc_lon = np.arange(lo-radii,lo+radii+1)
                dic_BATS = dic_resid[ri_ens,la,lo]

                if argo_2015_mask[loc_lat[0]:loc_lat[-1]+1, loc_lon[0]:loc_lon[-1]+1].sum() >=1: #check that we actually have ARGO profiles

                    coef_temp = np.zeros((2*radii+1,2*radii+1))
                    coef_sal = np.zeros((2*radii+1,2*radii+1))

                    sample_indices = np.where(argo_2015_mask[loc_lat[0]:loc_lat[-1]+1, loc_lon[0]:loc_lon[-1]+1]==1)
        
                    theta_subsamp = np.zeros((1500, np.shape(sample_indices)[1]))
                    sal_subsamp = np.zeros((1500, np.shape(sample_indices)[1]))
        
                    for j in np.arange(np.shape(sample_indices)[1]):
                        theta_subsamp[:,j] = theta_resid[ri_ens,loc_lat[sample_indices[0][j]], loc_lon[sample_indices[1][j]]].data
                        sal_subsamp[:,j] = ens_sal[ri_ens,loc_lat[sample_indices[0][j]], loc_lon[sample_indices[1][j]]].data
        
                    input_vars = np.concatenate((theta_subsamp,sal_subsamp,np.expand_dims(pco2_extended[ri_ens],1)),axis=1)
        
                    dummy = scipy.linalg.lstsq(input_vars, dic_BATS)
                    output_var_theta = dummy[0][0:np.shape(sample_indices)[1]]
                    output_var_sal = dummy[0][np.shape(sample_indices)[1]:-1]
                    output_var_pco2 = dummy[0][-1]
        
                    for j in np.arange(np.shape(sample_indices)[1]):
                        coef_temp[sample_indices[0][j],sample_indices[1][j]] = output_var_theta[j]
                        coef_sal[sample_indices[0][j],sample_indices[1][j]] = output_var_sal[j]
                    coef_pco2 = output_var_pco2

                    T_test = theta_resid[ri_i,loc_lat, loc_lon]
                    S_test = ens_sal[ri_i,loc_lat, loc_lon]
                    DIC_truth = dic_resid[ri_i,la,lo]
                    DIC_truth_detrend = detrend_ens(np.expand_dims(DIC_truth, (1,2)), 0, 60)[:,0,0]
                
                    test_argo = np.sum(coef_temp * T_test, axis=(1,2)) + np.sum(coef_sal * S_test, axis=(1,2)) + coef_pco2 * pco2_extended[ri_i]
                    test_argo_detrend = detrend_ens(np.expand_dims(test_argo, (1,2)), 0, 60)[:,0,0] 
                    
                    rmse_argo[i,la,lo] = np.sqrt(np.nanmean((test_argo - DIC_truth.data)**2))
                    rmse_argo_dt[i,la,lo] = RMSE(DIC_truth_detrend, test_argo_detrend,0)
        if np.mod(la,9)==0:
            print(str(i) + ' ' + str(la/90))
            
error_improv_ens0 = (-rmse_argo + rmse_orig) / rmse_orig    
error_improv_ensd0 = (-rmse_argo_dt + rmse_orig_dt) / rmse_orig_dt

fig = plt.figure(figsize=(12, 6))
ax0 = plt.subplot(2, 3, 1, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Ensemble minimum')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, np.min(error_improv_ens0, axis = 0), vmin=-.1, vmax=1,
                   transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=newcmp)

ax1 = plt.subplot(2, 3, 2, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(b) Ensemble average')
ax1.coastlines('50m')
ax1.pcolormesh(lon, lat, np.mean(error_improv_ens0, axis = 0), vmin=-.1, vmax=1, 
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp)

ax2 = plt.subplot(2, 3, 3, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(c) Ensemble maximum')
ax2.coastlines('50m')
ax2.pcolormesh(lon, lat, np.max(error_improv_ens0, axis = 0), vmin=-.1, vmax=1, 
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp)

ax3 = plt.subplot(2, 3, 4, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(d) Ensemble minimum')
ax3.coastlines('50m')
ax3.contourf(lon, lat, np.min(error_improv_ensd0, axis = 0), np.arange(-.1,1.1,.1), extend='both',
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp)

ax4 = plt.subplot(2, 3, 5, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(e) Ensemble average')
ax4.coastlines('50m')
ax4.pcolormesh(lon, lat, np.mean(error_improv_ensd0, axis = 0), vmin=-.1,vmax=1,
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp)

ax5 = plt.subplot(2, 3, 6, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(f) Ensemble maximum')
ax5.coastlines('50m')
im = ax5.pcolormesh(lon, lat, np.max(error_improv_ensd0, axis = 0), vmin=-.1, vmax=1,
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp)

ax_cbar1 = fig.add_axes([0.3, 0.05, .4, 0.05])
fig.colorbar(im, cax=ax_cbar1, orientation="horizontal", pad=0.2)
plt.savefig('/Users/keturner/Desktop/argo5_100_ens_sewn2.png', format='png', dpi=300)

##############################################################################
## Figures using different search radii for the NORESM

f_dic = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/dissic_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
nor_dic = rf.integ_layer(f_dic, f_dic.dissic, limit_z, years_used, mdict)[0]
f_theta = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/thetao_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
nor_theta = rf.integ_layer(f_theta, f_theta.thetao, limit_z, years_used, mdict)[0]
f_so = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/so_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
nor_so = rf.integ_layer(f_so, f_so.so, limit_z, years_used, mdict)[0]
           
dic_nor = nor_dic - np.nanmean(nor_dic, axis=0)
theta_nor = nor_theta - np.nanmean(nor_theta, axis=0)
sal_nor = nor_so - np.nanmean(nor_so, axis=0)

RMSE_NOR = RMSE(dic_nor, 0,0)

RMSE_NOR_dt = RMSE(scipy.signal.detrend(dic_nor,0),0,0)

radii_rmse_argo = np.ones((4,180,359)) * np.nan
radii_rmse_argo_dt = np.ones((4,180,359)) * np.nan

radii = [0,1,2,5]
r = radii[3]
idx = 3

for la in np.arange(0+r,180-1-r):
    for lo in np.arange(0+r,360-1-r):
        
        DIC_truth = dic_nor[:,la,lo]
    
        if ens_mask[la,lo]==0: #check that we have DIC at the location in the first place
            loc_lat = np.arange(la-r,la+r+1)
            loc_lon = np.arange(lo-r,lo+r+1)
            dic_BATS = dic_resid[:,la,lo]

            if argo_2015_mask[loc_lat[0]:loc_lat[-1]+1, loc_lon[0]:loc_lon[-1]+1].data.sum() ==0:
                
                input_vars = np.expand_dims(pco2_extended,1)
                dummy = scipy.linalg.lstsq(input_vars, dic_BATS)
                
                test_argo = dummy[0][0] * pco2_extended[:60]
                        
                radii_rmse_argo[idx,la,lo] = RMSE(test_argo, DIC_truth.data, 0)
                radii_rmse_argo_dt[idx,la,lo] = RMSE(scipy.signal.detrend(DIC_truth.data), scipy.signal.detrend(test_argo),0)
            elif argo_2015_mask[loc_lat[0]:loc_lat[-1]+1, loc_lon[0]:loc_lon[-1]+1].data.sum() >=1: #check that we actually have ARGO profiles

                coef_temp = np.zeros((2*r+1,2*r+1))
                coef_sal = np.zeros((2*r+1,2*r+1))

                sample_indices = np.where(argo_2015_mask[loc_lat[0]:loc_lat[-1]+1, loc_lon[0]:loc_lon[-1]+1]==1)
    
                theta_subsamp = np.zeros((1800, np.shape(sample_indices)[1]))
                sal_subsamp = np.zeros((1800, np.shape(sample_indices)[1]))
    
                for j in np.arange(np.shape(sample_indices)[1]):
                    theta_subsamp[:,j] = theta_resid[:,loc_lat[sample_indices[0][j]], loc_lon[sample_indices[1][j]]].data
                    sal_subsamp[:,j] = ens_sal[:,loc_lat[sample_indices[0][j]], loc_lon[sample_indices[1][j]]].data
    
                input_vars = np.concatenate((theta_subsamp,sal_subsamp,np.expand_dims(pco2_extended,1)),axis=1)
    
                dummy = scipy.linalg.lstsq(input_vars, dic_BATS)
                output_var_theta = dummy[0][0:np.shape(sample_indices)[1]]
                output_var_sal = dummy[0][np.shape(sample_indices)[1]:-1]
                output_var_pco2 = dummy[0][-1]
    
                for j in np.arange(np.shape(sample_indices)[1]):
                    coef_temp[sample_indices[0][j],sample_indices[1][j]] = output_var_theta[j]
                    coef_sal[sample_indices[0][j],sample_indices[1][j]] = output_var_sal[j]
                    coef_pco2 = output_var_pco2

                T_test = theta_nor[:,loc_lat, loc_lon]
                S_test = sal_nor[:,loc_lat, loc_lon]

                test_argo = np.sum(coef_temp * T_test, axis=(1,2)) + np.sum(coef_sal * S_test, axis=(1,2)) + coef_pco2 * pco2_extended[:60]
                        
                radii_rmse_argo[idx,la,lo] = RMSE(test_argo, DIC_truth.data, 0)
                radii_rmse_argo_dt[idx,la,lo] = RMSE(scipy.signal.detrend(DIC_truth.data), scipy.signal.detrend(test_argo),0)
        if (np.mod(la,9)==0) & (lo==150):
            print(str(r) + ' ' + str(la/180))
                               
fig = plt.figure(figsize=(8,12))
ax0 = plt.subplot(4, 2, 1, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, (RMSE_NOR - radii_rmse_argo[0,:,:])/RMSE_NOR,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG'),rasterized=True)

ax1 = plt.subplot(4, 2, 2, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(b) Pointwise observations')
ax1.coastlines('50m')
im1 = ax1.pcolormesh(lon, lat, (RMSE_NOR_dt - radii_rmse_argo_dt[0,:,:])/RMSE_NOR_dt,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG'),rasterized=True)
        
ax0 = plt.subplot(4, 2, 3, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(c) 1$^{\circ}$ observation radius')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, (RMSE_NOR - radii_rmse_argo[1,:,:])/RMSE_NOR,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG'),rasterized=True)

ax1 = plt.subplot(4, 2, 4, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(d) 1$^{\circ}$ observation radius')
ax1.coastlines('50m')
im1 = ax1.pcolormesh(lon, lat, (RMSE_NOR_dt - radii_rmse_argo_dt[1,:,:])/RMSE_NOR_dt,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG'),rasterized=True)

ax0 = plt.subplot(4, 2, 5, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(e) 2$^{\circ}$ observation radius')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, (RMSE_NOR - radii_rmse_argo[2,:,:])/RMSE_NOR,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG'),rasterized=True)

ax1 = plt.subplot(4, 2, 6, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(f) 2$^{\circ}$ observation radius')
ax1.coastlines('50m')
im1 = ax1.pcolormesh(lon, lat, (RMSE_NOR_dt - radii_rmse_argo_dt[2,:,:])/RMSE_NOR_dt,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG'),rasterized=True)

ax0 = plt.subplot(4, 2, 7, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(g) 5$^{\circ}$ observation radius')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, (RMSE_NOR - radii_rmse_argo[3,:,:])/RMSE_NOR,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG'),rasterized=True)

ax1 = plt.subplot(4, 2, 8, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(h) 5$^{\circ}$ observation radius')
ax1.coastlines('50m')
im1 = ax1.pcolormesh(lon, lat, (RMSE_NOR_dt - radii_rmse_argo_dt[3,:,:])/RMSE_NOR_dt,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG'),rasterized=True)

ax_cbar1 = fig.add_axes([0.3, 0.05, .4, 0.05])
fig.colorbar(im1, cax=ax_cbar1, orientation="horizontal", pad=0.2)
plt.savefig('/Users/keturner/Desktop/norradii.png', format='png')

###############average ensemble improvement for different radii

fig = plt.figure(figsize=(8,12))
ax0 = plt.subplot(4, 2, 1, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, np.mean(error_improv_ens0, axis=0),
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG',8),rasterized=True)

ax1 = plt.subplot(4, 2, 2, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax1.coastlines('50m')
im1 = ax1.pcolormesh(lon, lat, np.mean(error_improv_ensd0, axis=0),
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG',8),rasterized=True)

ax2 = plt.subplot(4, 2, 3, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax2.coastlines('50m')
im2 = ax2.pcolormesh(lon, lat, np.mean(error_improv_ens1, axis=0),
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG',8),rasterized=True)

ax3 = plt.subplot(4, 2, 4, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax3.coastlines('50m')
im3 = ax3.pcolormesh(lon, lat, np.mean(error_improv_ensd1, axis=0),
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG',8),rasterized=True)

ax4 = plt.subplot(4, 2, 5, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax4.coastlines('50m')
im4 = ax4.pcolormesh(lon, lat, np.mean(error_improv_ens2, axis=0),
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG',8),rasterized=True)

ax5 = plt.subplot(4, 2, 6, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax5.coastlines('50m')
im5 = ax5.pcolormesh(lon, lat, np.mean(error_improv_ensd2, axis=0),
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG',8),rasterized=True)

ax6 = plt.subplot(4, 2, 7, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax6.coastlines('50m')
im6 = ax6.pcolormesh(lon, lat, np.mean(error_improv_ens5, axis=0),
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG',8),rasterized=True)

ax7 = plt.subplot(4, 2, 8, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Pointwise observations')
ax7.coastlines('50m')
im7 = ax7.pcolormesh(lon, lat, np.mean(error_improv_ensd5, axis=0),
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-1,vmax=1,cmap=plt.get_cmap('PiYG',8),rasterized=True)