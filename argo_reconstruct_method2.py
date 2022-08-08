#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:58:06 2022

@author: keturner
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap

import sys
sys.path.append('/Users/keturner/CODE/carbon_timeseries')
import regression_fcts as rf
import cartopy.crs as ccrs


dissic_SORTED = np.load('/Users/keturner/carbon_timeseries/sorted_model_data/dissic_100m.npy')
thetao_SORTED = np.load('/Users/keturner/carbon_timeseries/sorted_model_data/thetao_100m.npy')
so_SORTED = np.load('/Users/keturner/carbon_timeseries/sorted_model_data/so_100m.npy')
argo_coverage= xr.open_dataset('/Users/keturner/Desktop/argo_mask.nc')
argo_mask = argo_coverage.argo_annualcoverage

years_used = -60
limit_z = 100
m0 = xr.open_dataset('/Users/keturner/nc_files/CMIP6/forcings/mol1.nc', decode_times=False)
pco2 = m0.mole_fraction_of_carbon_dioxide_in_air[years_used:,0]
pco2_extended = np.tile(pco2.data - np.nanmean(pco2.data), 
                        int(np.shape(so_SORTED)[0]/np.abs(years_used)))

lon = np.arange(0.5,359.5)
lat = np.arange(-89.5,90.5)

mdict = {'lat':'lat', 'lon':'lon', 'lev':'lev', 'lev_bnds':'lev_bnds'}

ens_dic =  xr.DataArray(data=dissic_SORTED)
ens_theta =  xr.DataArray(data=thetao_SORTED)
ens_sal = xr.DataArray(data=so_SORTED)

# Creating masks to plot figures
test1 = np.isnan(np.mean(dissic_SORTED/dissic_SORTED, axis=0))
test2 = np.mean(dissic_SORTED, axis=0) != 0
mask_tot = test1 & test2 * 1

mask_tot[21:103,72:74] = mask_tot[21:103,72:74]*2
mask_tot[103:105,72] = mask_tot[103:105,72]*2
mask_tot[176:,:] = mask_tot[176:,:]*2
mask_tot[172:176,70:265] = mask_tot[172:176,70:265]*2
# plt.pcolormesh(mask_tot)
# plt.colorbar()


ens_mask = np.load('/Users/keturner/carbon_timeseries/coeff_matrices/mask_ens.npy')

interp_mask = ens_mask & (mask_tot==2)
land_mask = ens_mask & (mask_tot!=2)

spec = cm.get_cmap('YlGnBu', 10)
newcolors = np.zeros((11,4))
newcolors[1:, :] = spec(np.linspace(0, 1, 10))
newcolors[:1, :] = colors.to_rgba('indianred')
newcmp = ListedColormap(newcolors)

             
def RMSE(var1, var2, ax):
    rmse = np.sqrt(np.nanmean((var1-var2)**2,axis=ax))
    return rmse

def sew_seams(input_mat, ens_mask): #do not use a masked array for this function
    interp_out = np.zeros(np.shape(input_mat))
    input_dummy = input_mat * ens_mask
    input_dummy2 = input_mat * (1-ens_mask)
    
    for i in np.arange(180):
        if (np.sum(input_dummy2[i,:]==0)!=0) & (np.sum(input_dummy2[i,:]==0) < 359):    # checking to see if we have something to interpolate across
            idx_interp = np.arange(359)[input_dummy2[i,:]==0] 
            idx_data = np.arange(359)[input_dummy2[i,:]!=0]
            interp_out[i,idx_interp] = np.interp(idx_interp, idx_data, input_dummy2[i,idx_data], period=360)
            interp_out[i,idx_data] = input_mat[i,idx_data]
    return interp_out

def detrend_ens(input_ts, idx_t, tlength):
    tot_tlength = np.size(input_ts, idx_t)
    output_ts = np.ones(np.shape(input_ts))*np.nan
    if idx_t == 0:
        for i in np.arange(np.int(tot_tlength/tlength)):
            sample = input_ts[i*tlength:(i+1)*tlength, :,:]
            output_ts[i*tlength:(i+1)*tlength, :,:] = scipy.signal.detrend(sample, idx_t)
    return output_ts
       

####################################################################################
## 23 March - testing coefficients for residuals without original pco2 regression
dic_resid = ens_dic - ens_dic.mean(axis=0)
theta_resid = ens_theta - ens_theta.mean(axis=0)
so_resid = ens_sal - ens_sal.mean(axis=0)

coeff_ensCO2 = np.zeros([180,359])
coeff_ensT = np.zeros([180,359])
coeff_ensS = np.zeros([180,359])
for i in np.arange(0,180):
        for j in np.arange(0,359):
            if theta_resid[0,i,j] != 0 :
                input_vars = np.stack((pco2_extended, theta_resid[:,i,j],ens_sal[:,i,j]), axis=-1)
                dummy = np.linalg.lstsq(input_vars, dic_resid[:,i,j], rcond=-1)
                
                coeff_ensCO2[i,j] = dummy[0][0]
                coeff_ensT[i,j] = dummy[0][1]
                coeff_ensS[i,j] = dummy[0][2]
        print(i)

cCO2 = np.ma.masked_where(land_mask,sew_seams(coeff_ensCO2, ens_mask))
cT = np.ma.masked_where(land_mask,sew_seams(coeff_ensT, ens_mask))
cS = np.ma.masked_where(land_mask,sew_seams(coeff_ensS, ens_mask))
####
####
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 3, 1, projection=ccrs.Robinson(central_longitude=180))
plt.title('(a) pCO$_2$ coefficients')
ax1.coastlines('50m')
im1 = ax1.contourf(lon, lat, cCO2, np.arange(-.08,.09,.01),extend='both', transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))
plt.colorbar(im1,ax=ax1, fraction=0.04, pad=0.04, orientation="horizontal")

ax2 = plt.subplot(1, 3, 2, projection=ccrs.Robinson(central_longitude=180))
plt.title('(b) T coefficients')
ax2.coastlines('50m')
im2 = ax2.contourf(lon, lat, cT, np.arange(-.04,.05,.01),extend='both', transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))
plt.colorbar(im2,ax=ax2, fraction=0.04, pad=0.04, orientation="horizontal")

ax3 = plt.subplot(1, 3, 3, projection=ccrs.Robinson(central_longitude=180))
plt.title('(c) S coefficients')
ax3.coastlines('50m')
im3 = ax3.contourf(lon, lat, cS, np.arange(-.2,.24,.04),extend='both', transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))
plt.colorbar(im3,ax=ax3, fraction=0.04, pad=0.04, orientation="horizontal")
plt.tight_layout()
plt.savefig('/Users/keturner/Desktop/ens_coef_100_sewn.eps', format='eps')
####
####
####################################################################################
## 25 March - correlations between variables

corr_dicp_ens = np.zeros((180,359))
corr_tp_ens = np.zeros((180,359))
corr_ps_ens = np.zeros((180,359))
corr_dict_ens = np.zeros((180,359))
corr_ts_ens = np.zeros((180,359))
corr_dics_ens = np.zeros((180,359))
R2 = np.zeros((180,359))
for ii in np.arange(120,180):
        for j in np.arange(0,359):
            if dic_resid[1000,ii,j] != 0 :
                corr_dicp_ens[ii,j] = np.corrcoef(dic_resid[:,ii,j], pco2_extended)[0,1]
                corr_tp_ens[ii,j] = np.corrcoef(theta_resid[:,ii,j], pco2_extended)[0,1]
                corr_ps_ens[ii,j] = np.corrcoef(ens_sal[:,ii,j], pco2_extended)[0,1]
                corr_dict_ens[ii,j] = np.corrcoef(dic_resid[:,ii,j], theta_resid[:,ii,j])[0,1]
                corr_ts_ens[ii,j] = np.corrcoef(ens_sal[:,ii,j], theta_resid[:,ii,j])[0,1]
                corr_dics_ens[ii,j] = np.corrcoef(ens_sal[:,ii,j], dic_resid[:,ii,j])[0,1]
                
                R = np.array([[1, corr_tp_ens[ii,j], corr_ps_ens[ii,j]],
                     [corr_tp_ens[ii,j], 1, corr_ts_ens[ii,j]],
                     [corr_ps_ens[ii,j], corr_ts_ens[ii,j], 1]])
                c = np.array([corr_dicp_ens[ii,j], corr_dict_ens[ii,j], corr_dics_ens[ii,j]])
                R2[ii,j] = np.matmul(c, np.matmul(scipy.linalg.inv(R),np.transpose(c)))
        print(ii)
        
dicp1 = np.ma.masked_where(land_mask,sew_seams(corr_dicp_ens, ens_mask))
tp1 = np.ma.masked_where(land_mask,sew_seams(corr_tp_ens, ens_mask))
sp1 = np.ma.masked_where(land_mask,sew_seams(corr_ps_ens, ens_mask))
dict1 = np.ma.masked_where(land_mask,sew_seams(corr_dict_ens, ens_mask))
ts1 = np.ma.masked_where(land_mask,sew_seams(corr_ts_ens, ens_mask))
dics1 = np.ma.masked_where(land_mask,sew_seams(corr_dics_ens, ens_mask))

####
####
fig = plt.figure(figsize=(15, 6))
ax1 = plt.subplot(2, 3, 1, projection=ccrs.Robinson(central_longitude=180))
plt.title('(a) pCO$_2$/DIC')
ax1.coastlines('50m')
ax1.contourf(lon, lat, dicp1, np.arange(-1,1.2,.2), transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))

ax2 = plt.subplot(2, 3, 2, projection=ccrs.Robinson(central_longitude=180))
plt.title('(b) pCO$_2$/temperature')
ax2.coastlines('50m')
ax2.contourf(lon, lat, tp1, np.arange(-1,1.2,.2), transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))

ax3 = plt.subplot(2, 3, 3, projection=ccrs.Robinson(central_longitude=180))
plt.title('(c) pCO$_2$/salinity')
ax3.coastlines('50m')
ax3.contourf(lon, lat, sp1, np.arange(-1,1.2,.2), transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))

ax4 = plt.subplot(2, 3, 4, projection=ccrs.Robinson(central_longitude=180))
plt.title('(d) temperature/DIC')
ax4.coastlines('50m')
ax4.contourf(lon, lat, dict1, np.arange(-1,1.2,.2), transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))

ax5 = plt.subplot(2, 3, 5, projection=ccrs.Robinson(central_longitude=180))
plt.title('(e) salinity/DIC')
ax5.coastlines('50m')
ax5.contourf(lon, lat, dics1, np.arange(-1,1.2,.2), transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))

ax6 = plt.subplot(2, 3, 6, projection=ccrs.Robinson(central_longitude=180))
plt.title('(f) temperature/salinity')
ax6.coastlines('50m')
im = ax6.contourf(lon, lat, ts1, np.arange(-1,1.2,.2), transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))

ax_cbar1 = fig.add_axes([0.3, 0.05, .4, 0.05])
fig.colorbar(im, cax=ax_cbar1, orientation="horizontal", pad=0.2)
plt.savefig('/Users/keturner/Desktop/ens_corr_100_500_sewn.eps', format='eps')
####
####

################################################################################
## 28 March, covariance breakdown between trends and detrended variables (DIC, T, S)

dic_resid = ens_dic - ens_dic.mean(axis=0)
theta_resid = ens_theta - ens_theta.mean(axis=0)
so_resid = ens_sal - ens_sal.mean(axis=0)

pco2_coef_T,_,_ = rf.multivar_reg(theta_resid,pco2_extended, 1)
pco2_coef_C,_,_ = rf.multivar_reg(dic_resid,pco2_extended, 1)
pco2_coef_S,_,_ = rf.multivar_reg(so_resid,pco2_extended, 1)

    fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=((12.5,4.5)))
    f0 = axs[0].pcolormesh(pco2_coef_T, vmin=-2, vmax=2, cmap="RdBu_r")
    axs[0].set_title(r'(a) T regression coefficient $\alpha$', loc='left')
    plt.colorbar(f0,ax=axs[0])
    
    f1 = axs[1].pcolormesh(pco2_coef_S, vmin=-.3, vmax=.3, cmap="RdBu_r")
    axs[1].set_title(r'(b) S regression coefficient $\beta$', loc='left')
    plt.colorbar(f1, ax=axs[1])
    
    f2 = axs[2].pcolormesh(pco2_coef_C, vmin=-.07, vmax=.07, cmap="RdBu_r")
    axs[2].set_title(r'(c) DIC regression coefficient $\gamma$', loc='left')
    plt.colorbar(f2, ax=axs[2])
    plt.savefig('/Users/keturner/Desktop/pco2_coef.eps', format='eps')

corr_dict_ens = np.zeros((180,359))
corr_ts_ens = np.zeros((180,359))
corr_dics_ens = np.zeros((180,359))
for ii in np.arange(0,180):
        for j in np.arange(0,359):
            if dic_resid[1000,ii,j] != 0 :
                corr_dict_ens[ii,j] = np.corrcoef(dic_resid[:,ii,j], theta_resid[:,ii,j])[0,1]
                corr_ts_ens[ii,j] = np.corrcoef(so_resid[:,ii,j], theta_resid[:,ii,j])[0,1]
                corr_dics_ens[ii,j] = np.corrcoef(so_resid[:,ii,j], dic_resid[:,ii,j])[0,1]
        print(ii)
        
std_dic = np.std(dic_resid, axis=0)
std_theta = np.std(theta_resid, axis=0)
std_so = np.std(so_resid, axis=0)
var_pco2 = np.std(pco2_extended)**2

cov_ts_anom = np.zeros((180,359))
cov_dics_anom = np.zeros((180,359))
cov_dict_anom = np.zeros((180,359))
for ii in np.arange(0,180):
        for j in np.arange(0,359):
            if ens_dic[1000,ii,j].data != 0 :
                t_dd = theta_resid[:,ii,j] - pco2_coef_T[ii,j] * (pco2_extended - np.nanmean(pco2))
                s_dd = so_resid[:,ii,j] - pco2_coef_S[ii,j] * (pco2_extended - np.nanmean(pco2))
                dic_dd = dic_resid[:,ii,j] - pco2_coef_C[ii,j] * (pco2_extended - np.nanmean(pco2))
                cov_ts_anom[ii,j] = np.cov(t_dd, s_dd)[0,1]
                cov_dics_anom[ii,j] = np.cov(s_dd, dic_dd)[0,1]
                cov_dict_anom[ii,j] = np.cov(dic_dd, t_dd)[0,1]
                

    fig, axs = plt.subplots(3,4, sharex=True, sharey=True, figsize=((13.5,9.5)))                
    f0 = axs[0,0].contourf(np.ma.masked_where(land_mask,sew_seams(corr_ts_ens, ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[0,0].set_title(r"(a)  $\rho$(T',S')", loc="left")
    f1 = axs[1,0].contourf(np.ma.masked_where(land_mask,sew_seams(corr_dict_ens, ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[1,0].set_title(r"(e)  $\rho$(T',DIC')", loc="left")
    f2 = axs[2,0].contourf(np.ma.masked_where(land_mask,sew_seams(corr_dics_ens, ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[2,0].set_title(r"(i)  $\rho$(S',DIC')", loc="left")
    
    f0a = axs[0,1].contourf(np.ma.masked_where(land_mask,sew_seams(pco2_coef_T*pco2_coef_S*var_pco2/(std_theta*std_so), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[0,1].set_title(r"(b) pCO$_2$ term, T' and S'", loc="left")
    f1a = axs[1,1].contourf(np.ma.masked_where(land_mask,sew_seams(pco2_coef_T*pco2_coef_C*var_pco2/(std_theta*std_dic), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[1,1].set_title(r"(f) pCO$_2$ term, T' and DIC'", loc="left")
    f2a = axs[2,1].contourf(np.ma.masked_where(land_mask,sew_seams(pco2_coef_S*pco2_coef_C*var_pco2/(std_so*std_dic), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[2,1].set_title(r"(j) pCO$_2$ term, S' and DIC'", loc="left")
    
    f0b = axs[0,2].contourf(np.ma.masked_where(land_mask,sew_seams(cov_ts_anom/(std_theta*std_so), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[0,2].set_title(r"(c) non-pCO$_2$ term, T' and S'", loc="left")
    f1b = axs[1,2].contourf(np.ma.masked_where(land_mask,sew_seams(cov_dict_anom/(std_theta*std_dic), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[1,2].set_title(r"(g) non-pCO$_2$ term, T' and DIC'", loc="left")
    f2b = axs[2,2].contourf(np.ma.masked_where(land_mask,sew_seams(cov_dics_anom/(std_dic*std_so), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[2,2].set_title(r"(k) non-pCO$_2$ term, S' and DIC'", loc="left")
    
    f0test = axs[0,3].contourf(np.ma.masked_where(land_mask,sew_seams(corr_ts_ens - (pco2_coef_T*pco2_coef_S*var_pco2 + cov_ts_anom)/(std_theta*std_so), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[0,3].set_title(r"(d) approximation error", loc="left")
    f1test = axs[1,3].contourf(np.ma.masked_where(land_mask,sew_seams(corr_dict_ens - (pco2_coef_T*pco2_coef_C*var_pco2 + cov_dict_anom)/(std_theta*std_dic), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[1,3].set_title(r"(h) approximation error", loc="left")
    f2test = axs[2,3].contourf(np.ma.masked_where(land_mask,sew_seams(corr_dics_ens - (pco2_coef_C*pco2_coef_S*var_pco2 + cov_dics_anom)/(std_dic*std_so), ens_mask)),
                             np.arange(-1,1.2,.2), cmap="RdBu_r")
    axs[2,3].set_title(r"(l) approximation error", loc="left")
    plt.savefig('/Users/keturner/Desktop/corr_breakdown.eps', format='eps')


dict_pco2 = np.ma.masked_where(land_mask,sew_seams(pco2_coef_T*pco2_coef_C*var_pco2/(std_theta*std_dic), ens_mask))
dict_anom = np.ma.masked_where(land_mask,sew_seams(cov_dict_anom/(std_theta*std_dic), ens_mask))

    fig = plt.figure(figsize=(10, 3.7))
    ax5 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=180))
    plt.title(r"(a) $\rho(T',DIC')$, pCO$_2$ term")
    ax5.coastlines('50m')
    ax5.contourf(lon, lat, dict_pco2, np.arange(-1,1.2,.2), transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))

    ax6 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=180))
    plt.title(r"(b) $\rho(T',DIC')$, non-pCO$_2$ term")
    ax6.coastlines('50m')
    im = ax6.contourf(lon, lat, dict_anom, np.arange(-1,1.2,.2), transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap=plt.get_cmap('RdBu_r'))

    #fig.subplots_adjust(bottom=0.2)
    ax_cbar1 = fig.add_axes([0.25, 0.1, .5, 0.05])
    fig.colorbar(im, cax=ax_cbar1, orientation="horizontal", pad=0.2)
    plt.tight_layout()
    plt.savefig('/Users/keturner/Desktop/dict_corr_breakdown_sewn.eps', format='eps')

################################
## 26 April - eliminating one model type from the ensemble
coeff_CO2_elim= np.zeros([6,180,359])
coeff_t_elim= np.zeros([6,180,359])
coeff_s_elim= np.zeros([6,180,359])

for i in np.arange(0,6):
    ri_i = np.arange(i*5*60,(i+1)*5*60)
    full = np.arange(30*60)
    test_index = np.delete(full,ri_i)

    for ii in np.arange(0,180):
            for j in np.arange(0,359):
                if dic_resid[ri_i[0],ii,j] != 0 :
                    input_vars = np.stack((pco2_extended[test_index], theta_resid[test_index,ii,j],ens_sal[test_index,ii,j]), axis=-1)
                    dummy = scipy.linalg.lstsq(input_vars, dic_resid[test_index,ii,j])
                
                    coeff_CO2_elim[i,ii,j] = dummy[0][0]
                    coeff_t_elim[i,ii,j] = dummy[0][1]
                    coeff_s_elim[i,ii,j] = dummy[0][2]
    print(i)
    
fig = plt.figure(figsize=(12, 6))
ax0 = plt.subplot(2, 3, 1, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) ACCESS removed')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, np.ma.masked_where(land_mask,sew_seams(coeff_CO2_elim[0,:,:],ens_mask)),
                   vmin =-0.15, vmax =0.15, transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=plt.get_cmap('RdBu_r', 12),rasterized=True)

ax1 = plt.subplot(2, 3, 2, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(b) CanESM removed')
ax1.coastlines('50m')
im1 = ax1.pcolormesh(lon, lat, np.ma.masked_where(land_mask,sew_seams(coeff_CO2_elim[1,:,:],ens_mask)),
                   vmin =-0.15, vmax =0.15, transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=plt.get_cmap('RdBu_r', 12),rasterized=True)

ax2 = plt.subplot(2, 3, 3, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(c) CESM removed')
ax2.coastlines('50m')
im2 = ax2.pcolormesh(lon, lat, np.ma.masked_where(land_mask,sew_seams(coeff_CO2_elim[2,:,:],ens_mask)),
                   vmin =-0.15, vmax =0.15, transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=plt.get_cmap('RdBu_r', 12),rasterized=True)

ax3 = plt.subplot(2, 3, 4, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(d) IPSL removed')
ax3.coastlines('50m')
im3 = ax3.pcolormesh(lon, lat, np.ma.masked_where(land_mask,sew_seams(coeff_CO2_elim[3,:,:],ens_mask)),
                   vmin =-0.15, vmax =0.15, transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=plt.get_cmap('RdBu_r', 12),rasterized=True)

ax4 = plt.subplot(2, 3, 5, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(e) MPI-LR removed')
ax4.coastlines('50m')
im4 = ax4.pcolormesh(lon, lat, np.ma.masked_where(land_mask,sew_seams(coeff_CO2_elim[4,:,:],ens_mask)),
                   vmin =-0.15, vmax =0.15, transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=plt.get_cmap('RdBu_r', 12),rasterized=True)

ax5 = plt.subplot(2, 3, 6, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(f) UKESM removed')
ax5.coastlines('50m')
im5 = ax5.pcolormesh(lon, lat, np.ma.masked_where(land_mask,sew_seams(coeff_CO2_elim[5,:,:],ens_mask)),
                   vmin =-0.15, vmax =0.15, transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=plt.get_cmap('RdBu_r', 12),rasterized=True)
ax_cbar1 = fig.add_axes([0.3, 0.05, .4, 0.05])
fig.colorbar(im5, cax=ax_cbar1, orientation="horizontal", pad=0.2)
plt.savefig('/Users/keturner/Desktop/sens_cCO2_100_500.eps', format='eps')
   

    
rmse_ensfit = np.zeros((6,180,359))
rmse_prior = np.zeros((6,180,359))
rmse_ensfit_detrend = np.zeros((6,180,359))
rmse_prior_detrend = np.zeros((6,180,359))
var_exp_detrend = np.zeros((6,180,359))

corr_at = np.zeros((6,180,359))
corr_at_detrend = np.zeros((6,180,359))

for i in np.arange(0,6):
    ri_i = np.arange(i*5*60,(i+1)*5*60)
    dic_truth = dic_resid[ri_i,:,:]
    dic_recon_enscov = coeff_t_elim[i,:,:] * theta_resid[ri_i,:,:] + \
        coeff_s_elim[i,:,:] * ens_sal[ri_i,:,:] + \
        coeff_CO2_elim[i,:,:] * np.transpose(np.tile(pco2_extended[:300],(359,180,1)),(2,1,0))

    # for ii in np.arange(0,180):
    #         for j in np.arange(0,359):
    #             if dic_resid[ri_i[0],ii,j] != 0 :  
    #                 corr_at_detrend[i,ii,j] = np.corrcoef(scipy.signal.detrend(dic_truth[:,ii,j]), 
    #                                                       scipy.signal.detrend(dic_recon_enscov[:,ii,j]))[0,1]
    
    dic_truth_detrend = detrend_ens(dic_truth, 0, 60)
    dic_recon_detrend = detrend_ens(dic_recon_enscov, 0, 60)

    # rmse_ensfit[i,:,:] = RMSE(dic_truth,dic_recon_enscov,0)
    # rmse_prior[i,:,:] = RMSE(dic_truth,0,0)
    
    # rmse_prior_detrend[i,:,:] = RMSE(dic_truth_detrend,0,0)
    # rmse_ensfit_detrend[i,:,:] = RMSE(dic_truth_detrend,dic_recon_detrend,0)
    
    var_exp_detrend[i,:,:] = 100 * (1-np.var(dic_truth_detrend - dic_recon_detrend, axis=0)/np.var(dic_truth_detrend, axis=0))

    
error_improv_ens = (-rmse_ensfit + rmse_prior) / rmse_prior
error_improv_ens_detrend = (-rmse_ensfit_detrend + rmse_prior_detrend) / rmse_prior_detrend

# plt.pcolormesh(error_improv_ens[0,:,:], vmin=-1, vmax=1)
# plt.colorbar()

plt.pcolormesh(np.nanmean(var_exp_detrend,axis=0), vmin=0, vmax=100, cmap=plt.get_cmap('jet', 20))
plt.colorbar()


plt.pcolormesh(np.nanmean(var_exp_detrend, axis=0), vmin=0, vmax=100, cmap = plt.get_cmap('YlGn'))
plt.colorbar()

impr_max =  np.ma.masked_where(land_mask,sew_seams(np.nanmax(error_improv_ens, axis=0),ens_mask))
impr_avg =  np.ma.masked_where(land_mask,sew_seams(np.nanmean(error_improv_ens, axis=0),ens_mask))
impr_min =  np.ma.masked_where(land_mask,sew_seams(np.nanmin(error_improv_ens, axis=0),ens_mask))

impr_maxd =  np.ma.masked_where(land_mask,sew_seams(np.nanmax(error_improv_ens_detrend, axis=0),ens_mask))
impr_avgd =  np.ma.masked_where(land_mask,sew_seams(np.nanmean(error_improv_ens_detrend, axis=0),ens_mask))
impr_mind =  np.ma.masked_where(land_mask,sew_seams(np.nanmin(error_improv_ens_detrend, axis=0),ens_mask))

plt.pcolormesh(impr_avg**2 - impr_avgd**2)
plt.colorbar()
####
####
fig = plt.figure(figsize=(12, 6))
ax0 = plt.subplot(2, 3, 1, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(a) Ensemble minimum')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, impr_min, vmin=-.1, vmax=1,
                   transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                   cmap=newcmp,rasterized=True)


ax1 = plt.subplot(2, 3, 2, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(b) Ensemble average')
ax1.coastlines('50m')
ax1.pcolormesh(lon, lat, impr_avg, vmin=-.1, vmax=1, 
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp,rasterized=True)

ax2 = plt.subplot(2, 3, 3, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(c) Ensemble maximum')
ax2.coastlines('50m')
ax2.pcolormesh(lon, lat, impr_max, vmin=-.1, vmax=1, 
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp,rasterized=True)

ax3 = plt.subplot(2, 3, 4, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(d) Ensemble minimum')
ax3.coastlines('50m')
ax3.pcolormesh(lon, lat, impr_mind, vmin=-.1, vmax=1, 
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp,rasterized=True)

ax4 = plt.subplot(2, 3, 5, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(e) Ensemble average')
ax4.coastlines('50m')
ax4.pcolormesh(lon, lat, impr_avgd, vmin=-.1, vmax=1, 
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp,rasterized=True)

ax5 = plt.subplot(2, 3, 6, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(f) Ensemble maximum')
ax5.coastlines('50m')
im = ax5.pcolormesh(lon, lat, impr_maxd, vmin=-.1, vmax=1, 
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), 
             cmap=newcmp,rasterized=True)
ax_cbar1 = fig.add_axes([0.3, 0.05, .4, 0.05])
fig.colorbar(im, cax=ax_cbar1, orientation="horizontal", pad=0.2)
plt.savefig('/Users/keturner/Desktop/ptwise_100_500_ens_sewn22.eps', format='eps')

plt.plot(np.arange(2), np.arange(2))
plt.title('Relative RMSE improvement $\epsilon$, full signal')
plt.savefig('/Users/keturner/Desktop/t2.eps', format='eps')