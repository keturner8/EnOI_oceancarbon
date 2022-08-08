#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 13:57:46 2022
Run argo_reconstruct_method2 and man_argo_plots FIRST just to make sure these plots work...
@author: keturner
"""
####################################################################################
## 24 March - testing coefficients for NorESM reconstruction
f_dic = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/dissic_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
nor_dic = rf.integ_layer(f_dic, f_dic.dissic, limit_z, years_used, mdict)[0]
f_theta = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/thetao_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
nor_theta = rf.integ_layer(f_theta, f_theta.thetao, limit_z, years_used, mdict)[0]
f_so = xr.open_dataset('/Users/keturner/nc_files/CMIP6/mNorESM/so_Oyr_NorESM2-LM_historical_r1i1p1f1_gr_1950-2014_1x1r.nc')
nor_so = rf.integ_layer(f_so, f_so.so, limit_z, years_used, mdict)[0]
        

nor_dicresid = nor_dic - np.nanmean(nor_dic, axis=0)
nor_thetaresid = nor_theta - np.nanmean(nor_theta, axis=0)
nor_soresid = nor_so - np.nanmean(nor_so, axis=0)

nor_dic_recon = cT * nor_thetaresid + cS * nor_soresid + cCO2 * np.transpose(np.tile(pco2 - np.mean(pco2),(359,180,1)),(2,1,0))

plt.plot(nor_dicresid[:,20,200], label='model truth')
plt.plot(nor_dic_recon[:,20,200], label='reconstruction')
plt.legend()
plt.title("Integrated DIC at 70S, 160W")
plt.savefig('/Users/keturner/Desktop/reconstruct_100_bad.eps', format='eps')

plt.plot(nor_dicresid[:,121,295], label='model truth')
plt.plot(nor_dic_recon[:,121,295], label='reconstruction')
plt.legend()
plt.title("Integrated DIC at 32N, 60W")
plt.savefig('/Users/keturner/Desktop/reconstruct_100_good.eps', format='eps')

nor_rmse_prior = np.ma.masked_where(land_mask,sew_seams(rmse(nor_dicresid.data,0,0), ens_mask))
nor_rmse_recon = np.ma.masked_where(land_mask,sew_seams(rmse(nor_dicresid.data,nor_dic_recon.data,0), ens_mask))

####
####
fig= plt.figure(figsize=(12, 6))

ax0 = plt.subplot(1, 3, 1, projection=ccrs.Robinson(central_longitude=180))
ax0.coastlines('50m')
im0 = ax0.contourf(lon, lat, nor_rmse_prior, np.arange(0,4.5,.5),
                  transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), extend='both')
plt.colorbar(im0,ax=ax0, fraction=0.04, pad=0.04, orientation="horizontal")
ax0.set_title('(a) NorESM standard deviation', loc='left')

ax1 = plt.subplot(1, 3, 2, projection=ccrs.Robinson(central_longitude=180))
ax1.coastlines('50m')
im1 = ax1.contourf(lon, lat, nor_rmse_recon, np.arange(0,4.5,.5),
                  transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), extend='both')
plt.colorbar(im1,ax=ax1, fraction=0.04, pad=0.04, orientation="horizontal")
ax1.set_title('(b) T/S reconstruction RMSE', loc='left')

ax2 = plt.subplot(1, 3, 3, projection=ccrs.Robinson(central_longitude=180))
ax2.coastlines('50m')
im2 = ax2.contourf(lon, lat,(nor_rmse_prior - nor_rmse_recon)/nor_rmse_prior, np.arange(-1,1.2,.2),
                  transform=ccrs.PlateCarree(central_longitude=0.0, globe=None),  cmap = plt.get_cmap('RdBu'), extend='both')
plt.colorbar(im2,ax=ax2, fraction=0.04, pad=0.04, orientation="horizontal")
ax2.set_title('(c) Relative improvement', loc='left')
plt.savefig('/Users/keturner/Desktop/rmse_nor_100_sewn.eps', format='eps')
####
####
################################################################################
## 24 March, testing trend and detrended variability captured by reconstruction
nor_truth_detrend = scipy.signal.detrend(nor_dicresid, axis=0)
nor_recon_detrend = scipy.signal.detrend(nor_dic_recon, axis=0)

nor_truth_trend = np.diff(nor_dicresid - nor_truth_detrend, axis=0)[0,:,:]
nor_recon_trend = np.diff(nor_dic_recon - nor_recon_detrend, axis=0)[0,:,:]

test1 = (nor_recon_trend - nor_truth_trend) / nor_truth_trend
trend_ouest = np.ma.masked_where(land_mask, sew_seams(test1, ens_mask))
rmse_detrend = np.ma.masked_where(ens_mask, rmse(nor_truth_detrend,nor_recon_detrend,0))
rmse_detrend_truth = np.ma.masked_where(ens_mask, np.std(nor_truth_detrend,axis=0))

####
####
fig= plt.figure(figsize=(12, 6))
ax0 = plt.subplot(1,2,1, projection=ccrs.Robinson(central_longitude=180))
ax0.coastlines('50m')
im0 = ax0.contourf(lon, lat, trend_ouest * 100, np.arange(-100,110,10),
             transform=ccrs.PlateCarree(central_longitude=0.0, globe=None), cmap='RdBu_r')
plt.colorbar(im0)
plt.title('Trend over/underestimation as percentage')
plt.savefig('/Users/keturner/Desktop/trend_100.eps', format='eps')

plt.contourf((rmse_detrend_truth - rmse_detrend)/rmse_detrend_truth, np.arange(-1,1.2,.2), cmap = plt.get_cmap('RdBu'), extend='both')
plt.colorbar()
plt.title('RMSE reduction from reconstruction (detrended data)')
plt.savefig('/Users/keturner/Desktop/rmse_detrend_100.eps', format='eps')
####
####

area_f = xr.open_dataset('/Users/keturner/Downloads/areacello_gr.nc')
area = area_f.areacello[:,:359]

area2 = np.tile(area, [60,1,1])

##need to alter these file nmes and radii for different setups 
nor_recon_5deg_2002 = np.ones((60,180,359)) * np.nan
nor_recon_5deg_2015 = np.ones((60,180,359)) * np.nan

dic_nor = nor_dic - np.nanmean(nor_dic, axis=0)
theta_nor = nor_theta - np.nanmean(nor_theta, axis=0)
sal_nor = nor_so - np.nanmean(nor_so, axis=0)

molC2gC = 12.011

radii = [0,1,2,5]
r = radii[3]
idx = 3

for la in np.arange(0+r,180-1-r):
    for lo in np.arange(0+r,360-1-r):
    
        if ens_mask[la,lo]==0: #check that we have DIC at the location in the first place
            loc_lat = np.arange(la-r,la+r+1)
            loc_lon = np.arange(lo-r,lo+r+1)
            dic_BATS = dic_resid[:,la,lo]

            if argo_2015_mask[loc_lat[0]:loc_lat[-1]+1, loc_lon[0]:loc_lon[-1]+1].data.sum() ==0:
                
                input_vars = np.expand_dims(pco2_extended,1)
                dummy = scipy.linalg.lstsq(input_vars, dic_BATS)
                
                nor_recon_5deg_2015[:,la,lo] = test_argo = dummy[0][0] * pco2_extended[:60]
                        
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

                nor_recon_5deg_2015[:,la,lo] = np.sum(coef_temp * T_test, axis=(1,2)) + np.sum(coef_sal * S_test, axis=(1,2)) + coef_pco2 * pco2_extended[:60]

        if (np.mod(la,9)==0) & (lo==150):
            print(str(r) + ' ' + str(la/180))
            
plt.pcolormesh(np.std(scipy.signal.detrend(dic_nor, axis=0), axis=0))
plt.colorbar()
            
plt.plot(nor_recon_5deg_2015[:,121,295])
plt.plot(nor_recon_5deg_2002[:,121,295])
plt.plot(dic_nor[:,121,295], 'k')

plt.plot(nor_recon_5deg_2015[:,112,201])
plt.plot(nor_recon_5deg_2002[:,112,201])
plt.plot(dic_nor[:,112,201], 'k')

recon_2002_5 = np.nansum(nor_recon_5deg_2002*area2, axis=(1,2))
recon_2015_5 = np.nansum(nor_recon_5deg_2015*area2, axis=(1,2))
nor_global = np.nansum(dic_nor*area2, axis=(1,2)) 

fig = plt.figure(figsize=(18,8))

ax0 = plt.subplot(2, 3, 1)
plt.plot(np.arange(1955,2015), nor_global* molC2gC*1e-15, 'k', label='NorESM truth')
plt.plot(np.arange(1955,2015),recon_2002_5* molC2gC*1e-15, 'b--', label='2002 recon.')
plt.plot(np.arange(1955,2015),recon_2015_5* molC2gC*1e-15, 'r--', label='2015 recon.')
plt.title('(a) Global upper-ocean carbon timeseries')
plt.legend()
plt.ylabel('PgC')

ax0 = plt.subplot(2, 3, 2)
plt.plot(np.arange(1955,2015), dic_nor[:,121,295]* molC2gC, 'k', label='NorESM truth')
plt.plot(np.arange(1955,2015),nor_recon_5deg_2002[:,121,295]* molC2gC, 'b--', label='0 obs., RMSE = 0.493')
plt.plot(np.arange(1955,2015),nor_recon_5deg_2015[:,121,295]* molC2gC, 'r--', label='55 obs., RMSE = 0.157')
plt.title('(b) BATS upper-ocean carbon')
plt.legend()
plt.ylabel('gC m$^{-2}$')

ax0 = plt.subplot(2, 3, 3)
plt.plot(np.arange(1955,2015), dic_nor[:,112,201]* molC2gC, 'k', label='NorESM truth')
plt.plot(np.arange(1955,2015),nor_recon_5deg_2002[:,112,201]* molC2gC, 'b--', label='1 obs., RMSE = 0.540')
plt.plot(np.arange(1955,2015),nor_recon_5deg_2015[:,112,201]* molC2gC, 'r--', label='62 obs., RMSE = 0.377')
plt.title('(c) HOT upper-ocean carbon')
plt.legend()
plt.ylabel('gC m$^{-2}$')

ax0 = plt.subplot(2, 3, 4, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(d) NorESM $\Delta$DIC, 2001-2000')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, (dic_nor[46,:,:]-dic_nor[45,:,:])* molC2gC,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-60, vmax=60,cmap=plt.get_cmap('RdBu'))

ax0 = plt.subplot(2, 3, 5, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(e) 2002 recon. $\Delta$DIC, 2001-2000')
ax0.coastlines('50m')
im0 = ax0.pcolormesh(lon, lat, (nor_recon_5deg_2002[46,:,:]-nor_recon_5deg_2002[45,:,:])* molC2gC,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-60, vmax=60,cmap=plt.get_cmap('RdBu'))

ax0 = plt.subplot(2, 3, 6, projection=ccrs.Robinson(central_longitude=180))
plt.title(r'(f) 2015 recon. $\Delta$DIC, 2001-2000')
ax0.coastlines('50m')
im2 = ax0.pcolormesh(lon, lat, (nor_recon_5deg_2015[46,:,:]-nor_recon_5deg_2015[45,:,:])* molC2gC,
                     transform=ccrs.PlateCarree(central_longitude=0, globe=None),
                     vmin=-60, vmax=60,cmap=plt.get_cmap('RdBu'))

ax_cbar1 = fig.add_axes([0.3, 0.05, .4, 0.05])
cbar = fig.colorbar(im2, cax=ax_cbar1, orientation="horizontal", pad=0.2)
cbar.set_label('$\Delta$DIC, gC m$^{-2}$', fontsize=12)
plt.savefig('/Users/keturner/Desktop/recon_glob2.png', format='png', dpi=300)

## spatial correlations 
nt = np.reshape(dic_nor[46,:,:]-dic_nor[45,:,:], [180*359, 1]).data
nt_idx = nt == 0
nt[nt_idx] = np.nan
n2002 = np.reshape(nor_recon_5deg_2002[46,:,:]-nor_recon_5deg_2002[45,:,:], [180*359, 1])
n2002[nt_idx] = np.nan
n2015 = np.reshape(nor_recon_5deg_2015[46,:,:]-nor_recon_5deg_2015[45,:,:], [180*359, 1])
n2015[nt_idx] = np.nan
plt.scatter(nt,n2015)

test = np.corrcoef(nt,n2015)[0,1]
