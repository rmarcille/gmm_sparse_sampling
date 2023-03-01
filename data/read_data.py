import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import datetime
import time 
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def read_wind_data(path, area, mask, LON, EOFs_u, EOFs_v):
    df_area = pd.DataFrame()
    for i, year in enumerate([2016, 2017, 2018]):
        for j,filename in enumerate(os.listdir(path + '/' + str(year) + '/nc_files/')):
            month = int(filename[5:7])
            day = int(filename[8:10])
            hours = [i for i in range(24)]
            dt_index = [datetime.datetime(year, month, day, hour) for hour in hours]
            data = nc.Dataset(path + '/' + str(year) + '/nc_files/' + filename)
            
            u = data['u'][:-1, :, :] * mask
            v = data['v'][:-1, :, :] * mask
    
            u = np.ma.masked_array(u, np.isnan(u))
            v = np.ma.masked_array(v, np.isnan(v))
            
            sea = ~u.mask
            u = np.reshape(u[sea], (24, len(LON)))
            v = np.reshape(v[sea], (24, len(LON)))
            u = pd.DataFrame(u)
            v = pd.DataFrame(v)
            
            dfin = pd.concat([u, v], axis = 1)
            dfin.index = dt_index

            n_input_points = u.shape[1]
            cols = ['u' + str(i) for i in range(n_input_points)] + ['v' + str(i) for i in range(n_input_points)]
            colsY= ['eofu' + str(i) for i in range(n_input_points)] + ['eofv' + str(i) for i in range(n_input_points)]

            #Concatenate in rows 
            if (j==0)*(i==0):
                df_area = dfin
                df_area.columns = cols
            else :
                dfin.columns = cols
                df_area = pd.concat([df_area, dfin])
    Yu = pd.DataFrame(np.dot(df_area.iloc[:, :n_input_points], EOFs_u.T))
    Yv = pd.DataFrame(np.dot(df_area.iloc[:, n_input_points:], EOFs_v.T))
    df_output = pd.concat([Yu, Yv], axis = 1)
    df_output.index = df_area.index
    return df_area, df_output
    
def read_eofs(path0, area, mask, N_eofs, LON):
    read_EOFs = np.load(path0 + area + '/EOFs_wind.npz')
    var_EOFs = read_EOFs.files
    EOFs_u = read_EOFs[var_EOFs[0]][:N_eofs, :, :] * mask
    EOFs_v = read_EOFs[var_EOFs[1]][:N_eofs, :, :] * mask
    EOFs_u = np.ma.masked_array(EOFs_u, np.isnan(EOFs_u))
    EOFs_v = np.ma.masked_array(EOFs_v, np.isnan(EOFs_v))
    sea2 = ~EOFs_u.mask
    EOFs_v = np.reshape(EOFs_v[sea2], (N_eofs, len(LON)))
    EOFs_u = np.reshape(EOFs_u[sea2], (N_eofs, len(LON)))
    V_svd = pd.concat([pd.DataFrame(EOFs_u.T), pd.DataFrame(EOFs_v.T)], axis = 1)
    return EOFs_u, EOFs_v, V_svd

def read_lat_lon(path0, area): 
    path = path0 + area
    #Read lat and lon, compute masked array
    mask = nc.Dataset(path + '/mask.nc')
    lat = np.squeeze(mask.variables['lat'][:])
    lon = np.squeeze(mask.variables['lon'][:])
    mask = mask['mask'][:]
    mask[mask==1]=float('nan')
    mask[mask==0]=1
    [LON, LAT] = np.meshgrid(lon, lat)
    LON = LON*mask
    LAT = LAT*mask
    LON = np.ma.masked_array(LON, np.isnan(LON))
    LAT = np.ma.masked_array(LAT, np.isnan(LAT))
    land = LON.mask
    sea = ~land
    LON = LON[sea].data
    LAT = LAT[sea].data
    df_latlon = pd.DataFrame({'lat' : LAT, 'lon' : LON})
    return df_latlon, LAT, LON, mask, lat, lon


def standardize(y, mean, std):
    return (y - mean)/std


def score_reconstructed(Yt, Yp, V, LAT, LON, Npc, mean_u=0, mean_v=0, std_u=1, std_v=1):
    NpcT = int(Yt.shape[1]/2)
    Ytestunorm = np.dot(Yt[:, :NpcT], V[:, :NpcT].T)
    Ytestunorm = (Ytestunorm - mean_u)/std_u
    Ytestvnorm = np.dot(Yt[:, NpcT:], V[:, NpcT:].T)
    Ytestvnorm = (Ytestvnorm - mean_v)/std_v
    
    Ytestnorm = np.hstack((Ytestunorm, Ytestvnorm))
    umagtest = np.sqrt(Ytestunorm**2 + Ytestvnorm**2)
    
    Ypredunorm = np.dot(Yp[:, :Npc], V[:, :Npc].T)
    Ypredunorm = (Ypredunorm - mean_u)/std_u
    Ypredvnorm = np.dot(Yp[:, Npc:], V[:, NpcT:Npc + NpcT].T)
    Ypredvnorm = (Ypredvnorm - mean_v)/std_v
    Ypredrecnorm = np.hstack((Ypredunorm, Ypredvnorm))
    umagpred = np.sqrt(Ypredunorm**2 + Ypredvnorm**2)

    err = mean_squared_error(umagtest, umagpred, squared = False)
    err_uv = mean_squared_error(Ypredrecnorm, Ytestnorm, squared = False)
    umagi = []
    df_umag = []
    
    return err, err_uv, umagi, df_umag




def score_reconstructed_max_mean(Yt, Yp, V, LAT, LON, Npc, mean_u=0, mean_v=0, std_u=1, std_v=1):
    NpcT = int(Yt.shape[1]/2)
    Ytestunorm = np.dot(Yt[:, :NpcT], V[:, :NpcT].T)
    Ytestunorm = (Ytestunorm - mean_u)/std_u
    Ytestvnorm = np.dot(Yt[:, NpcT:], V[:, NpcT:].T)
    Ytestvnorm = (Ytestvnorm - mean_v)/std_v
    
    Ytestnorm = np.hstack((Ytestunorm, Ytestvnorm))
    umagtest = np.sqrt(Ytestunorm**2 + Ytestvnorm**2)
    
    Ypredunorm = np.dot(Yp[:, :Npc], V[:, :Npc].T)
    Ypredunorm = (Ypredunorm - mean_u)/std_u
    Ypredvnorm = np.dot(Yp[:, Npc:], V[:, NpcT:Npc + NpcT].T)
    Ypredvnorm = (Ypredvnorm - mean_v)/std_v
    Ypredrecnorm = np.hstack((Ypredunorm, Ypredvnorm))
    umagpred = np.sqrt(Ypredunorm**2 + Ypredvnorm**2)
    max_umagpred = umagpred.max(axis = 0)
    max_umagtest = umagtest.max(axis = 0)
    err_umax = mean_squared_error(max_umagpred, max_umagtest, squared = False)
    mean_umagpred = umagpred.mean(axis = 0)
    mean_umagtest = umagtest.mean(axis = 0)
    err_umean = mean_squared_error(mean_umagpred, mean_umagtest, squared = False)

    
    err = mean_squared_error(umagtest, umagpred, squared = False)
    err_uv = mean_squared_error(Ypredrecnorm, Ytestnorm, squared = False)
    umagi = []
    df_umag = []
    return err, err_uv, umagi, df_umag, err_umax, err_umean


def LeastSquares(X, Y, Xt, Yt, V, LAT, LON, Npc=10, nrmse = False): 
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),Y)
    Ypred = np.dot(beta_hat.T, np.hstack((np.ones((Xt.shape[0], 1)), Xt)).T).T
    mean_u_train = np.dot(Y[:, :Npc], V[:, :Npc].T).mean()
    mean_v_train = np.dot(Y[:, Npc:], V[:, Npc:].T).mean()
    std_u_train = np.dot(Y[:, :Npc], V[:, :Npc].T).std()
    std_v_train = np.dot(Y[:, Npc:], V[:, Npc:].T).std()
    if nrmse : 
        errRecons, errRecons_uv, umagi, df_umag = score_reconstructed(Yt, Ypred, V, LAT, LON,10, mean_u_train,mean_v_train ,std_u_train,std_v_train)
    else:
        errRecons, errRecons_uv, umagi, df_umag = score_reconstructed(Yt, Ypred, V, LAT, LON,10)
    return errRecons_uv, Ypred, umagi

def LeastSquares_max_mean(X, Y, Xt, Yt, V, LAT, LON, Npc=10, nrmse = False): 
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),Y)
    Ypred = np.dot(beta_hat.T, np.hstack((np.ones((Xt.shape[0], 1)), Xt)).T).T
    mean_u_train = np.dot(Y[:, :Npc], V[:, :Npc].T).mean()
    mean_v_train = np.dot(Y[:, Npc:], V[:, Npc:].T).mean()
    std_u_train = np.dot(Y[:, :Npc], V[:, :Npc].T).std()
    std_v_train = np.dot(Y[:, Npc:], V[:, Npc:].T).std()
    if nrmse : 
        errRecons, errRecons_uv, umagi, df_umag, err_umax, err_umean = score_reconstructed_max_mean(Yt, Ypred, V, LAT, LON,10, mean_u_train,mean_v_train ,std_u_train,std_v_train)
    else:
        errRecons, errRecons_uv, umagi, df_umag, err_umax, err_umean = score_reconstructed_max_mean(Yt, Ypred, V, LAT, LON,10)
    return errRecons_uv, Ypred, umagi, err_umax, err_umean