# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 16:22:15 2023

@author: robin.marcille
"""
import numpy as np
from sklearn.metrics import mean_squared_error

def score_reconstructed(Yt, Yp, V, Npc):
    
    NpcT = int(Yt.shape[1]/2)
    Ytestu = np.dot(Yt[:, :NpcT], V[:, :NpcT].T)
    Ytestv = np.dot(Yt[:, NpcT:], V[:, NpcT:].T)
    
    Ytest = np.hstack((Ytestu, Ytestv))
    umagtest = np.sqrt(Ytestu**2 + Ytestv**2)
    
    Ypredu = np.dot(Yp[:, :Npc], V[:, :Npc].T)
    Ypredv = np.dot(Yp[:, Npc:], V[:, NpcT:Npc + NpcT].T)
    Ypredrec = np.hstack((Ypredu, Ypredv))
    umagpred = np.sqrt(Ypredu**2 + Ypredv**2)

    err = mean_squared_error(umagtest, umagpred, squared = False)
    err_uv = mean_squared_error(Ypredrec, Ytest, squared = False)
    return err, err_uv
