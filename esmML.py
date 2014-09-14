#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import numpy as np
from sklearn.metrics import r2_score,explained_variance_score,mean_absolute_error

from scipy import interpolate

# calculate the fit to the data R2
def calcR2(model,data,azGridSpacing=0.5,zenGridSpacing=0.5):

    az = np.linspace(0,360, int(360./azGridSpacing)+1 )
    zen = np.linspace(0,90, int(90./zenGridSpacing)+1 )
    x = [] # actual data
    y = [] # predited data
    ctr = 0
    model = np.nan_to_num(model)
    model_test = interpolate.interp2d(az, zen, model.reshape(az.size * zen.size,), kind='linear')
    for i in range(0,np.shape(data)[0]):
        if not np.isnan(data[i,2]) and not np.isnan(model_test(data[i,0],data[i,1])):
            x.append(data[i,2])
            y.append(model_test(data[i,0],data[i,1])[0])

    r2  = r2_score(np.array(x),np.array(y)) 
    #evs = explained_variance_score(np.array(x),np.array(y)) 
    mae = mean_absolute_error(np.array(x),np.array(y)) 
    #print("r2:",r2)
    #print("r2,mae",r2,mae)
    return r2, mae

