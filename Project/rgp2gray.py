# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 20:02:51 2018

@author: Lenovo-PC
"""
import numpy as np
def GetGray(img):
#    return (0.2989 .* np.array(img)[:,:,0] + 0.5870 .* np.array(img)[:,:,1] + 0.1140 .* np.array(img)[:,:,2])
    gray = np.full((len(np.array(img)[:,0,0]),len(np.array(img)[0,:,0])),0.0)
    for i in range(len(np.array(img)[:,0,0])):
        for j in range(len(np.array(img)[0,:,0])):
            gray[i,j] = (0.2989 * np.array(img)[i,j,0] + 0.5870 * np.array(img)[i,j,1] + 0.1140 * np.array(img)[i,j,2])
    return gray