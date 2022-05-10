# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:39:55 2022

@author: paual
"""

import numpy as np
import random as rand

def infer_B1_map(T1_map, B1_min, B1_max):
    
    m, n = T1_map.shape
    
    x = np.arange(0, m)
    y = np.arange(0, n)
    
    xv, yv = np.meshgrid(x, y)
    
    xv = np.abs(xv - np.round(m/2))
    yv = np.abs(yv - np.round(n/2))
    
    radius = rand.uniform(m / 4, m / 2)
     
    B1_map = (xv**2 + yv**2) / radius ** 2

    return B1_map