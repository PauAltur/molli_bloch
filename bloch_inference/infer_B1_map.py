# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:39:55 2022

@author: paual
"""

import numpy as np

def infer_B1_map(T1_map, B1_min, radius):
    
    m, n = T1_map.shape
    
    x = np.arange(0, m)
    y = np.arange(0, n)
    
    xv, yv = np.meshgrid(x, y)
    
    xv = np.abs(xv - np.round(m/2))
    yv = np.abs(yv - np.round(n/2))
    
    distance_map = np.sqrt(xv**2 + yv**2)
    distance_map_norm = distance_map / distance_map.max()

    B1_max = (((1 - B1_min) * distance_map.max()) / radius) + B1_min
    B1_map = (B1_max - B1_min) * distance_map_norm + B1_min

    return B1_map