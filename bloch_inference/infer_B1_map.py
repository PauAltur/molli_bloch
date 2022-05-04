# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:39:55 2022

@author: paual
"""

import numpy as np

def infer_B1_map(T1_map, B1_min, B1_max):
    
    m, n = T1_map.shape
    
    x = np.arange(0, m)
    y = np.arange(0, n)
    
    xv, yv = np.meshgrid(x, y)
    
    xv = np.abs(xv - np.round(m/2))
    yv = np.abs(yv - np.round(n/2))
    
    circle_map = (xv**2 + yv**2) / np.random.randint(0, np.round(m/2)) ** 2
    
    B1_map_over = ((circle_map - 1) * (circle_map > 1))
    
    ## Scale the gradient of B1 over and under the unit circle
    
    return B1_map