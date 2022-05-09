# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:30:29 2022

@author: paual
"""


import numpy as np

def infer_T2_map(mask, T1_map):

    T2_fake_map = T1_map / 10

    T1_masked = T1_map * mask

    myo_mask = np.logical_and(T1_masked <= 1300, T1_masked != 0)
    T1_masked[myo_mask] = 40 # substitute by real values
    T1_masked[T1_masked > 1300] = 180 

    T2_fake_map_masked = T2_fake_map * np.logical_not(mask)
    
    T2_map = T2_fake_map_masked + T1_masked

    return T2_map 
    
    