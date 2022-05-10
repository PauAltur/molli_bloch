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
    blood_mask = T1_masked > 1300

    myo_var = (5 - (-5)) * np.random.random_sample(size = mask.shape) + (-5)
    blood_var =  (5 - (-5)) * np.random.random_sample(size = mask.shape) + (-5)

    T2_myo = (40 + myo_var) * myo_mask
    T2_blood = (180 + blood_var) * blood_mask

    T2_fake_map = T2_fake_map * np.logical_not(mask)

    T2_map = T2_fake_map + T2_myo + T2_blood

    return T2_map
    
    