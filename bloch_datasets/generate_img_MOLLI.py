# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:09:50 2022

@author: paual
"""
import numpy as np
from molli_bloch.bloch_sequences import simulate_MOLLI
from molli_bloch.bloch_inference import infer_T2_map
from molli_bloch.bloch_inference import infer_B1_map



def generate_img_MOLLI_dataset(t_array, T1_map, B1_range = (0.99, 1.01)):
    
    m, n = T1_map.shape
    
    T2_map = infer_T2_map(T1_map)
    B1_map = infer_B1_map(T1_map, B1_range[0], B1_range[1])
    
    simulated_MOLLI = np.empty((len(t_array), m, n))
    
    for i in range(m):
        for j in range(n):
            
            T1 = T1_map[i, j]
            T2 = T2_map[i, j]
            B1 = T2_map[i, j]
            
            simulated_MOLLI[:, i, j] = simulate_MOLLI(t_array, T1, T2, B1)
            
    return simulated_MOLLI, T2_map, B1_map