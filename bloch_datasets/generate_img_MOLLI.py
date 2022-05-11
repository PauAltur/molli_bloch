# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:09:50 2022

@author: paual
"""
import numpy as np
from molli_bloch.bloch_sequences import simulate_MOLLI
from molli_bloch.bloch_inference import infer_T2_map
from molli_bloch.bloch_inference import infer_B1_map



def generate_img_MOLLI(t_array, T1_map, mask, B1_min, B1_radius):
    
    m, n = T1_map.shape
    
    T2_map = infer_T2_map(mask, T1_map)
    B1_map = infer_B1_map(T1_map, B1_min, B1_radius)
    
    simulated_MOLLI = np.empty((m, n, len(t_array)))
    
    for i in range(m):
        for j in range(n):
            if mask[i, j] == 1:
            
                T1 = T1_map[i, j]
                T2 = T2_map[i, j]
                B1 = B1_map[i, j]
                
                M_ord_abs, M_ord, M_raw = simulate_MOLLI(t_array, T1, T2, B1)
                simulated_MOLLI[i, j, :] = M_ord_abs[1, :]
            
    return simulated_MOLLI, T2_map, B1_map