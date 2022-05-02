# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:40:49 2022

@author: paual
"""
import numpy as np
import math

def T_i_from_hr(hr):
    '''
    Function to generate synthetic time vectors for MOLLI series acquisiton
    with a specific heart rate.

    Parameters
    ----------
    hr : int
        Heart rate in beats per minute.

    Returns
    -------
    T_i : numpy.array
        Array that contains the inversion times to generate a MOLLI series. This
        can be used as the input to generate MOLLI series using the function
        molli_bloch.bloch_sequences.simulateMOLLI().

    '''
    
    hr_ms = hr / 60000      # convert to bpms
    period_ms = 1 / hr_ms   # calculate period
    
    T_chr = np.empty((11,))
    T_chr[0] = 0.15 * period_ms   # first acquisition
    
    for i in range(5):
        T_chr[i + 1] = T_chr[i] + period_ms # 2nd to 5th acquistions
    
    T_chr[6] = T_chr[5] + 3000   # 3000 ms of rest
        
    frac_rest = T_chr[6] / period_ms   # heartbeat fraction after 3000 ms rest
    frac_next = math.ceil(frac_rest)   # find heartbeat fraction of next same moment in heartbeat
    
    T_chr[7] = frac_next * period_ms - 0.2 * period_ms  # calculate inversion time for 2nd IR experiment
    T_chr[8] = frac_next * period_ms   # find time of next same moment in heartbeat
    
    for i in range(2):
        T_chr[i + 9] = T_chr[i + 8] + period_ms   # 6th to 8th acquisition
    
    # reorder and calculate inversion times
    T_i = np.array([T_chr[0], T_chr[8] - T_chr[7], T_chr[1], T_chr[9] - T_chr[7], T_chr[2], T_chr[10] - T_chr[7], T_chr[3], T_chr[4]])
    
    return T_i