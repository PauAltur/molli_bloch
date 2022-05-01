# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:21:54 2022

@author: paual
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

def load_MOLLI_dataset(path):
    
    
    df = pd.read_csv(path)
    
    mollis = np.empty((int(len(df)/2), 2, 8))
    params = np.empty((int(len(df)/2), 3))
    
    for i in tqdm(range(0, len(df), 2)):
        
        mollis[int(i/2), 0, :] = df.iloc[i, :8]
        mollis[int(i/2), 1, :] = df.iloc[i+1, :8]
        params[int(i/2), :] = df.iloc[i, 8:]
        
    return mollis, params