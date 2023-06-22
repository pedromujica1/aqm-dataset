#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:52:23 2023

@author: mateus
"""

import pandas as pd


fn = 'IAG/estacao_iag.csv'
df = pd.read_csv(fn)
print(df.columns)

for col in df.columns:
    c = col.split()
    print("".join(c[0:len(c) -1]).lower())
    
df.columns = ["".join(c.split()[0:len(c.split()) -1]).lower() for c in df.columns] 
df.to_csv('IAG/iag_df.csv', index=True)  # Replace 'path/to/save/modified_df.csv' with the actual file path where you want to save the DataFrame
