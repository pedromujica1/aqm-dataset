#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:52:23 2023

@author: mateus
"""

import pandas as pd
import numpy as np

#fn = 'IAG/estacao_iag.csv'
fn = 'IAG/iag_complementar.csv'
df = pd.read_csv(fn, decimal=',')
df.set_index('time', inplace=True)
df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M').tz_localize('Brazil/East').strftime('%Y-%m-%d %H:%M:00')
#df.index = pd.to_datetime(df.index)#.strftime('%Y-%m-%d %H:%M:00')

def convert_float(value):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        return np.nan
    except AttributeError:
        return value

for col in df.columns:
    c = col.split()
    print("".join(c[0:len(c) -1]).lower())
    
df.columns = ["".join(c.split()[0:len(c.split()) -1]).lower() for c in df.columns] 
df = df.add_prefix('iag_')

dt = df.dtypes.values == 'object'
for col in df.dtypes.index[dt]:
    df[col] = df[col].apply(convert_float).astype('float32')


df['iag_co'].plot()
df.to_csv('iag_df_complementar.csv', index=True)  # Replace 'path/to/save/modified_df.csv' with the actual file path where you want to save the DataFrame
