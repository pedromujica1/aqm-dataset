#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 20:57:58 2023

@author: mateus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def convert_float(value):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        return np.nan
    except AttributeError:
        return value

fn1 = 'envcity_aqm_df.csv'
fn2 = 'iag_df.csv'



df = pd.read_csv(fn1)
df.set_index('time', inplace=True)

dt = df.dtypes.values == 'object'
for col in df.dtypes.index[dt]:
    df[col] = df[col].apply(convert_float).astype('float32')


print(df.dtypes)
co_iag = df['iag_co']
co_iag.plot()
# co_iag = pd.to_numeric(df['iag_co'], errors='raise')

plt.gcf().autofmt_xdate()