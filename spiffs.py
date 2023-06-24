#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:33:50 2023

@author: mateus
"""
import pandas as pd
import matplotlib.pyplot as plt

def convert_float(value):
    try:
        return float(value.replace(',', '.'))
    except ValueError:
        return -5
    except AttributeError:
        return -8


df1 = pd.read_csv("all_tago_df.csv")
df1.set_index('time', inplace=True)

df2 = pd.read_csv("data_spiffs.csv")
df2['time'] += 1679055000 + 3*60*60
df2.set_index('time', inplace=True)
df2.index = pd.to_datetime(df2.index ,unit='s').tz_localize('Brazil/East').strftime('%Y-%m-%d %H:%M:00')

df2 = df2.add_prefix("e2sp_")

df2.index = pd.DatetimeIndex(df2.index)
df2= df2[~df2.index.duplicated()] 

df2 = df2.resample('1T').asfreq()

dt = df2.dtypes.values == 'object'
for col in df2.dtypes.index[dt]:
    df2[col] = df2[col].apply(convert_float).astype('float64')


# Step 4: Output the joined DataFrame
for col in df2.columns:
    df1.loc[str(df2.index[0]):str(df2.index[-1]), col] = df2[col].values


for i in df1.dtypes:
    print(i)

df1['iag_co'].plot()
#print(df2[col])
#print(df1.loc[str(df2.index[0]):str(df2.index[-1])][col])

df1.to_csv('envcity_aqm_df.csv', decimal='.')