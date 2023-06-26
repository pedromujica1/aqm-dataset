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


#orfun = df2.resample('1min').asfreq()
#df2 = df2.reindex_like(forfun)

plt.figure()
df2['e2sp_temp'].plot(marker = '.')
plt.show()

dt = df2.dtypes.values == 'object'
for col in df2.dtypes.index[dt]:
    df2[col] = df2[col].apply(convert_float).astype('float64')

print(df2.columns)

for col in df2.columns:
    for idx in df2.index:
        df1.loc[idx,col] = df2.loc[idx, col]
#for col in ['e2sp_temp', 'e2sp_umid']:
#     print(col)
#     df1.loc[str(df2.index[0]):str(df2.index[-1]), col] = df2[col].values
    

# for val, idx in zip(df2['e2sp_temp'], df2['e2sp_temp'].index):
#     print(idx, val)
    
plt.figure()
plt.plot(df2['e2sp_temp'].values, marker='.')

#print(df2[col])
#print(df1.loc[str(df2.index[0]):str(df2.index[-1])][col])

print(df2.index[0], str(df2.index[-1]))
df1.to_csv('envcity_aqm_df_teste.csv', decimal='.')