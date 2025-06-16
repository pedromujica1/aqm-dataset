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


df1 = pd.read_csv("e2sp_df.csv")
df1.set_index('time', inplace=True)

df2 = pd.read_csv("data_spiffs.csv")
df2['time'] += 1679055000 + 3*60*60
df2.set_index('time', inplace=True)
df2.index = pd.to_datetime(df2.index ,unit='s').tz_localize('Brazil/East').strftime('%Y-%m-%d %H:%M:00')

df2 = df2.add_prefix("e2sp_")

df2.index = pd.DatetimeIndex(df2.index)
#df2= df2[~df2.index.duplicated()] 

#orfun = df2.resample('1min').asfreq()
#df2 = df2.reindex_like(forfun)


dt = df2.dtypes.values == 'object'
for col in df2.dtypes.index[dt]:
    df2[col] = df2[col].apply(convert_float).astype('float64')

joined_df = pd.concat([df1, df2], axis=0, verify_integrity=False)

#for col in ['e2sp_temp', 'e2sp_umid']:
#     print(col)
#     df1.loc[str(df2.index[0]):str(df2.index[-1]), col] = df2[col].values
     

print(df1.index.min(), str(df1.index.max()))
print(df2.index.min(), str(df2.index.max()))
# print(joined_df.index.min(), str(joined_df.index.max()))
# joined_df.to_csv('spiffs_concat.csv', decimal='.')

joined_df.index = pd.DatetimeIndex(joined_df.index)
joined_df = joined_df.sort_values('time').groupby('time').agg('mean')
joined_df.to_csv('e2sp_df.csv', decimal='.')