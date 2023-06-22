#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:33:50 2023

@author: mateus
"""
import pandas as pd

df1 = pd.read_csv("all_tago_df.csv")
df1.set_index('time', inplace=True)

df2 = pd.read_csv("data_spiffs.csv")
df2['time'] += 1679055000 + 3*60*60
df2.set_index('time', inplace=True)
df2.index = pd.to_datetime(df2.index,unit='s').tz_localize('Brazil/East').strftime('%Y-%m-%d %H:%M:00')

df2 = df2.add_prefix("e2sp_")

df2 = df2.resample('1T').asfreq()

# Step 3: Join the DataFrames based on a common column(s)
joined_df = pd.merge(df1, df2, left_index=True, right_index=True)
#joined_df = df1.join(df2, lsuffix='_df1', rsuffix='_df2')


# Step 4: Output the joined DataFrame
print(joined_df)
