#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:09:41 2023

@author: mateus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aqm = pd.read_csv('envcity_aqm_df.csv')
aqm.set_index('time', inplace=True)
aqm[(aqm > 50)] = np.nan
aqm[(aqm < 0)] = np.nan

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

ax_temp = ax.twinx()
ax_ref = ax_temp.twinx()


ax.set(ylabel = 'Tensão')
ax_temp.set(ylabel = 'Temperatura')
ax_ref.set(ylabel = 'Reference')

ax_temp.spines.right.set_position(('axes', 1.1))
ax_temp.tick_params(axis='y')


aqm['e2sp_ox_we'].loc['2023-03-18 10:00:00':'2023-03-22 10:00:00'].plot(marker = '.', linewidth = 0.1, ax=ax, markersize=2, label = 'AE')
aqm['e2sp_ox_ae'].loc['2023-03-18 10:00:00':'2023-03-22 10:00:00'].plot(marker = '.', linewidth = 0.1, ax=ax, markersize=2, label = 'WE')
aqm['e2sp_temp'].loc['2023-03-18 10:00:00':'2023-03-22 10:00:00'].plot(marker = '.', linewidth = 0.1, ax=ax_temp, markersize=2, label = 'Temperatura')
plt.legend()

ax.set_ylim([0, 0.8])
aqm['iag_o3'].loc['2023-03-18 10:00:00':'2023-03-22 10:00:00'].plot(marker = '.', markersize=1, linewidth = 0.1, color = 'b', ax=ax_ref, label = ' Ref')
plt.legend()