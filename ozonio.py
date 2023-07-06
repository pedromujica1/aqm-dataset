#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 18:09:41 2023

@author: mateus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alphasense_b_sensors.alphasense_sensors import *

import matplotlib.style as mplstyle
# mplstyle.use('fast')

aqm = pd.read_csv('envcity_aqm_df.csv')
aqm.set_index('time', inplace=True)
aqm.index = pd.to_datetime(aqm.index)
aqm = aqm.loc['2023-06-01 16:41:00':'2023-06-23 12:00:00']

aqm = aqm[['e2sp_ox_we', 'e2sp_ox_ae', 'e2sp_temp', 'iag_o3', 'e2sp_no2_we', 'e2sp_no2_ae', 'e2sp_co_we', 'iag_co']]
aqm[(aqm > 50)] = np.nan
aqm[(aqm < 0)] = np.nan
aqm = aqm.dropna(axis=0)

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

ax_ref = ax.twinx()
ax_temp = ax.twinx()

ax.set(ylabel = 'Tensão')
ax_temp.set(ylabel = 'Temperatura')
ax_ref.set(ylabel = 'Reference')

ax_temp.spines.right.set_position(('axes', 1.1))
ax_temp.tick_params(axis='y')

ox = Alphasense_Sensors("OX-B431", "204240461")
no2 = Alphasense_Sensors("NO2-B43F", "202742056")

ox_ppb, _, _, _ = ox.all_algorithms(raw_we=1000*aqm['e2sp_ox_we'], raw_ae=1000*aqm['e2sp_ox_ae'], temp=aqm['e2sp_temp'])
# ox_ppb = aqm['e2sp_ox_we'].to_numpy() - 0.3
print(ox_ppb.shape)
aqm['e2sp_ox'] = ox_ppb
aqm['e2sp_ox_we'].plot(marker = '.', linewidth = 0.1, ax=ax, markersize=2, label = 'AE')
aqm['e2sp_ox_ae'].plot(marker = '.', linewidth = 0.1, ax=ax, markersize=2, label = 'WE')
aqm['e2sp_temp'].plot(marker = '.', linewidth = 0.1, ax=ax_temp, markersize=2, label = 'Temperatura', color='gray')

ax.set_ylim([0, 0.8])
aqm['iag_o3'].plot(marker = '.', markersize=1, linewidth = 0.1, color = 'b', ax=ax_ref, label = 'Ref')
aqm['e2sp_ox'].plot(marker = '.', markersize=1, linewidth = 0.1, color = 'g', ax=ax_ref, label = 'EnvCity')

# ax.yaxis.label.set_color(p1.get_color())
# twin1.yaxis.label.set_color(p2.get_color())
ax_temp.yaxis.label.set_color(ax_temp.get_lines()[0].get_color())
ax_temp.tick_params(axis='y', colors=ax_temp.get_lines()[0].get_color())

# ax.tick_params(axis='y', colors=p1.get_color())
# twin1.tick_params(axis='y', colors=p2.get_color())
# twin2.tick_params(axis='y', colors=p3.get_color())

hzin = []
lzin = []
h, l = ax_ref.get_legend_handles_labels()
hzin += h
lzin += l
h, l = ax.get_legend_handles_labels()
hzin += h
lzin += l
h, l = ax_temp.get_legend_handles_labels()
hzin += h
lzin += l

ax.legend(hzin, lzin, markerscale=8)
plt.gcf().autofmt_xdate()
plt.show()



no2_ppb, _, _, _ = no2.all_algorithms(raw_we=1000*aqm['e2sp_no2_we'], raw_ae=1000*aqm['e2sp_no2_ae'], temp=aqm['e2sp_temp'])
no2_ppb[no2_ppb < 0] = 0

# fig, ax = plt.subplots()
# fig.subplots_adjust(right=0.75)
# aqm['e2sp_no2'] = no2_ppb
# aqm['e2sp_no2_we'].plot(ax=ax)
# aqm['e2sp_no2_ae'].plot(ax=ax)
# aqm['e2sp_no2'].plot(ax=ax.twinx())


fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

ox_ppb2, _, _, _ = ox.all_algorithms(raw_we=1000*aqm['e2sp_ox_we'] - no2_ppb*ox.no2_sensitivity/1000, raw_ae=1000*aqm['e2sp_ox_ae'], temp=aqm['e2sp_temp'])

aqm['e2sp_ox2'] = ox_ppb2
# aqm['e2sp_ox2'].plot(ax=ax, marker = '.', markersize = 2, linewidth=0.5, label = 'OX retirando NO2')
# aqm['e2sp_ox'].plot(ax=ax, marker = '.', markersize = 2, linewidth=0.5, label = 'OX + NO2')
# aqm['iag_o3'].plot(ax=ax, marker = '.', markersize = 2, linewidth=0.5, label ='Referencia')''
# plt.plot(aqm['e2sp_ox2'].index, aqm['e2sp_ox2'].values)
# plt.plot(aqm['iag_o3'].index + pd.Timedelta(hours=3), aqm['iag_o3'].values)
# plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
# plt.gcf().autofmt_xdate()
# ax.set_ylim(0, 220)
# ax.legend()
#%%
from envcity_plot_lib import *
from sklearn.metrics import r2_score
from scipy import stats


plt.plot(aqm['e2sp_co_we'].index, aqm['e2sp_co_we'].values + 1.5)
plt.plot(aqm['iag_co'].index + pd.Timedelta(hours=0), aqm['iag_co'].values)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))
plt.gcf().autofmt_xdate()

# e1 = {'ox' : aqm['e2sp_ox2']}
# e2 = {'ox' : aqm['iag_o3']}

# plot_data_by_time_and_regr_plot(e1, e2, labels=['ox'], latex_labels=['O_x'])

#%%