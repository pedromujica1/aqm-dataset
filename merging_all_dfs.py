#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:14:08 2023

@author: mateus
"""

import pandas as pd

# Step 1: Load the first dataset
df1 = pd.read_csv('e1_df.csv')  # Replace 'path/to/dataset1.csv' with the actual file path of your first dataset
df1.set_index('time', inplace=True)
# Step 2: Load the second dataset
df2 = pd.read_csv('e2_df.csv')  # Replace 'path/to/dataset2.csv' with the actual file path of your second dataset
df2.set_index('time', inplace=True)

df3 = pd.read_csv('e2sp_df.csv')  # Replace 'path/to/dataset2.csv' with the actual file path of your second dataset
df3.set_index('time', inplace=True)

df4 = pd.read_csv('iag_df.csv')
df4.set_index('time', inplace=True)

df5 = pd.read_csv('iag_df_complementar.csv')
df5.set_index('time', inplace=True)

# Step 3: Join the datasets side by side
joined_df = pd.concat([df1, df2, df3, df4, df5], axis=0, verify_integrity=False)
#joined_df.drop_duplicates(inplace=True)

for i in joined_df.dtypes:
    print(i)


joined_df.index = pd.DatetimeIndex(joined_df.index)
joined_df = joined_df.sort_values('time').groupby('time').agg('mean')
joined_df.to_csv('envcity_aqm_df_ok.csv', decimal='.')

joined_df['iag_co'].plot()
# Step 4: Output the joined DataFrame
print(joined_df)
