import pandas as pd
import glob

# Step 1: Import the necessary libraries

# Step 2: Get a list of all the monthly datasets in a specific directory
file_list = glob.glob('E2-SP/*.csv')  # Replace 'path/to/monthly_datasets/' with the actual directory path containing your monthly datasets

# Step 3: Create an empty list to store the individual DataFrames
dfs = []

# Step 4: Iterate over the file list, read each CSV file as a DataFrame, and append it to the `dfs` list
for file in file_list:
    df = pd.read_csv(file, parse_dates=['time'])
    dfs.append(df)

# Step 5: Concatenate the individual DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Step 6: Filter the merged DataFrame and transform it into a new DataFrame where the data is placed by columns
pivoted_df = merged_df.pivot(index='time', columns='variable', values='value')
print(pivoted_df.columns)
pivoted_df.drop(labels = ['counter_up', 'datarate', 'delay', 'duplicate', 'encrypted_payload',
'fport', 'freq', 'gateway', 'gps_alt', 'gps_location', 'gps_time','hardware_chain', 'hardware_tmst',
'header_ack', 'header_adr','header_adr_ack_req', 'header_class_b', 'header_confirmed',
'header_type', 'header_version', 'location', 'modulation_bandwidth',
'modulation_coderate', 'modulation_spreading', 'modulation_type'], axis = 1, inplace=True)
#pivoted_df.drop(labels = ['parse_error', 'payload', 'field1', 'field2',
#'field3', 'field4', 'field5', 'field6', 'field7', 'field8', 'fport',
#'frequency', 'frm_payload', 'gateway_eui', 'application_id', 'device_id', 'timestamp'], axis = 1, inplace=True)

# Step 7: Transform all data in the pivoted DataFrame into numeric data
pivoted_df = pivoted_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
pivoted_df.index = pivoted_df.index.floor('min')
pivoted_df = pivoted_df.groupby(pivoted_df.index).first()

# Step 8: Resample the DataFrame with equally spaced index
pivoted_df = pivoted_df.resample('1T').asfreq()


# Step 11: Add a prefix of 'e1_' to each column
pivoted_df = pivoted_df.add_prefix('e2sp_')

# Step 12: Save the modified DataFrame to a file (e.g., CSV)
pivoted_df.to_csv('E2-SP/e2sp_df.csv', index=True)  # Replace 'path/to/save/modified_df.csv' with the actual file path where you want to save the DataFrame

# Step 9: Output the transformed DataFrame
print(pivoted_df)


# %%
