import pandas as pd
import numpy as np

import os 
#abs_path = os.path.abspath('..')
abs_path = os.path.abspath(os.path.join('..','..'))

# Set seed for reproducibility (otherwise at every run the results change)
np.random.seed(8718354)

# 0. Macros
start, end = 2002, 2022
start_1 = start + 1
nfirms, nlinks = 1000, 10000

# Create directories if they don't exist
#os.makedirs( os.path.join(abs_path, 'tmp'), exist_ok=True)
#os.makedirs( os.path.join(abs_path, 'output'), exist_ok=True)

#%% 1. Firm-to-firm data
for t in range(start, end):
    df = pd.DataFrame(index=range(nlinks))
    df['year'] = t
    df['vat_i'] = np.floor((nfirms - 1) * np.random.uniform(size=nlinks) + 1) # draw sample from uniform distribution between 1 and 999
    df['vat_j'] = np.floor((nfirms - 1) * np.random.uniform(size=nlinks) + 1)
    df = df[df['vat_i'] != df['vat_j']]  # Drop potential self-loops
    df = df.drop_duplicates(subset=['vat_i', 'vat_j'])  # Don't allow for multiple edges in the same year
    df['sales_ij'] = 250 + np.exp(np.random.normal(size=len(df)) * 5 + 5) # sales values drawn from log-normal distribution
    
    # convert to string variables
    df['vat_i'] = df['vat_i'].astype(int).astype(str)
    df['vat_j'] = df['vat_j'].astype(int).astype(str)
    
    # save to tmp folder
    df.to_csv( os.path.join(abs_path, 'task0_random_data','tmp',f'network_{t}.csv'), index=False)

# Collect all cross-sections and create panel
panel = pd.concat([pd.read_csv(os.path.join(abs_path,'task0_random_data', 'tmp',f'network_{t}.csv')) for t in range(start, end + 1)])

# convert to string variables
panel['vat_i'] = panel['vat_i'].astype(str)
panel['vat_j'] = panel['vat_j'].astype(str)

panel.to_stata(os.path.join(abs_path, 'task0_random_data','output',f'BE_network_{start}_{end}.dta'), write_index=False)

#%% 2. Annual accounts + VAT declarations
for t in range(start, end):
    df = pd.DataFrame(index=range(1, nfirms + 1))
    df['year'] = t
    df['vat'] = df.index
    
    for var in ['turnover', 'inputs_total']:
        df[var] = np.floor(np.exp(np.random.normal(size=nfirms) * 5) + 1) + 1000 # draw from log-normal distribution 
    
    df['nace'] = 100 + np.random.randint(0, 9601, size=nfirms) # assign a random NACE sector to each firm
    df['nace'] = np.floor(df['nace'] / 10) * 10
    
    df.to_csv(os.path.join(abs_path, 'task0_random_data', 'tmp', f'firms_{t}.csv'), index=False)

#%% Collect data for a balanced panel
firms_panel = pd.concat([pd.read_csv(os.path.join(abs_path,'task0_random_data', 'tmp', f'firms_{t}.csv')) for t in range(start, end + 1)])

# convert to string variables
firms_panel['vat'] = firms_panel['vat'].astype(str)
firms_panel['nace'] = firms_panel['nace'].astype(int).astype(str)
firms_panel['nace'] = firms_panel['nace'].apply(lambda x: '0' + x if len(x) < 4 else x) # put a 0 before the sectors with NACE code < 1000
    
firms_panel.to_stata(os.path.join(abs_path,'task0_random_data', 'output', f'BE_annac_{start}_{end}.dta'), write_index=False)

