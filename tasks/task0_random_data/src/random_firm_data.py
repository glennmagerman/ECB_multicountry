import pandas as pd
import numpy as np
import os
from common.utilities import macros_from_config

def random_annual_acc(start, end, nfirms, tmp_path, output_path):

    # Annual accounts - firm-level data
    for t in range(start, end + 1):
        df = pd.DataFrame(index=range(1, nfirms + 1))
        df['year'] = t
        df['vat'] = df.index
        
        for var in ['turnover', 'inputs_total']:
            df[var] = np.floor(np.exp(np.random.normal(size=nfirms) * 5) + 1) + 1000 # draw from log-normal distribution 
        
        df['nace'] = 100 + np.random.randint(0, 9601, size=nfirms) # assign a random NACE sector to each firm
        df['nace'] = np.floor(df['nace'] / 100) * 100 # restrict num of potential NACE codes
        
        df.to_csv(os.path.join(tmp_path, f'firms_{t}.csv'), index=False)

    # Collect data for a balanced panel
    firms_panel = pd.concat([pd.read_csv(os.path.join(tmp_path, f'firms_{t}.csv')) for t in range(start, end + 1)])

    # convert to string variables
    firms_panel['vat'] = firms_panel['vat'].astype(str)
    firms_panel['nace'] = firms_panel['nace'].astype(int).astype(str)
    firms_panel['nace'] = firms_panel['nace'].apply(lambda x: '0' + x if len(x) < 4 else x) # put a 0 before the sectors with NACE code < 1000
        
    firms_panel.to_csv(os.path.join(output_path, f'firm_data_{start}_{end}.csv'), index=False)

def create_random_firm_data(abs_path, tmp_path, output_path):

    start, end, nfirms, nlinks = macros_from_config(abs_path)

    random_annual_acc(start, end, nfirms, tmp_path, output_path)