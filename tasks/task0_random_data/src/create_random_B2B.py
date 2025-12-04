import pandas as pd
import numpy as np
import os 
from common.utilities import macros_from_config

# Generate firm-to-firm data
def b2b_random(start, end, nfirms, nlinks, tmp_path, output_path):

    for t in range(start, end + 1):
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
        df.to_csv( os.path.join(tmp_path,f'network_{t}.csv'), index=False)

    # Collect all cross-sections and create panel
    panel = pd.concat([pd.read_csv(os.path.join(tmp_path,f'network_{t}.csv')) for t in range(start, end + 1)])

    # convert to string variables
    panel['vat_i'] = panel['vat_i'].astype(str)
    panel['vat_j'] = panel['vat_j'].astype(str)

    panel.to_csv(os.path.join(output_path,f'B2B_network_{start}_{end}.csv'), index=False)

def create_random_B2B(abs_path, tmp_path, output_path):

    start, end, nfirms, nlinks = macros_from_config(abs_path)

    b2b_random(start, end, nfirms, nlinks, tmp_path, output_path)