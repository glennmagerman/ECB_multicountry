import pandas as pd
import numpy as np
import os
from common.utilities import macros_from_config

def quarterly_accounts(start, end, nfirms, tmp_path, output_path):

    # Quarterly data generation with firm repetition across quarters
    for t in range(start, end + 1):
        # Create a DataFrame for all firms with the same year and NACE codes
        df_year = pd.DataFrame(index=range(1, nfirms + 1))
        df_year['year'] = t
        df_year['vat'] = df_year.index

        # Assign NACE sector randomly to each firm
        df_year['nace'] = 100 + np.random.randint(0, 9601, size=nfirms)
        df_year['nace'] = np.floor(df_year['nace'] / 100) * 100  # Restrict NACE codes

        # Create a DataFrame that repeats firms for each quarter
        df_year = df_year.loc[df_year.index.repeat(4)].reset_index(drop=True)  # Repeat each firm for 4 quarters

        # Add a quarter column (1 to 4) for each firm
        df_year['quarter'] = np.tile([1, 2, 3, 4], len(df_year) // 4)

        # Generate turnover, inputs total, and investment for each firm-quarter
        for var in ['turnover', 'inputs_total', 'investment']:
            df_year[var] = np.floor(np.exp(np.random.normal(size=len(df_year)) * 5) + 1) + 1000

        # Save each year's quarterly data to a CSV file in the tmp folder
        df_year.to_csv(os.path.join(tmp_path, f'firms_{t}_quarterly.csv'), index=False)

    # Collect all quarterly cross-sections and create a quarterly panel
    quarterly_firms_panel = pd.concat([
        pd.read_csv(os.path.join(tmp_path, f'firms_{t}_quarterly.csv'))
        for t in range(start, end + 1)
    ])

    # Convert variables to appropriate formats
    quarterly_firms_panel['vat'] = quarterly_firms_panel['vat'].astype(str)
    quarterly_firms_panel['nace'] = quarterly_firms_panel['nace'].astype(int).astype(str)
    quarterly_firms_panel['nace'] = quarterly_firms_panel['nace'].apply(lambda x: '0' + x if len(x) < 4 else x)  # Format NACE codes

    quarterly_firms_panel.to_csv(os.path.join(output_path, f'quarterly_firm_data_{start}_{end}.csv'), index=False)

def create_quarterly_firm_data(abs_path, tmp_path, output_path):

    start, end, nfirms, nlinks = macros_from_config(abs_path)

    quarterly_accounts(start, end, nfirms, tmp_path, output_path)
