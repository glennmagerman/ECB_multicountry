# Author: Alberto Palazzolo
# First version:  February 2024
# This version: November 2025

# NOTES:
# This is the master script of task3_network_statistics. The script takes as input
# the data cleaned in task2 and computes various statistics.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import pandas as pd
from task3_network_statistics.src.ext_mgn_correlations import master_ext_mgn_correlations
from task3_network_statistics.src.ccdf import master_CCDF
from task3_network_statistics.src.distributions import master_distributions
from task3_network_statistics.src.coefficients_of_variation import master_cv
from task3_network_statistics.src.var_decomposition import master_var_decomp
from task3_network_statistics.src.monpol import master_monpol
from common.utilities import initialize_task, maintenance, copy_output_from_task, create_folders_for_years, extract_start_end_years
from common.load_data import extract_config

def master_task3():
    
    # 1. Define the absolute paths
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.abspath(os.path.join(abs_path, 'input'))
    #tmp_path = os.path.abspath(os.path.join(abs_path, 'tmp'))
    output_path = os.path.abspath(os.path.join(abs_path, 'output'))

    # 2. Initialize task
    # 2a. Create task folders
    initialize_task(abs_path)

    # 2b. Create folders for storing output for each year
    create_folders_for_years(abs_path, output_path)

    # 3. Copy output from previous task
    copy_output_from_task(abs_path, os.path.join('task2_clean_data','output'), ['full_data_cleaned.parquet', 'panel.parquet'])
    copy_output_from_task(abs_path, os.path.join('raw_data'), ['gdp_data.csv', 'irfs_int_up_quarterly.csv',  'irfs_p_int_up_quarterly.csv',
                                                                'ses_int_up_quarterly.csv',  'ses_p_int_up_quarterly.csv',  'vcov_bs_int_up_quarterly.csv',
                                                                'vcov_p_bs_int_up_quarterly.csv'])

    # 4. Load the data
    full_df = pd.read_parquet(os.path.join(input_path, 'full_data_cleaned.parquet'))
    panel_df = pd.read_parquet(os.path.join(input_path, 'panel.parquet'))
    gdp_df = pd.read_csv(os.path.join(input_path, 'gdp_data.csv'))
    
    # 5. Extract the years that need to be analyzed
    config = extract_config(abs_path)
    country = config['country']
    start, end = extract_start_end_years(abs_path)
    
    # 6. Analysis
    # 6a. Correlations
    master_ext_mgn_correlations(panel_df, gdp_df, output_path, start, end, country)
    
    # 6b. CCDF
    master_CCDF(full_df, panel_df, output_path, start, end, country)
    
    # 6c. Distributions
    master_distributions(panel_df, output_path, start, end, country)
    
    # 6d. Coefficient of variations
    master_cv(full_df, output_path, start, end, country)
    
    # 6e. Variance decomposition
    master_var_decomp(full_df, panel_df, output_path, start, end, country)
    
    # 6f. Monetary policy
    master_monpol(panel_df, input_path, output_path, country)

    # 7. Maintenance
    maintenance(abs_path)

if __name__ == '__main__':
    master_task3()







