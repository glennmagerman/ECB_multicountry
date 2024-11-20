# Author: Alberto Palazzolo
# First version:  February 2024
# This version: September 2024

# NOTES:
# This is the master script of task2_network_statistics. The script takes as input
# the data cleaned in task1 and computes various measure of network statistics.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import pandas as pd
from task2_network_statistics.src.sum_stat import master_sum_stat
from task2_network_statistics.src.ext_mgn_distributions import master_ext_mgn_distributions
from task2_network_statistics.src.ext_mgn_correlations import master_ext_mgn_correlations
from task2_network_statistics.src.ccdf_degrees import master_CCDF_degrees
from task2_network_statistics.src.int_mgn_distr_sales import master_int_mgn_distr_sales
from task2_network_statistics.src.int_mgn_distr_purchases import master_int_mgn_distr_purchases
from task2_network_statistics.src.int_mgn_correlations import master_int_mgn_correlations
from task2_network_statistics.src.upstreamness import upstreamness
from common.utilities import initialize_task, maintenance, copy_output_from_previous_task, create_folders_for_years, extract_start_end_years

def master_task2():
    
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
    copy_output_from_previous_task(abs_path, 'task1_clean_data')

    # Load the cleaned data
    full_df = pd.read_parquet(os.path.join(input_path, 'full_data_cleaned.parquet'))

    # Extract the years that need to be analyzed
    start, end = extract_start_end_years(abs_path)

    # 4. Execute subtasks
    # 4a. General summary statistics
    master_sum_stat(full_df, output_path, start, end)

    # 4b. Extensive margin distributions
    master_ext_mgn_distributions(full_df, output_path, start, end)

    # 4c. Extensive margin correlations
    master_ext_mgn_correlations(full_df, output_path, start, end)

    # 4d. CCDF of degree distributions
    master_CCDF_degrees(full_df, output_path, start, end)

    # 4e. Intensive margin distributions of sales
    master_int_mgn_distr_sales(full_df, output_path, start, end)

    # 4f. Intensive margin distributions of purchases
    master_int_mgn_distr_purchases(full_df, output_path, start, end)

    # 4g. Intensive margin correlations
    master_int_mgn_correlations(full_df, output_path, start, end)

    # 4h. Upstreamness
    upstreamness(full_df, output_path, start, end)

    # 5. Maintenance
    maintenance(abs_path)

if __name__ == '__main__':
    master_task2()







