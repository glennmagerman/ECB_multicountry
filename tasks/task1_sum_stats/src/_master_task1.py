# Author: Alberto Palazzolo
# First version:  February 2024
# This version: September 2024

# NOTES:
# This is the master script of task1_sum_stats. The script takes as input
# the raw data and computes some summary statistics before cleaning the data.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from common.utilities import initialize_task, maintenance, copy_output_from_task
from common.load_data import load_data, extract_b2b_config, extract_config, extract_firm_data_config, extract_data_type
from task1_sum_stats.src.sum_stat import master_sum_stat

def master_task1():
    
    # 1. Define the absolute paths
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.abspath(os.path.join(abs_path, 'input'))
    tmp_path = os.path.abspath(os.path.join(abs_path, 'tmp'))
    output_path = os.path.abspath(os.path.join(abs_path, 'output'))

    # 2. Initialize task
    # 2a. Create task folders
    initialize_task(abs_path)
    
    # 3. Extract information from configuration file
    config = extract_config(abs_path)
    data_type = extract_data_type(config)
    name_real_data_B2B, extension_real_B2B_data = extract_b2b_config(config)
    name_real_data_firm, extension_real_data_firm = extract_firm_data_config(config)
    #start, end = extract_start_end_years(abs_path)
    
    if data_type == 'real':
        copy_output_from_task(abs_path, 'raw_data', [f'{name_real_data_B2B}.{extension_real_B2B_data}', f'{name_real_data_firm}.{extension_real_data_firm}'])
    elif data_type == 'random':
        copy_output_from_task(abs_path, 'task0_random_data/output', ['B2B_network_2002_2022.csv', 'firm_data_2002_2022.csv'])
    
    # 4. Execute subtasks
    B2B_df, firms_df = load_data(abs_path, input_path, tmp_path)
    
    master_sum_stat(B2B_df, firms_df, output_path)
    
    # 5. Maintenance
    maintenance(abs_path)

if __name__ == '__main__':
    master_task1()