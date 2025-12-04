# Author: Alberto Palazzolo
# First version:  February 2024
# This version: September 2024

# NOTES:
# This is the master script of task2_clean_data. The script takes as input
# the raw data (either random or real) and gives as output the cleaned data
# that will be used in the following analyses.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from task2_clean_data.src.clean_firm_df import clean_firm_df
from task2_clean_data.src.clean_B2B_df import clean_B2B_df
from task2_clean_data.src.merge_and_clean_data import merge_and_clean_data
from task2_clean_data.src.create_panel import create_panel
from common.utilities import initialize_task, maintenance, copy_output_from_task
from common.load_data import load_data, extract_b2b_config, extract_config, extract_firm_data_config, extract_data_type

def master_task2():
    
    # 1. Define the absolute paths
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.abspath(os.path.join(abs_path, 'input'))
    tmp_path = os.path.abspath(os.path.join(abs_path, 'tmp'))
    output_path = os.path.abspath(os.path.join(abs_path, 'output'))

    # 2. Initialize task
    initialize_task(abs_path)
    
    config = extract_config(abs_path)
    data_type = extract_data_type(config)
    name_real_data_B2B, extension_real_B2B_data = extract_b2b_config(config)
    name_real_data_firm, extension_real_data_firm = extract_firm_data_config(config)
    #start, end = extract_start_end_years(abs_path)
    
    if data_type == 'real':
        copy_output_from_task(abs_path, 'raw_data', [f'{name_real_data_B2B}.{extension_real_B2B_data}', f'{name_real_data_firm}.{extension_real_data_firm}'])
    elif data_type == 'random':
        copy_output_from_task(abs_path, os.path.join('task0_random_data','output'), ['B2B_network_2002_2022.csv', 'firm_data_2002_2022.csv'])

    # 3. Execute subtasks
    # 3a. Load data
    B2B_df, firms_df = load_data(abs_path, input_path, tmp_path)

    # 3b. Cleaning firm data
    clean_firm_df(tmp_path, firms_df)

    # 3c. Cleaning B2B data
    clean_B2B_df(tmp_path, B2B_df)

    # 3d. Merge and clean the combined data
    merge_and_clean_data(tmp_path, output_path)
    
    # 3e. Create panel for analysis
    create_panel(tmp_path, output_path)

    # 4. Maintenance
    maintenance(abs_path)

if __name__ == '__main__':
    master_task2()
