import pandas as pd
import os 
import yaml
import shutil
from common.utilities import extract_start_end_years

# Upload data

def extract_config(abs_path):

    # Load configuration from YAML file
    path_config_file = os.path.abspath(os.path.join(abs_path, '..', 'config', 'config.yaml'))
    with open(path_config_file, 'r') as file:
        config = yaml.safe_load(file)

    return config

def extract_data_type(config):
    
    # data type (if real or random)
    data_type = config['data_type']

    return data_type

def extract_b2b_config(config):

    # name and extension of B2B data
    name_real_data_B2B = config['b2b_data']['file_name']
    extension_real_data_B2B = config['b2b_data']['extension']
    
    return name_real_data_B2B, extension_real_data_B2B

def extract_firm_data_config(config):

    # name and extension of firm data
    name_real_data_firm = config['firm_data']['file_name']
    extension_real_data_firm = config['firm_data']['extension']
    
    return name_real_data_firm, extension_real_data_firm

def read_data(
        data_type: str, 
        abs_path, 
        input_path, 
        name_real_data_B2B: str, 
        extension_real_data_B2B: str, 
        name_real_data_firm: str, 
        extension_real_data_firm: str, 
        start: int, 
        end: int):

    if data_type == 'random':
        
        # copy the random data from the previous task in the input folder
        dir1 = os.path.abspath(os.path.join(abs_path, '..', 'task0_random_data','output'))
        target_dir = input_path
        for filename in os.listdir(dir1):
            src_path = os.path.join(dir1, filename)  # Source file path
            dest_path = os.path.join(target_dir, filename)  # Destination file path
            shutil.copy2(src_path, dest_path)  # Copy and replace file
        
        B2B_df = pd.read_csv( os.path.join(input_path,f'B2B_network_{start}_{end}.csv') )
        firms_df = pd.read_csv( os.path.join(input_path,f'firm_data_{start}_{end}.csv'))
        
    elif data_type == 'real':
        
        # in this case the initial real data should be manually included in the input folder of task1
        # B2B data
        if extension_real_data_B2B == 'csv':
            B2B_df = pd.read_csv( os.path.join(input_path,f'{name_real_data_B2B}.{extension_real_data_B2B}') )
        elif extension_real_data_B2B == 'dta':
            B2B_df = pd.read_stata( os.path.join(input_path,f'{name_real_data_B2B}.{extension_real_data_B2B}') )
        
        # firm data
        if extension_real_data_firm == 'csv':
            firms_df = pd.read_csv( os.path.join(input_path,f'{name_real_data_firm}.{extension_real_data_firm}') )
        elif extension_real_data_firm == 'dta':
            firms_df = pd.read_stata( os.path.join(input_path,f'{name_real_data_firm}.{extension_real_data_firm}') )
    
    return B2B_df, firms_df


# Master function
def load_data(abs_path, input_path, tmp_path):

    config = extract_config(abs_path)
    data_type = extract_data_type(config)
    name_real_data_B2B, extension_real_B2B_data = extract_b2b_config(config)
    name_real_data_firm, extension_real_data_firm = extract_firm_data_config(config)
    start, end = extract_start_end_years(abs_path)

    B2B_df, firms_df = read_data(data_type, abs_path, input_path, 
                                 name_real_data_B2B, extension_real_B2B_data, 
                                 name_real_data_firm, extension_real_data_firm, start, end)
    
    return B2B_df, firms_df


