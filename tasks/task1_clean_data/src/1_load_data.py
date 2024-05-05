import pandas as pd

import os 
import shutil
if 'abs_path' not in globals():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path,'..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% 1. Upload data

flag = os.getenv('data')
name_real_data_B2B = os.getenv('name_real_B2B')
name_real_data_firm = os.getenv('name_real_firm')
start, end = os.getenv('start_year'), os.getenv('end_year')

if flag == 'random':
    
    # copy the random data from the previous task in the input folder
    dir1 = os.path.join(abs_path, 'task0_random_data','output')
    target_dir = os.path.join(abs_path, 'task1_clean_data','input') 
    for filename in os.listdir(dir1):
        src_path = os.path.join(dir1, filename)  # Source file path
        dest_path = os.path.join(target_dir, filename)  # Destination file path
        shutil.copy2(src_path, dest_path)  # Copy and replace file
    
    B2B_df = pd.read_stata( os.path.join(abs_path, 'task1_clean_data','input',f'B2B_network_{start}_{end}.dta') )
    firms_df = pd.read_stata( os.path.join(abs_path, 'task1_clean_data','input',f'firm_data_{start}_{end}.dta'))
    
elif flag == 'real':
    
    # in this case the initial real data should be manually included in the input folder of task1
    B2B_df = read_data( os.path.join(abs_path, 'task1_clean_data','input',f'{name_real_data_B2B}'))
    firms_df = read_data( os.path.join(abs_path, 'task1_clean_data','input',f'{name_real_data_firm}') )
    

#%% 2. Save to pickles

data_dict = {'B2B_df': B2B_df, 'firms_df': firms_df}
save_workspace(data_dict, os.path.join( abs_path, 'task1_clean_data','tmp','input_data.pkl'))

