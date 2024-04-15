import pandas as pd
import numpy as np

import os 
import shutil
abs_path = os.path.abspath(os.path.join('..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

# Choose whether to use random or real data
"""
0: RANDOM DATA
1: REAL DATA
"""
flag = 1

if flag == 0:
    
    # copy the random data from the previous task in the input folder
    dir1 = os.path.join(abs_path, 'task0_random_data','output')
    target_dir = os.path.join(abs_path, 'task1_network_statistics','input') 
    for filename in os.listdir(dir1):
        src_path = os.path.join(dir1, filename)  # Source file path
        dest_path = os.path.join(target_dir, filename)  # Destination file path
        shutil.copy2(src_path, dest_path)  # Copy and replace file
    
    B2B_df = pd.read_stata( os.path.join(abs_path, 'task1_network_statistics','input','BE_network_2002_2022.dta') )
    firms_df = pd.read_stata( os.path.join(abs_path, 'task1_network_statistics','input','BE_annac_2002_2022.dta'))
    
elif flag == 1:
    # in this case the initial real data should be manually included in the input folder of task1
    B2B_df = pd.read_stata( os.path.join(abs_path, 'task1_network_statistics','input','B2B_data_pseudonymized.dta'))
    firms_df = pd.read_stata( os.path.join(abs_path, 'task1_network_statistics','input','firm_data_final.dta') )
    
# Save to pickles
data_dict = {'B2B_df': B2B_df, 'firms_df': firms_df}
save_workspace(data_dict, os.path.join( 'tmp','init_data.pkl'))