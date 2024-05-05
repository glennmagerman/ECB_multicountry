# Author: Alberto Palazzolo
# First version:  February 2024
# This version: April 2024

# NOTES:
# This is the master script of task2_network_statistics. The script takes as input
# the data cleaned in task1 and computes various measure of network statistics.

import shutil
import os
if 'abs_path' not in globals():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path,'..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% 1. Initialize task
folders = ['tmp','output']
for folder in folders:
    if os.path.exists( os.path.join(abs_path,'task2_network_statistics',folder) ):
        shutil.rmtree( os.path.join(abs_path,'task2_network_statistics',folder) )

# Create task folders
folders = ['tmp','output','input']
for folder in folders:
    if not os.path.exists( os.path.join(abs_path,'task2_network_statistics', folder) ):
        os.makedirs( os.path.join(abs_path,'task2_network_statistics',folder) )
        print(f"Folder 'task2_network_statistics/{folder}' created successfully.")
    else:
        print(f"Folder 'task2_network_statistics/{folder}' already exists.")
    
# Create folders for storing output for each year
start, end = int(os.getenv('start_year')), int(os.getenv('end_year'))

for year in range(start, end + 1):
    
    # create the 'year' folder
    if not os.path.exists( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}') ):
        os.makedirs( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}') )
        
    # create folder for kernel densities
    if not os.path.exists( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'kernel_densities') ):
        os.makedirs( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'kernel_densities') )
        
    # create folder for moments tables
    if not os.path.exists( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'moments') ):
        os.makedirs( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'moments') )
       
    # create folder for correlations
    if not os.path.exists( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'correlations') ):
        os.makedirs( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'correlations') )
        
    # create folder for CCDFs
    if not os.path.exists( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'CCDF') ):
        os.makedirs( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'CCDF') )

#%% 2. Copy output from previous task

dir1 = os.path.join(abs_path, 'task1_clean_data','output')
target_dir = os.path.join(abs_path, 'task2_network_statistics','input') 
for filename in os.listdir(dir1):
    src_path = os.path.join(dir1, filename)  # Source file path
    dest_path = os.path.join(target_dir, filename)  # Destination file path
    shutil.copy2(src_path, dest_path)  # Copy and replace file


#%% 3. Execute subtasks
exec( open(os.path.join(abs_path, 'task2_network_statistics','src','1_sum_stat.py' )).read() )
exec( open(os.path.join(abs_path, 'task2_network_statistics','src','2_ext_mgn_distributions.py' )).read() )
exec( open(os.path.join(abs_path, 'task2_network_statistics','src','3_ext_mgn_correlations.py' )).read() )
exec( open(os.path.join(abs_path, 'task2_network_statistics','src','4_ccdf_degrees.py' )).read() )
exec( open(os.path.join(abs_path, 'task2_network_statistics','src','5_int_mgn_distr_sales.py' )).read() )
exec( open(os.path.join(abs_path, 'task2_network_statistics','src','6_int_mgn_distr_purchases.py' )).read() )
exec( open(os.path.join(abs_path, 'task2_network_statistics','src','7_int_mgn_correlations.py' )).read() )

# Maintenance (remove temporary folder)
if os.path.exists( os.path.join(abs_path,'task2_network_statistics','tmp') ):
    shutil.rmtree( os.path.join(abs_path,'task2_network_statistics','tmp') )
