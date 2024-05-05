# Author: Alberto Palazzolo
# First version:  February 2024
# This version: April 2024

# NOTES:
# This is the master script of task1_clean_data. The script takes as input
# the raw data (either random or real) and gives as output the cleaned data
# that will be used in the following analyses.

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
    if os.path.exists( os.path.join(abs_path,'task1_clean_data',folder) ):
        shutil.rmtree( os.path.join(abs_path,'task1_clean_data',folder) )

# Create task folders
folders = ['tmp','output','input']
for folder in folders:
    if not os.path.exists( os.path.join(abs_path,'task1_clean_data', folder) ):
        os.makedirs( os.path.join(abs_path,'task1_clean_data',folder) )
        print(f"Folder 'task1_clean_data/{folder}' created successfully.")
    else:
        print(f"Folder 'task1_clean_data/{folder}' already exists.")

#%% 2. Execute subtasks
exec( open(os.path.join(abs_path, 'task1_clean_data','src','1_load_data.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_clean_data','src','2_data_cleaning.py' )).read() )

#%% 3. Maintenance (remove temporary folder)
if os.path.exists( os.path.join(abs_path,'task1_clean_data','tmp') ):
    shutil.rmtree( os.path.join(abs_path,'task1_clean_data','tmp') )

