# Author: Alberto Palazzolo
# First version:  February 2024
# This version: April 2024

# NOTES:
# This is the master script of task0_random_data. The script generates random data
# for the analysis, which are analysed faster than real data.

import shutil
import os
if 'abs_path' not in globals():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path,'..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

# Initialize task
folders = ['tmp','output']
for folder in folders:
    if os.path.exists( os.path.join(abs_path,'task0_random_data',folder) ):
        shutil.rmtree( os.path.join(abs_path,'task0_random_data',folder) )

# Create task folders
folders = ['tmp','output','input']
for folder in folders:
    if not os.path.exists( os.path.join(abs_path,'task0_random_data', folder) ):
        os.makedirs( os.path.join(abs_path,'task0_random_data',folder) )
        print(f"Folder 'task0_random_data/{folder}' created successfully.")
    else:
        print(f"Folder 'task0_random_data/{folder}' already exists.")
        
# Execute subtasks
exec( open(os.path.join(abs_path, 'task0_random_data','src','1_create_random_data.py' )).read() )

# Maintenance (remove temporary folder)
if os.path.exists( os.path.join(abs_path,'task0_random_data','tmp') ):
    shutil.rmtree( os.path.join(abs_path,'task0_random_data','tmp') )

