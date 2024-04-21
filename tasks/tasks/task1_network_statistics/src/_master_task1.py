# Author: Alberto Palazzolo
# First version:  February 2024
# This version: April 2024

# NOTES:
# This is the master script of task1_network_statistics. The script takes as input the raw
# data (either randomly generated in the previous task or the real data) and computes
# various measure of network statistics.

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
    if os.path.exists( os.path.join(abs_path,'task1_network_statistics',folder) ):
        shutil.rmtree( os.path.join(abs_path,'task1_network_statistics',folder) )

# Create task folders
for folder in folders:
    if not os.path.exists( os.path.join(abs_path,'task1_network_statistics', folder) ):
        os.makedirs( os.path.join(abs_path,'task1_network_statistics',folder) )
        print(f"Folder '{folder}' created successfully.")
    else:
        print(f"Folder '{folder}' already exists.")
        
if not os.path.exists( os.path.join(abs_path, 'task1_network_statistics','input') ):
    os.makedirs( os.path.join(abs_path, 'task1_network_statistics','input') )
    print("Folder 'input' created successfully.")
else:
    print("Folder 'input' already exists.")

# 4. Execute subtasks
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','1_load_data.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','2_sum_stat.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','3_ext_mgn_distributions.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','4_ext_mgn_correlations.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','5_ccdf_degrees.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','6_int_mgn_distributions.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','7_int_mgn_distributions_shares.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','8_assortativity.py' )).read() )
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','9_int_mgn_correlations.py' )).read() )

# Maintenance (remove temporary folder)
#if os.path.exists( os.path.join(abs_path,'task1_network_statistics','tmp') ):
#    shutil.rmtree( os.path.join(abs_path,'task1_network_statistics','tmp') )
