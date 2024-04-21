# Project: Multi-Country B2B Networks ECB
# Author: Alberto Palazzolo
# First version:  April 2024
# This version: April 2024

# NOTES:
# This is the master script of the code. The script calls the master files of
# each task, ensuring a smooth running of the codes of each task.

import os
abs_path = os.path.abspath('.')

import sys
sys.path.append( abs_path )

# 1. Choose real or random data
os.environ['data'] = 'random' # change to 'random' if using randomly-generated data of task0

# 2. Report the name (with the file extension) of the B2B and firm-level real data
# NOTE: supported extensions are .dta or .csv
os.environ['name_real_B2B'] = 'B2B_data_pseudonymized.dta'
os.environ['name_real_firm'] = 'firm_data_final.dta'

# 2. Execute subtasks
exec( open(os.path.join(abs_path, 'task0_random_data','src','_master_task0.py' )).read(), {"abs_path": abs_path})
exec( open(os.path.join(abs_path, 'task1_network_statistics','src','_master_task1.py' )).read(), {"abs_path": abs_path} )
    
