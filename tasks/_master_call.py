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
# NOTE: type 'random' if using randomly-generated data of task0
# type 'real' if using real data (to be added manually in folder task2/input)
os.environ['data'] = 'real' 

# 2. Report the name (with the file extension) of the B2B and firm-level real data
# NOTE: supported extensions are .dta or .csv
os.environ['name_real_B2B'] = 'B2B_data_pseudonymized.dta'
os.environ['name_real_firm'] = 'firm_data_final.dta'

# 3. Report the years that the real data cover
# NOTE: this range of years will be also used for creating the random data
os.environ['start_year'], os.environ['end_year'] = '2002', '2021'

# 4. Execute subtasks
exec( open(os.path.join(abs_path, 'task0_random_data','src','_master_task0.py' )).read(), {"abs_path": abs_path})
exec( open(os.path.join(abs_path, 'task1_clean_data','src','_master_task1.py' )).read(), {"abs_path": abs_path} )
exec( open(os.path.join(abs_path, 'task2_network_statistics','src','_master_task2.py' )).read(), {"abs_path": abs_path} )
    