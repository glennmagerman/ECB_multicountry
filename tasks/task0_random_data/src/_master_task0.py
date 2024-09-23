# Author: Alberto Palazzolo
# First version: February 2024
# This version: September 2024

# NOTES:
# This is the master script of task0_random_data. The script generates random data
# for the analysis, which are analysed faster than real data.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from task0_random_data.src.create_random_B2B import create_random_B2B
from task0_random_data.src.random_firm_data import create_random_firm_data
from task0_random_data.src.quarterly_firm_data import create_quarterly_firm_data
from common.utilities import initialize_task, maintenance
import numpy as np

def master_task0():

    # 1. Define the absolute paths
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.abspath(os.path.join(abs_path, 'input'))
    tmp_path = os.path.abspath(os.path.join(abs_path, 'tmp'))
    output_path = os.path.abspath(os.path.join(abs_path, 'output'))

    # 2. Initialize task
    initialize_task(abs_path)

    # Set seed for reproducibility (otherwise at every run the results change)
    np.random.seed(8718354)

    # 2a. Generate random B2B
    create_random_B2B(abs_path, tmp_path, output_path)

    # 2b. Generate random (annual) firm-level data
    create_random_firm_data(abs_path, tmp_path, output_path)

    # 2c. Generate random *quarterly* firm-level data (for monetary policy analysis)
    create_quarterly_firm_data(abs_path, tmp_path, output_path)

    # 3. Maintenance
    maintenance(abs_path)

if __name__ == '__main__':
    master_task0()

