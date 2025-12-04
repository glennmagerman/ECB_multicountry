# Project: Multi-Country B2B Networks ECB
# Author: Alberto Palazzolo
# First version:  April 2024
# This version: November 2025

# NOTES:
# This is the master script of the code. The script calls the master files of
# each task, ensuring a smooth running of the codes of each task.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from task0_random_data.src._master_task0 import master_task0
from task1_sum_stats.src._master_task1 import master_task1
from task2_clean_data.src._master_task2 import master_task2
from task3_network_statistics.src._master_task3 import master_task3

def master_call():

    # Run all the master files of each task
    # Generate random data
    master_task0()

    # Summary statistics
    master_task1()
    
    # Clean the data and create panel
    master_task2()

    # Compute network statistics
    master_task3()

if __name__ == '__main__':
    master_call()