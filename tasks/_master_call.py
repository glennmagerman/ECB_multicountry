# Project: Multi-Country B2B Networks ECB
# Author: Alberto Palazzolo
# First version:  April 2024
# This version: September 2024

# NOTES:
# This is the master script of the code. The script calls the master files of
# each task, ensuring a smooth running of the codes of each task.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from task0_random_data.src._master_task0 import master_task0
from task1_clean_data.src._master_task1 import master_task1
from task2_network_statistics.src._master_task2 import master_task2

def master_call():

    # Run all the master files of each task
    # Generate random data
    master_task0()

    # Clean the data
    master_task1()

    # Compute network statistics
    master_task2()

if __name__ == '__main__':
    master_call()