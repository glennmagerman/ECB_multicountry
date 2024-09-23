import pandas as pd
import os 
from task2_network_statistics.src.utilities_task2 import plot_densities_sales_or_purchases, kernel_densities_bysec, summary_tables

#%% 1. Compute distributions of total purchases and network purchases per year
def tot_and_network_purch_year(full_df, output_path, start, end):

    plot_densities_sales_or_purchases(full_df, output_path, 'purchases', 'total', start, end)
    plot_densities_sales_or_purchases(full_df, output_path, 'purchases', 'network', start, end)

#%% 2. Compute distributions of total and network purchases per year by sector of the seller
def tot_and_network_purch_bysec(full_df, output_path, start, end):

    kernel_densities_bysec(full_df, output_path, 'purchases', start, end)

#%% 3. Construct summary tables for total inputs 

def summary_table_purchases_by_year(full_df, output_path, start, end):  

    summary_tables(full_df, output_path, 'purchases', start, end)

# Master function
def master_int_mgn_distr_purchases(full_df, output_path, start, end):
    
    # 1. Compute distributions of total purchases and network purchases per year
    tot_and_network_purch_year(full_df, output_path, start, end)
    
    # 2. Compute distributions of total and network purchases per year by sector of the seller
    tot_and_network_purch_bysec(full_df, output_path, start, end)
    
    # 3. Construct summary tables for total inputs
    summary_table_purchases_by_year(full_df, output_path, start, end)
    