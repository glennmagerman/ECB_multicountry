import numpy as np
from common.utilities import demean_variable_in_df, kernel_density_plot
from task2_network_statistics.src.utilities_task2 import plot_densities_sales_or_purchases, kernel_densities_bysec, summary_tables

#%% 1. Compute distribution of bilateral sales (sales_ij) per year

def distribution_bilateral_sales(full_df, output_path, start, end):

    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year].copy()
        
        df_year['ln_sales_ij'] = np.log(df_year['sales_ij'])
        ln_sales_ij = np.array( df_year['ln_sales_ij'] ) # convert to numpy array for kernel density estimation
        
        kernel_density_plot(ln_sales_ij, 'Network sales per customer', 'Density', 'net_sales_pc.png', output_path, year)
        
        # now de-mean variables
        df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5) # drop nace code if there are less than 5 firms with that code
        lsales_ij_dem = demean_variable_in_df('ln_sales_ij', 'nace_i', df_year)
        kernel_density_plot(lsales_ij_dem, 'Network sales per customer, demeaned', 'Density', 'net_sales_pc_demeaned.png', output_path, year)

#%% 2. Compute distributions of total sales (turnover) and network sales per year
def tot_and_network_sales_year(full_df, output_path, start, end):

    plot_densities_sales_or_purchases(full_df, output_path, 'sales', 'total', start, end)
    plot_densities_sales_or_purchases(full_df, output_path, 'sales', 'network', start, end)
    
#%% 3. Compute distributions of total and network sales per year by sector of the seller
def sales_plot_by_industry(full_df, output_path, start, end):

    kernel_densities_bysec(full_df, output_path, 'sales', start, end)

#%% 4. Construct summary tables for turnover 
def summary_tables_turnover(full_df, output_path, start, end):

    summary_tables(full_df, output_path, 'sales', start, end)

# Master function
def master_int_mgn_distr_sales(full_df, output_path, start, end):

    # 1. Compute distribution of bilateral sales (sales_ij) per year
    distribution_bilateral_sales(full_df, output_path, start, end)

    # 2. Compute distributions of total sales (turnover) and network sales per year
    tot_and_network_sales_year(full_df, output_path, start, end)

    # 3. Compute distributions of total and network sales per year by sector of the seller
    sales_plot_by_industry(full_df, output_path, start, end)

    # 4. Construct summary tables for turnover
    summary_tables_turnover(full_df, output_path, start, end)
    