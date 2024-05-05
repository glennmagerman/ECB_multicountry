import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os 
if 'abs_path' not in globals():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path,'..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% INPUT
dict_data = load_workspace( os.path.join( abs_path, 'task2_network_statistics','input','init_data.pkl') )
full_df = dict_data['full_df']

#%% 1. Compute distribution of bilateral sales (sales_ij) per year

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    
    df_year['ln_sales_ij'] = np.log(df_year['sales_ij'])
    ln_sales_ij = np.array( df_year['ln_sales_ij'] )
    grid, kde_densities = kernel_density_plot(ln_sales_ij)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Network sales per customer')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output',
                              f'{year}', 'kernel_densities', 'net_sales_pc.png'), 
                dpi=300, bbox_inches='tight' )
    plt.close()
    
    # now de-mean variables
    df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5) # drop nace code if there are less than 5 firms with that code
    lsales_ij_dem = demean_variable_in_df('ln_sales_ij', 'nace_i', df_year)
    grid, kde_densities = kernel_density_plot(lsales_ij_dem)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Network sales per customer, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'net_sales_pc_demeaned.png'), dpi=300, bbox_inches='tight' )
    plt.close()

#%% 2. Compute distributions of total sales (turnover) and network sales per year

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    
    # transform variable in logs
    df_year['ln_turnover_i'] = np.log(df_year['turnover_i'])
    df_year['ln_network_sales_i'] = np.log(df_year.groupby('vat_i')['sales_ij'].transform('sum'))
    
    ## 2a. Total sales densities
    ln_turnover_i = np.array( df_year['ln_turnover_i'] )
    grid, kde_densities = kernel_density_plot(ln_turnover_i)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Total sales')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output',
                              f'{year}', 'kernel_densities', 'turnover.png'), 
                dpi=300, bbox_inches='tight' )
    plt.close()
    
    # now de-mean variables
    df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5)
    lturnover_dem_i = demean_variable_in_df('ln_turnover_i', 'nace_i', df_year)
    grid, kde_densities = kernel_density_plot(lturnover_dem_i)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Total sales, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'turnover_demeaned.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
    ## 2b. Network sales densities
    ln_net_sales_i = np.array( df_year['ln_network_sales_i'] )
    grid, kde_densities = kernel_density_plot(ln_net_sales_i)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Network sales')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output',
                              f'{year}', 'kernel_densities', 'network_sales.png'), 
                dpi=300, bbox_inches='tight' )
    plt.close()
    
    # now de-mean variables
    df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5)
    lnet_sales_dem_i = demean_variable_in_df('ln_network_sales_i', 'nace_i', df_year)
    grid, kde_densities = kernel_density_plot(lnet_sales_dem_i)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Network sales, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'network_sales_demeaned.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
#%% 3. Compute distributions of total and network sales per year by sector of the seller

industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction', 
            'Market services','Non-market services']

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    
    ## 3a. Plot turnover kernel density by industry
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        ln_turnover_i = np.array( np.log( df_year_sec['turnover_i'] ) )
        grid, kde_densities = kernel_density_plot(ln_turnover_i)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
    
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Total sales')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'turnover_bysec.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now with de-meaned variables
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        # now de-mean variables
        df_year_sec['ln_turnover_i'] = np.log(df_year_sec['turnover_i'])
        lturnover_dem_i = demean_variable_in_df('ln_turnover_i', 'nace_i', df_year_sec)
        grid, kde_densities = kernel_density_plot(lturnover_dem_i)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
    
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Total sales, demeaned')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_seller_bysec_demeaned.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    ## 3b. Plot networks sales kernel density by industry
    df_year['ln_network_sales_i'] = np.log(df_year.groupby('vat_i')['sales_ij'].transform('sum'))
    
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        ln_net_sales_i = np.array( df_year_sec['ln_network_sales_i']  )
        grid, kde_densities = kernel_density_plot(ln_net_sales_i)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
        
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Network sales')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'network_sales_bysec.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now with de-meaned variables
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        # now de-mean variables
        lnet_sales_dem_i = demean_variable_in_df('ln_network_sales_i', 'nace_i', df_year_sec)
        grid, kde_densities = kernel_density_plot(lnet_sales_dem_i)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
        
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Network sales, demeaned')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'network_sales_bysec_demeaned.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
#%% 4. Construct summary tables for turnover 

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()

    turnover_i_sum_bysec = []
    
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()

        turnover_moments_sec = calculate_distribution_moments( df_year_sec['turnover_i'] / 1000000 ) # report values in mln euros
        turnover_i_sum_bysec.append( turnover_moments_sec )
       
    turnover_i_moments_full = calculate_distribution_moments( df_year['turnover_i'] / 1000000 )
    turnover_i_sum_bysec.append( turnover_i_moments_full )
    turnover_i_sum_bysec_tab = pd.DataFrame(turnover_i_sum_bysec, index=industries + ['All'])
    
    #Save table to csv
    turnover_i_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task2_network_statistics', 'output', 
                                          f'{year}', 'moments','turnover_bysec.csv') )  