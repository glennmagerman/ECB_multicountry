import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.ticker import FuncFormatter, MaxNLocator
from linearmodels import PanelOLS

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

#%% 1. Compute distributions of total purchases and network purchases per year

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    
    # transform variable in logs
    df_year['ln_inputs_j'] = np.log(df_year['inputs_j'].replace(0,np.nan))
    df_year = df_year.dropna(subset=['ln_inputs_j']) # drop NaNs
    df_year['ln_network_inputs_j'] = np.log(df_year.groupby('vat_j')['sales_ij'].transform('sum'))
    
    ## 1a. Total purchases densities
    ln_inputs_j = np.array( df_year['ln_inputs_j'] )
    grid, kde_densities = kernel_density_plot(ln_inputs_j)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Total purchases')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output',
                              f'{year}', 'kernel_densities', 'inputs.png'), 
                dpi=300, bbox_inches='tight' )
    plt.close()
    
    # now de-mean variables
    df_year = df_year.groupby('nace_j').filter(lambda x: len(x) >= 5)
    linputs_dem_j = demean_variable_in_df('ln_inputs_j', 'nace_j', df_year)
    grid, kde_densities = kernel_density_plot(linputs_dem_j)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Total purchases, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'inputs_demeaned.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
    ## 1b. Network purchases densities
    ln_net_inputs_j = np.array( df_year['ln_network_inputs_j'] )
    grid, kde_densities = kernel_density_plot(ln_net_inputs_j)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Network purchases')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output',
                              f'{year}', 'kernel_densities', 'network_purchases.png'), 
                dpi=300, bbox_inches='tight' )
    plt.close()
    
    # now de-mean 
    df_year = df_year.groupby('nace_j').filter(lambda x: len(x) >= 5)
    lnet_inputs_dem_j = demean_variable_in_df('ln_network_inputs_j', 'nace_j', df_year)
    grid, kde_densities = kernel_density_plot(lnet_inputs_dem_j)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Network purchases, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'network_purch_demeaned.png'), dpi=300, bbox_inches='tight' )
    plt.close()

#%% 2. Compute distributions of total and network purchases per year by sector of the seller

industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction', 
            'Market services','Non-market services']

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    
    ## 2a. Plot turnover kernel density by industry
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        ln_inputs_j = np.array( np.log( df_year_sec['inputs_j'].replace(0,np.nan).dropna()) )
        grid, kde_densities = kernel_density_plot(ln_inputs_j)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
        
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Total purchases')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'inputs_bysec.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now with de-meaned variables
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        # now de-mean variables
        df_year = df_year.groupby('nace_j').filter(lambda x: len(x) >= 5)
        df_year_sec['ln_inputs_j'] = np.log(df_year_sec['inputs_j'].replace(0,np.nan))
        df_year_sec = df_year_sec.dropna(subset=['ln_inputs_j']) # drop NaNs
        linputs_dem_j = demean_variable_in_df('ln_inputs_j', 'nace_j', df_year_sec)
        grid, kde_densities = kernel_density_plot(linputs_dem_j)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
        
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Total purchases, demeaned')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'purchases_bysec_demeaned.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    ## 2b. Plot networks sales kernel density by industry
    df_year['ln_network_purch_j'] = np.log(df_year.groupby('vat_j')['sales_ij'].transform('sum'))
    
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        ln_net_purch_j = np.array( df_year_sec['ln_network_purch_j']  )
        grid, kde_densities = kernel_density_plot(ln_net_purch_j)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
        
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Network purchases')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'network_purch_bysec.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now with de-meaned variables
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        # now de-mean variables
        df_year = df_year.groupby('nace_j').filter(lambda x: len(x) >= 5)
        lnet_purch_dem_j = demean_variable_in_df('ln_network_purch_j', 'nace_j', df_year_sec)
        grid, kde_densities = kernel_density_plot(lnet_purch_dem_j)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
        
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Network purchases, demeaned')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'network_purch_bysec_demeaned.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
#%% 3. Construct summary tables for total inputs 

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()

    inputs_j_sum_bysec = []
    
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()

        inputs_moments_sec = calculate_distribution_moments( df_year_sec['inputs_j'].dropna() / 1000000 ) # report values in mln euros
        inputs_j_sum_bysec.append( inputs_moments_sec )
       
    inputs_j_moments_full = calculate_distribution_moments( df_year['inputs_j'].dropna() / 1000000)
    inputs_j_sum_bysec.append( inputs_j_moments_full )
    inputs_j_sum_bysec_tab = pd.DataFrame(inputs_j_sum_bysec, index=industries + ['All'])
    
    #Save table to csv
    inputs_j_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task2_network_statistics', 'output', 
                                          f'{year}', 'moments','purchases_bysec.csv') ) 