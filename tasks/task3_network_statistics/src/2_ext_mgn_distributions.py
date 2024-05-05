import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
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

#%% 1. Calculate distributions and moments per year

for year in pd.unique( full_df['year'] ):
    df_year = full_df[full_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # Degrees
    out_degree = {node: degree for node, degree in G_year.out_degree() if degree > 0}  # Number of buyers per seller
    in_degree = {node: degree for node, degree in G_year.in_degree() if degree > 0} # Number of sellers per buyer
    
    # Plot kernel density of out-degree and in-degree distributions
    ## Out-degree plot
    log_outdeg = np.array( np.log(list(out_degree.values())) )
    grid, kde_densities = kernel_density_plot(log_outdeg)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Number of customers')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'out_degree.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
    # now de-mean variables
    df_year_out = df_year.copy()
    df_year_out['outdeg'] = df_year_out['vat_i'].map(out_degree)
    df_year_out['ln_outdeg'] = np.log(df_year_out['outdeg'])
    
    df_year_out = df_year_out.groupby('nace_i').filter(lambda x: len(x) >= 5)
    loutdeg_dem = demean_variable_in_df('ln_outdeg', 'nace_i', df_year_out)

    grid, kde_densities = kernel_density_plot(loutdeg_dem)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Number of customers, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'out_degree_demeaned.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
    ## In-degree plot
    log_indeg = np.array( np.log(list(in_degree.values())) )
    grid, kde_densities = kernel_density_plot(log_indeg)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Number of suppliers')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'in_degree.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
    # now de-mean variables
    df_year_in = df_year.copy()
    df_year_in['indeg'] = df_year_in['vat_j'].map(in_degree)
    df_year_in['ln_indeg'] = np.log(df_year_in['indeg'])
    
    df_year_in = df_year_in.groupby('nace_j').filter(lambda x: len(x) >= 5)
    lindeg_dem = demean_variable_in_df('ln_indeg', 'nace_j', df_year_in)
    
    grid, kde_densities = kernel_density_plot(lindeg_dem)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel('Number of suppliers, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'in_degree_demeaned.png'), dpi=300, bbox_inches='tight' )
    plt.close()

#%% 2. Calculate distributions and moments per year by sector of the seller

industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction', 
            'Market services','Non-market services']

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    ## 2a. Plot kernel density for out-degree by industry
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        log_outdeg = np.array( np.log(list(out_degree.values())) )
        grid, kde_densities = kernel_density_plot(log_outdeg)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
    
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Number of customers')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'out_degree_bysec.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now with de-meaned variables
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        df_year_sec['outdeg'] = df_year_sec['vat_i'].map(out_degree)
        df_year_sec['ln_outdeg'] = np.log(df_year_sec['outdeg'])
        
        # now de-mean variables
        df_year_sec = df_year_sec.groupby('nace_i').filter(lambda x: len(x) >= 5)
        loutdeg_dem = demean_variable_in_df('ln_outdeg', 'nace_i', df_year_sec)
        grid, kde_densities = kernel_density_plot(loutdeg_dem)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
    
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Number of customers, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'out_degree_demeaned_bysec.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
    ## 2b. Plot kernel density for in-degree by industry
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_j'].unique()
        in_degree = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        log_indeg = np.array( np.log(list(in_degree.values())) )
        grid, kde_densities = kernel_density_plot(log_indeg)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
    
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Number of suppliers')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task2_network_statistics', 'output', 
                             f'{year}', 'kernel_densities','in_degree_bysec.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now with de-meaned variables
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_j'].unique()
        in_degree = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        df_year_sec['indeg'] = df_year_sec['vat_j'].map(in_degree)
        df_year_sec['ln_indeg'] = np.log(df_year_sec['indeg'])
        
        # now de-mean variables
        df_year_sec = df_year_sec.groupby('nace_j').filter(lambda x: len(x) >= 5)
        lindeg_dem = demean_variable_in_df('ln_indeg', 'nace_j', df_year_sec)
        grid, kde_densities = kernel_density_plot(lindeg_dem)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
    
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel('Number of suppliers, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'in_degree_demeaned_bysec.png'), dpi=300, bbox_inches='tight' )
    plt.close()

#%% 3. Construct summary tables

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    ## 3a. Outdegree
    out_sum_bysec = []
    
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree_sec = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        out_moments_sec = calculate_distribution_moments(list(out_degree_sec.values()) )
        out_sum_bysec.append( out_moments_sec )
       
    out_degree_full = {node: degree for node, degree in G_year.out_degree() if degree > 0}
    out_moments_full = calculate_distribution_moments(list(out_degree_full.values()) )
    out_sum_bysec.append( out_moments_full )
    out_sum_bysec_tab = pd.DataFrame(out_sum_bysec, index=industries + ['All'])
    
    #Save table to csv
    out_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task2_network_statistics', 'output', 
                                          f'{year}', 'moments','out_degree_bysec.csv') )

    ## 3b. Indegree
    in_sum_bysec = []
    
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        in_degree_sec = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        in_moments_sec = calculate_distribution_moments(list(in_degree_sec.values()) )
        in_sum_bysec.append( in_moments_sec )
       
    in_degree_full = {node: degree for node, degree in G_year.in_degree() if degree > 0}
    in_moments_full = calculate_distribution_moments(list(in_degree_full.values()) )
    in_sum_bysec.append( in_moments_full )
    in_sum_bysec_tab = pd.DataFrame(in_sum_bysec, index=industries + ['All'])
    
    #Save table to csv
    in_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task2_network_statistics', 'output', 
                                          f'{year}', 'moments','in_degree_bysec.csv') )
    
