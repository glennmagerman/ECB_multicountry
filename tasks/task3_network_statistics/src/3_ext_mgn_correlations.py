import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import statsmodels.api as sm

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
    
#%% 1. Compute correlations between network sales vs outdegree

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # Find out-degree and add it to the dataframe
    out_degree = {node: degree for node, degree in G_year.out_degree()}
    df_year['outdeg'] = df_year['vat_i'].map(out_degree)
    
    # Create log-transformed variables
    df_year['ln_outdeg'] = np.log(df_year['outdeg'])
    df_year['ln_network_sales_i'] = np.log(df_year.groupby('vat_i')['sales_ij'].transform('sum'))
    
    # Estimate the FE model
    df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5)
    df_year['ln_network_sales_dem'] = demean_variable_in_df('ln_network_sales_i', 'nace_i', df_year)
    df_year['ln_outdeg_dem'] = demean_variable_in_df('ln_outdeg', 'nace_i', df_year)
    
    # regression on the (demeaned) underlying data
    df_year.set_index(['nace_i','year'],inplace=True)
    Y = df_year['ln_network_sales_dem']
    X = sm.add_constant(df_year[['ln_outdeg_dem']])  # Adds a constant term to the model
    model = PanelOLS(Y, X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    # aggregate data into 20 bins for network sales and calculate mean of ln_outdeg for each bin
    df_year['log_outdeg_dem_bin'] = pd.qcut(df_year['ln_outdeg_dem'], q=20, duplicates='drop')
    binned_data = df_year.groupby('log_outdeg_dem_bin').agg({
        'ln_network_sales_dem': 'mean',
        'ln_outdeg_dem': 'mean'
    }).reset_index()
    
    # Plotting
    plt.scatter(binned_data['ln_outdeg_dem'], binned_data['ln_network_sales_dem'])
    textstr = '\n'.join((
        f'Linear slope: {res.params["ln_outdeg_dem"]:.2f} ({res.std_errors["ln_outdeg_dem"]:.2f})',
        f'R-squared: {res.rsquared:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    set_ticks_log_scale(np.array(binned_data['ln_outdeg_dem']), step=2)
    set_ticks_log_scale(np.array(binned_data['ln_network_sales_dem']), step=2, axis='y')
    plt.ylabel('Network sales, demeaned')
    plt.xlabel('Number of customers, demeaned')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'correlations', 'outdeg_net_sales_dem.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
#%% 2. Compute correlations between network purchases vs indegree

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # Find out-degree and add it to the dataframe
    in_degree = {node: degree for node, degree in G_year.in_degree()}
    df_year['indeg'] = df_year['vat_j'].map(in_degree)
    
    # Create log-transformed variables
    df_year['ln_indeg'] = np.log(df_year['indeg'])
    df_year['ln_network_purch_j'] = np.log(df_year.groupby('vat_j')['sales_ij'].transform('sum'))
    
    # Estimate the FE model
    df_year = df_year.groupby('nace_j').filter(lambda x: len(x) >= 5)
    df_year['ln_network_purch_dem'] = demean_variable_in_df('ln_network_purch_j', 'nace_j', df_year)
    df_year['ln_indeg_dem'] = demean_variable_in_df('ln_indeg', 'nace_j', df_year)
    
    # regression on the (demeaned) underlying data
    df_year.set_index(['nace_j','year'],inplace=True)
    Y = df_year['ln_network_purch_dem']
    X = sm.add_constant(df_year[['ln_indeg_dem']])  # Adds a constant term to the model
    model = PanelOLS(Y, X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    # aggregate data into 20 bins for network sales and calculate mean of ln_outdeg for each bin
    df_year['log_indeg_dem_bin'] = pd.qcut(df_year['ln_indeg_dem'], q=20, duplicates='drop')
    binned_data = df_year.groupby('log_indeg_dem_bin').agg({
        'ln_network_purch_dem': 'mean',
        'ln_indeg_dem': 'mean'
    }).reset_index()
    
    # Plotting
    plt.scatter(binned_data['ln_indeg_dem'], binned_data['ln_network_purch_dem'])
    textstr = '\n'.join((
        f'Linear slope: {res.params["ln_indeg_dem"]:.2f} ({res.std_errors["ln_indeg_dem"]:.2f})',
        f'R-squared: {res.rsquared:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    set_ticks_log_scale(np.array(binned_data['ln_indeg_dem']), step=2)
    set_ticks_log_scale(np.array(binned_data['ln_network_purch_dem']), step=2, axis='y')
    plt.ylabel('Network purchases, demeaned')
    plt.xlabel('Number of suppliers, demeaned')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'correlations', 'outdeg_net_purch_dem.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
#%% 3. Compute correlations between outdegree and indegree

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    out_degree = {node: degree for node, degree in G_year.out_degree() }
    in_degree = {node: degree for node, degree in G_year.in_degree() if degree >0}
    df_year['outdeg'] = df_year['vat_i'].map(out_degree)
    df_year['indeg'] = df_year['vat_i'].map(in_degree)
    
    # transform variables in logs
    df_year['log_indeg'] = np.log(df_year['indeg'])
    df_year['log_outdeg'] = np.log(df_year['outdeg'])
    
    
    # de-mean the variables
    df_year = df_year.dropna(subset=['log_indeg']) # drop NaNs
    df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5)
    df_year['log_indeg_dem'] = demean_variable_in_df('log_indeg', 'nace_i', df_year)
    df_year['log_outdeg_dem'] = demean_variable_in_df('log_outdeg', 'nace_i', df_year)
    
    # run the regression to print the results on the plot
    df_year = df_year.set_index(['nace_i', 'year'])
    X = sm.add_constant(df_year['log_outdeg_dem'])  # Adds a constant term to the model
    model = PanelOLS(df_year['log_indeg_dem'], X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    # bin the data
    df_year['log_outdeg_dem_bin'] = pd.qcut(df_year['log_outdeg_dem'], q=20, duplicates='drop')
    binned_data = df_year.groupby('log_outdeg_dem_bin').agg({
        'log_indeg_dem': 'mean',
        'log_outdeg_dem': 'mean'
    }).reset_index()
    
    # Now plot the binned data
    plt.scatter(binned_data['log_outdeg_dem'], binned_data['log_indeg_dem'])
    set_ticks_log_scale(np.array(binned_data['log_outdeg_dem']), step=2)
    set_ticks_log_scale(binned_data['log_indeg_dem'], step=2,axis='y')
    textstr = '\n'.join((
        f'Linear slope: {res.params["log_outdeg_dem"]:.2f} ({res.std_errors["log_outdeg_dem"]:.2f})',
        f'R-squared: {res.rsquared:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.xlabel('Number of customers, demeaned')
    plt.ylabel('Number of suppliers, demeaned')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'correlations', 'indeg_outdeg.png'), dpi=300, bbox_inches='tight' )
    plt.close()






