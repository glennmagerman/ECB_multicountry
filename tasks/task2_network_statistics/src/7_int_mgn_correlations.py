import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
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
dict_data = load_workspace( os.path.join(abs_path, 'task2_network_statistics', 'input','init_data.pkl') )
full_df = dict_data['full_df']

#%% 1. Correlation between average network sales per customer and outdegree

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    out_degree = {node: degree for node, degree in G_year.out_degree() }
    df_year['outdeg'] = df_year['vat_i'].map(out_degree)
    df_year['network_sales_i'] = df_year.groupby('vat_i')['sales_ij'].transform('sum')
    
    # de-mean the log outdegree
    df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5)
    df_year['log_outdeg'] = np.log(df_year['outdeg'])
    df_year['log_outdeg_dem'] = demean_variable_in_df('log_outdeg', 'nace_i', df_year)
    
    # de-mean the log turnover
    df_year['log_sales_pc_i'] = np.log(df_year['network_sales_i'] / df_year['outdeg'])
    df_year['lsales_pc_i_dem'] = demean_variable_in_df('log_sales_pc_i', 'nace_i', df_year)
    
    # plot correlation with bin scatter
    df_year['log_outdeg_dem_bin'] = pd.qcut(df_year['log_outdeg_dem'], q=20, duplicates='drop')
    binned_data = df_year.groupby('log_outdeg_dem_bin').agg({
        'lsales_pc_i_dem': 'mean',
        'log_outdeg_dem': 'mean'
    }).reset_index()
    
    # run the regression to print the results on the plot
    df_year = df_year.set_index(['nace_i', 'year'])
    X = sm.add_constant(df_year['log_outdeg_dem'])  # Adds a constant term to the model
    model = PanelOLS(df_year['lsales_pc_i_dem'], X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    # Now plot the binned data
    plt.scatter(binned_data['log_outdeg_dem'], binned_data['lsales_pc_i_dem'])
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    set_ticks_log_scale(binned_data['log_outdeg_dem'], step=2)
    set_ticks_log_scale(binned_data['lsales_pc_i_dem'], step=1, axis='y')
    textstr = '\n'.join((
        f'Linear slope: {res.params["log_outdeg_dem"]:.2f} ({res.std_errors["log_outdeg_dem"]:.2f})',
        f'R-squared: {res.rsquared:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.xlabel('Number of customers, demeaned')
    plt.ylabel('Average sales per customer, demeaned')
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', 
                              f'{year}', 'correlations', 'avg_sales_pc_outdeg.png'), dpi=300, bbox_inches='tight' )
    plt.close()
    
