import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from linearmodels import PanelOLS
import statsmodels.api as sm
import os 
from common.utilities import demean_variable_in_df, set_ticks_log_scale, aggregate_and_bin
#%% 1. Correlation between average network sales per customer and outdegree
def plot_binned_data(binned_data, res):

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

def regress_outdegree_vs_salespc(df_year):
    
    # regression on the (demeaned) underlying data
    df_year.set_index(['nace_i','year'],inplace=True)
    Y = df_year[f'lsales_pc_i_dem']
    X = sm.add_constant(df_year[[f'log_outdeg_dem']])  # Adds a constant term to the model
    model = PanelOLS(Y, X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    return res

def demean_log_outdegree(df_year):
    df_year['log_outdeg'] = np.log(df_year['outdeg'])
    df_year['log_outdeg_dem'] = demean_variable_in_df('log_outdeg', 'nace_i', df_year)
    
    return df_year

def demean_log_turnover_pc(df_year):
    df_year['log_sales_pc_i'] = np.log(df_year['network_sales_i'] / df_year['outdeg'])
    df_year['lsales_pc_i_dem'] = demean_variable_in_df('log_sales_pc_i', 'nace_i', df_year)
    
    return df_year

def corr_avg_sales_pc_outdeg(full_df, output_path, start, end):

    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year].copy()
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        out_degree = {node: degree for node, degree in G_year.out_degree() }
        df_year['outdeg'] = df_year['vat_i'].map(out_degree)
        df_year['network_sales_i'] = df_year.groupby('vat_i')['sales_ij'].transform('sum')
        
        # de-mean the log outdegree
        df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5)
        df_year = demean_log_outdegree(df_year)
        
        # de-mean the log turnover
        df_year = demean_log_turnover_pc(df_year)
        
        # plot correlation with bin scatter
        binned_data = aggregate_and_bin(df_year, 'log_outdeg_dem', 'lsales_pc_i_dem')
        
        # run the regression to print the results on the plot
        res = regress_outdegree_vs_salespc(df_year)
        
        # Now plot the binned data
        plot_binned_data(binned_data, res)
        plt.savefig( os.path.join(output_path, f'{year}', 'correlations', 'avg_sales_pc_outdeg.png'), dpi=300, bbox_inches='tight' )
        plt.close()
    
# Master function
def master_int_mgn_correlations(full_df, output_path, start, end):

    # Compute correlations between average network sales per customer and outdegree
    corr_avg_sales_pc_outdeg(full_df, output_path, start, end)
    