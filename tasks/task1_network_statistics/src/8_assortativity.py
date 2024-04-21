import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from linearmodels import PanelOLS
from scipy.stats import gmean
import statsmodels.api as sm

import os 
if 'abs_path' not in globals():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path,'..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% INPUT
dict_data = load_workspace( os.path.join( abs_path, 'task1_network_statistics','tmp','init_data.pkl') )
full_df = dict_data['full_df']

#%% 1. Compute degree assortativity (as in Newman, 2003)
# (Pearson correlation coefficient between suppliers' and customers' degrees)

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    out_degree = {node: degree for node, degree in G_year.out_degree() }
    in_degree = {node: degree for node, degree in G_year.in_degree() }
    df_year['outdeg_i'] = df_year['vat_i'].map(out_degree).astype('float64')
    df_year['outdeg_j'] = df_year['vat_j'].map(out_degree).astype('float64')
    df_year['indeg_i'] = df_year['vat_i'].map(in_degree).astype('float64')
    df_year['indeg_j'] = df_year['vat_j'].map(in_degree).astype('float64')
    
    # compute degree assortativity coefficients
   
    out_out_assort = df_year['outdeg_i'].corr(df_year['outdeg_j'])
    out_in_assort = df_year['outdeg_i'].corr(df_year['indeg_j'])
    in_out_assort = df_year['indeg_i'].corr(df_year['outdeg_j'])
    in_in_assort = df_year['indeg_i'].corr(df_year['indeg_j'])
    
    # plot the out-out degree assortativity coefficient
    df_year['outdeg_i_bin'] = pd.qcut(df_year['outdeg_i'], q=100, duplicates='drop') # aggregate in bins
    binned_data = df_year.groupby('outdeg_i_bin').agg({
        'outdeg_i': 'mean',
        'outdeg_j': 'mean'
    }).reset_index()
    
    plt.scatter(binned_data['outdeg_j'], binned_data['outdeg_i'])
    plt.xlabel('Average number of customers of j')
    plt.ylabel('Average number of customers of i')
    
    textstr = '\n'.join((
        f'Degree assortativity: {out_out_assort:.4f}',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.55, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'correlations', 'out_out_assort.png'), dpi=300 )
    plt.close()
    
    # plot the out-in degree assortativity coefficient
    df_year['outdeg_i_bin'] = pd.qcut(df_year['outdeg_i'], q=100, duplicates='drop') # aggregate in bins
    binned_data = df_year.groupby('outdeg_i_bin').agg({
        'outdeg_i': 'mean',
        'indeg_j': 'mean'
    }).reset_index()
    
    plt.scatter(binned_data['indeg_j'], binned_data['outdeg_i'])
    plt.xlabel('Average number of suppliers of j')
    plt.ylabel('Average number of customers of i')
    
    textstr = '\n'.join((
        f'Degree assortativity: {out_in_assort:.4f}',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.55, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'correlations', 'out_in_assort.png'), dpi=300 )
    plt.close()
    
    # plot the in-out degree assortativity coefficient
    df_year['indeg_i_bin'] = pd.qcut(df_year['indeg_i'], q=100, duplicates='drop') # aggregate in bins
    binned_data = df_year.groupby('indeg_i_bin').agg({
        'indeg_i': 'mean',
        'outdeg_j': 'mean'
    }).reset_index()
    
    plt.scatter(binned_data['outdeg_j'], binned_data['indeg_i'])
    plt.xlabel('Average number of customers of j')
    plt.ylabel('Average number of suppliers of i')
    
    textstr = '\n'.join((
        f'Degree assortativity: {in_out_assort:.4f}',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.55, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'correlations', 'in_out_assort.png'), dpi=300 )
    plt.close()
    
    # plot the in-in degree assortativity coefficient
    df_year['indeg_i_bin'] = pd.qcut(df_year['indeg_i'], q=100, duplicates='drop') # aggregate in bins
    binned_data = df_year.groupby('indeg_i_bin').agg({
        'indeg_i': 'mean',
        'indeg_j': 'mean'
    }).reset_index()
    
    plt.scatter(binned_data['indeg_j'], binned_data['indeg_i'])
    plt.xlabel('Average number of suppliers of j')
    plt.ylabel('Average number of suppliers of i')
    
    textstr = '\n'.join((
        f'Degree assortativity: {in_in_assort:.4f}',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.55, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'correlations', 'in_in_assort.png'), dpi=300 )
    plt.close()


#%% 2. Compute downstream and upstream assortativity (as in Bernard et al. 2022)
# downstream assortativity: average number of suppliers of i's customers vs number of i's customers
# upstream assortativity: average number of customers of i's suppliers vs number of i's suppliers

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # calculate degrees
    out_degree = {node: degree for node, degree in G_year.out_degree()}
    in_degree = {node: degree for node, degree in G_year.in_degree()}
    df_year['outdeg'] = df_year['vat_i'].map(out_degree)
    df_year['indeg'] = df_year['vat_j'].map(in_degree)
    
    # calculate average indegree for each firm's customer
    avg_indegree_ij = {}
    for firm in pd.unique(df_year['vat_i']):
        # get the firm's customers
        customers = list(G_year.successors(firm))
        # calculate the indegrees of the firm's customers
        indegrees = [G_year.in_degree(customer) for customer in customers]
        # calculate the geometric mean of the indegrees
        gmean_indegree = gmean(indegrees)
        # store the result in the dictionary
        avg_indegree_ij[firm] = gmean_indegree
    df_year['avg_indegree_ij'] = df_year['vat_i'].map(avg_indegree_ij)
    
    # calculate average outdegree for each firm's supplier
    avg_outdegree_ji = {}
    for firm in pd.unique(df_year['vat_j']):
        # get the firm's suppliers
        suppliers = list(G_year.predecessors(firm))
        # calculate the indegrees of the firm's customers
        outdegrees = [G_year.out_degree(supplier) for supplier in suppliers]
        # calculate the geometric mean of the indegrees
        gmean_outdegree = gmean(outdegrees)
        # store the result in the dictionary
        avg_outdegree_ji[firm] = gmean_outdegree
    df_year['avg_outdegree_ji'] = df_year['vat_j'].map(avg_outdegree_ji)
    
    # 2a. Downstream assortativity
    df_year_seller = df_year.copy()
    df_year_seller = df_year_seller.set_index(['nace_i', 'year'])
    
    # de-mean the log outdegree
    df_year_seller['log_outdeg'] = np.log(df_year_seller['outdeg'])
    mod = PanelOLS.from_formula('log_outdeg ~ 1 + EntityEffects', df_year_seller)
    res = mod.fit()
    df_year_seller['log_outdeg_dem'] = np.array(res.resids) # extract residuals
    
    # de-mean the average indegree
    df_year_seller['log_avg_indeg_ij'] = np.log(df_year_seller['avg_indegree_ij'])
    mod = PanelOLS.from_formula('log_avg_indeg_ij ~ 1 + EntityEffects', df_year_seller)
    res = mod.fit()
    df_year_seller['log_avg_indeg_ij_dem'] = np.array(res.resids) # extract residuals
    
    df_year_seller['log_outdeg_dem_bin'] = pd.qcut(df_year_seller['log_outdeg_dem'], q=20, duplicates='drop')
    binned_data = df_year_seller.groupby('log_outdeg_dem_bin').agg({
        'log_avg_indeg_ij_dem': 'mean',
        'log_outdeg_dem': 'mean'
    }).reset_index()
    
    # run the regression to print the results on the plot
    X = sm.add_constant(df_year_seller['log_outdeg_dem'])  # Adds a constant term to the model
    model = PanelOLS(df_year_seller['log_avg_indeg_ij_dem'], X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    # Now plot the binned data
    plt.scatter(binned_data['log_outdeg_dem'], binned_data['log_avg_indeg_ij_dem'])
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    textstr = '\n'.join((
        f'Linear slope: {res.params["log_outdeg_dem"]:.2f} ({res.std_errors["log_outdeg_dem"]:.2f})',
        f'R-squared: {res.rsquared:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.6, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.xlabel('Number of customers, demeaned')
    plt.ylabel('Average number of suppliers, demeaned')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'correlations', 'downstream_assort.png'), dpi=300 )
    plt.close()
    
    # 2b. Upstream assortativity
    df_year_buyer = df_year.copy()
    df_year_buyer = df_year_buyer.set_index(['nace_j', 'year'])
    
    # de-mean the log outdegree
    df_year_buyer['log_indeg'] = np.log(df_year_buyer['indeg'])
    mod = PanelOLS.from_formula('log_indeg ~ 1 + EntityEffects', df_year_buyer)
    res = mod.fit()
    df_year_buyer['log_indeg_dem'] = np.array(res.resids) # extract residuals
    
    # de-mean the average indegree
    df_year_buyer['log_avg_outdeg_ji'] = np.log(df_year_buyer['avg_outdegree_ji'])
    mod = PanelOLS.from_formula('log_avg_outdeg_ji ~ 1 + EntityEffects', df_year_buyer)
    res = mod.fit()
    df_year_buyer['log_avg_outdeg_ji_dem'] = np.array(res.resids) # extract residuals
    
    df_year_buyer['log_indeg_dem_bin'] = pd.qcut(df_year_buyer['log_indeg_dem'], q=20, duplicates='drop')
    binned_data = df_year_buyer.groupby('log_indeg_dem_bin').agg({
        'log_avg_outdeg_ji_dem': 'mean',
        'log_indeg_dem': 'mean'
    }).reset_index()
    
    # run the regression to print the results on the plot
    X = sm.add_constant(df_year_buyer['log_indeg_dem'])  # Adds a constant term to the model
    model = PanelOLS(df_year_buyer['log_avg_outdeg_ji_dem'], X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    # Now plot the binned data
    plt.scatter(binned_data['log_indeg_dem'], binned_data['log_avg_outdeg_ji_dem'])
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    textstr = '\n'.join((
        f'Linear slope: {res.params["log_indeg_dem"]:.2f} ({res.std_errors["log_indeg_dem"]:.2f})',
        f'R-squared: {res.rsquared:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.6, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.xlabel('Number of suppliers, demeaned')
    plt.ylabel('Average number of customers, demeaned')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'correlations', 'upstream_assort.png'), dpi=300 )
    plt.close()
    
    
    
    
    
    
    
    
    
    
