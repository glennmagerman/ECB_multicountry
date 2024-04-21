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
dict_data = load_workspace( os.path.join( abs_path, 'task1_network_statistics','tmp','init_data.pkl') )
full_df = dict_data['full_df']

#%% 1. Calculate correlations on the full panel

B2B_df_full = full_df.copy()
G = nx.from_pandas_edgelist(B2B_df_full, 'vat_i', 'vat_j', create_using=nx.DiGraph())

# Find out-degree and in-degree and add them to the dataframe
out_degree = {node: degree for node, degree in G.out_degree() if degree > 0}
in_degree = {node: degree for node, degree in G.in_degree() if degree > 0}
B2B_df_full['outdeg'] = B2B_df_full['vat_i'].map(out_degree)
B2B_df_full['indeg'] = B2B_df_full['vat_i'].map(in_degree)

# Create log-transformed variables
B2B_df_full['ln_outdeg'] = np.log(B2B_df_full['outdeg'])
B2B_df_full['ln_indeg'] = np.log(B2B_df_full['indeg'])
B2B_df_full['ln_turnover'] = np.log(B2B_df_full['turnover_i'])

# Estimate the FE model
B2B_df_full.set_index(['nace_i','year'],inplace=True)

# Regression with out-degree
B2B_df_out = B2B_df_full.dropna(subset=['ln_outdeg', 'ln_turnover']) # drop NaN values
Y = B2B_df_out['ln_outdeg']
X = sm.add_constant(B2B_df_out[['ln_turnover']])  # Adds a constant term to the model
model = PanelOLS(Y, X, entity_effects=True, time_effects=True)
res = model.fit(cov_type='robust')

print(res)

# Extracting relevant statistics
params = res.params
stderr = res.std_errors
tvalues = res.tstats
pvalues = res.pvalues
conf_lower, conf_upper = res.conf_int().T.values # Extracting confidence intervals

results_df = pd.DataFrame({
    'Coefficient': params,
    'Std. Error': stderr,
    't-value': tvalues,
    'p-value': pvalues,
    '95% Conf. Interval Lower': conf_lower,
    '95% Conf. Interval Upper': conf_upper
})
results_df.to_csv( os.path.join(abs_path, 'task1_network_statistics', 'output','regression_results_outdeg.csv'))

# Regression with in-degree
B2B_df_in = B2B_df_full.dropna(subset=['ln_indeg', 'ln_turnover']) # drop NaN values
Y = B2B_df_in['ln_indeg']
X = sm.add_constant(B2B_df_in[['ln_turnover']])  # Adds a constant term to the model
model = PanelOLS(Y, X, entity_effects=True, time_effects=True)
res = model.fit(cov_type='robust')

print(res)

# Extracting relevant statistics
params = res.params
stderr = res.std_errors
tvalues = res.tstats
pvalues = res.pvalues
conf_lower, conf_upper = res.conf_int().T.values # Extracting confidence intervals

results_df = pd.DataFrame({
    'Coefficient': params,
    'Std. Error': stderr,
    't-value': tvalues,
    'p-value': pvalues,
    '95% Conf. Interval Lower': conf_lower,
    '95% Conf. Interval Upper': conf_upper
})
results_df.to_csv( os.path.join(abs_path, 'task1_network_statistics', 'output','regression_results_indeg.csv'))

#%% 2. Calculate correlations for each year (FE at the sector level)

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # Find out-degree and in-degree and add it to the dataframe
    out_degree = {node: degree for node, degree in G_year.out_degree() if degree > 0}
    in_degree = {node: degree for node, degree in G_year.in_degree() if degree > 0}
    df_year['outdeg'] = df_year['vat_i'].map(out_degree)
    df_year['indeg'] = df_year['vat_i'].map(in_degree)
    
    # Create log-transformed variables
    df_year['ln_outdeg'] = np.log(df_year['outdeg'])
    df_year['ln_indeg'] = np.log(df_year['indeg'])
    df_year['ln_turnover'] = np.log(df_year['turnover_i'])
    
    # Estimate the FE model
    df_year.set_index(['nace_i','year'],inplace=True)
    
    # Regression with out-degree
    df_year_out = df_year.copy().dropna(subset=['ln_outdeg', 'ln_turnover']) # drop NaN values
    Y = df_year_out['ln_outdeg']
    X = sm.add_constant(df_year_out[['ln_turnover']])  # Adds a constant term to the model
    model = PanelOLS(Y, X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    # Aggregating data into 20 bins for ln_turnover and calculating mean ln_outdeg for each bin
    bins = np.linspace(df_year_out['ln_turnover'].min(), df_year_out['ln_turnover'].max(), 21)
    df_year_out['bin'] = pd.cut(df_year_out['ln_turnover'], bins, labels=False)
    
    binned_data = df_year_out.groupby('bin')[['ln_turnover', 'ln_outdeg']].mean().reset_index()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(binned_data['ln_turnover'], binned_data['ln_outdeg'], color='blue')
    
    # Adding regression coefficients and standard deviations
    textstr = '\n'.join((
        f'Linear slope: {res.params["ln_turnover"]:.4f} ({res.std_errors["ln_turnover"]:.4f})',
        f'R-squared: {res.rsquared:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    # create path
    if not os.path.exists( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'correlations') ):
        os.makedirs( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'correlations') )
    
    plt.xlabel('Network sales, demeaned')
    plt.ylabel('Number of customers, demeaned')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'correlations', 'outdeg_sales.png'), dpi=300 )
    plt.close()
    
    # Regression with in-degree
    df_year_in = df_year.copy().dropna(subset=['ln_indeg', 'ln_turnover']) # drop NaN values
    Y = df_year_in['ln_indeg']
    X = sm.add_constant(df_year_in[['ln_turnover']])  # Adds a constant term to the model
    model = PanelOLS(Y, X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    # Aggregating data into 20 bins for ln_turnover and calculating mean ln_indeg for each bin
    bins = np.linspace(df_year_in['ln_turnover'].min(), df_year_in['ln_turnover'].max(), 21)
    df_year_in['bin'] = pd.cut(df_year_in['ln_turnover'], bins, labels=False)
    
    binned_data = df_year_in.groupby('bin')[['ln_turnover', 'ln_indeg']].mean().reset_index()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(binned_data['ln_turnover'], binned_data['ln_indeg'], color='blue')
    
    # Adding regression coefficients and standard deviations
    textstr = '\n'.join((
        f'Linear slope: {res.params["ln_turnover"]:.4f} ({res.std_errors["ln_turnover"]:.4f})',
        f'R-squared: {res.rsquared:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    plt.xlabel('Network sales, demeaned')
    plt.ylabel('Number of sellers, demeaned')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'correlations', 'indeg_sales.png'), dpi=300 )
    plt.close()
        
