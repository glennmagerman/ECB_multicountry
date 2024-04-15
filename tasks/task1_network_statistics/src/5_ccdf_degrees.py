import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw

import os 
abs_path = os.path.abspath(os.path.join('..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% INPUT
dict_data = load_workspace( os.path.join( 'tmp','init_data.pkl') )
B2B_df = dict_data['B2B_df']
firms_df = dict_data['firms_df']

# merge network and firm data
B2B_df = B2B_df.merge( firms_df[['year','vat','nace','corr_turnover']].rename(
    columns={'vat': 'vat_i', 'corr_turnover':'turnover'}), on=['vat_i','year'], how='inner' )

#%% 1. CCDF of degree distributions on the full panel

B2B_df_full = B2B_df.copy()
G = nx.from_pandas_edgelist(B2B_df_full, 'vat_i', 'vat_j', create_using=nx.DiGraph())

out_degree = {node: degree for node, degree in G.out_degree() if degree > 0}
in_degree = {node: degree for node, degree in G.in_degree() if degree > 0}

# CCDF of out-degree distribution
outdeg_distr = np.unique( np.array( list(out_degree.values()) ) )
ccdf_out = 1. - np.arange(1, len(outdeg_distr) + 1) / len(outdeg_distr) 

plt.figure(figsize=(10,6))
coeff = powerlaw.Fit(outdeg_distr).power_law.alpha
textstr = '\n'.join((
    f'Tail exponent:  {coeff:.4f}',
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.1, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

plt.loglog(outdeg_distr, ccdf_out, marker='.', linestyle='None')
plt.xlabel('Number of customers')
plt.ylabel('CCDF')
plt.grid(True)
plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 'outdeg_ccdf_full.png'), dpi=300 )

# CCDF of in-degree distribution
indeg_distr = np.unique( np.array( list(in_degree.values()) ) )
ccdf_in = 1. - np.arange(1, len(indeg_distr) + 1) / len(indeg_distr) 

plt.figure(figsize=(10,6))
coeff = powerlaw.Fit(indeg_distr).power_law.alpha
textstr = '\n'.join((
    f'Tail exponent: {coeff:.4f}',
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.1, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

plt.loglog(indeg_distr, ccdf_in, marker='.', linestyle='None')
plt.xlabel('Number of sellers')
plt.ylabel('CCDF')
plt.grid(True)
plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 'indeg_ccdf_full.png'), dpi=300 )


#%% 2. CCDF of degree distributions per year

for year in pd.unique(B2B_df['year']):
    df_year = B2B_df[B2B_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # Find out-degree and in-degree and add it to the dataframe
    out_degree = {node: degree for node, degree in G_year.out_degree() if degree > 0}
    in_degree = {node: degree for node, degree in G_year.in_degree() if degree > 0}
    
    # create path
    if not os.path.exists( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'CCDF') ):
        os.makedirs( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'CCDF') )
        
    # CCDF of out-degree distribution
    outdeg_distr = np.unique( np.array( list(out_degree.values()) ) )
    ccdf_out = 1. - np.arange(1, len(outdeg_distr) + 1) / len(outdeg_distr) 
    plt.figure(figsize=(10,6))
    coeff = powerlaw.Fit(outdeg_distr).power_law.alpha
    textstr = '\n'.join((
        f'Tail exponent: {coeff:.4f}',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.1, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.loglog(outdeg_distr, ccdf_out, marker='.', linestyle='None')
    plt.xlabel('Number of customers')
    plt.ylabel('CCDF')
    plt.grid(True)
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'CCDF', 'outdeg_ccdf_year.png'), dpi=300 )
    plt.close()

    # CCDF of in-degree distribution
    indeg_distr = np.unique( np.array( list(in_degree.values()) ) )
    ccdf_in = 1. - np.arange(1, len(indeg_distr) + 1) / len(indeg_distr) 
    plt.figure(figsize=(10,6))
    coeff = powerlaw.Fit(indeg_distr).power_law.alpha
    textstr = '\n'.join((
        f'Tail exponent: {coeff:.4f}',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.1, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.loglog(indeg_distr, ccdf_in, marker='.', linestyle='None')
    plt.xlabel('Number of sellers')
    plt.ylabel('CCDF')
    plt.grid(True)
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output',
                               f'{year}', 'CCDF', 'indeg_ccdf_year.png'), dpi=300 )
    plt.close()
    

