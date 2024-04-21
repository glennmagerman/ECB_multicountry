import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw

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

#%% 1. CCDF of degree distributions on the full panel

B2B_df_full = full_df.copy()
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

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
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
                              f'{year}', 'CCDF', f'outdeg_ccdf_{year}.png'), dpi=300 )
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
                               f'{year}', 'CCDF', f'indeg_ccdf_{year}.png'), dpi=300 )
    plt.close()
    

