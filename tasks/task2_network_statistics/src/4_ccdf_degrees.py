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
dict_data = load_workspace( os.path.join( abs_path, 'task2_network_statistics','input','init_data.pkl') )
full_df = dict_data['full_df']

#%% 1. CCDF of degree distributions per year

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year].copy()
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # Find out-degree and in-degree and add it to the dataframe
    out_degree = {node: degree for node, degree in G_year.out_degree() if degree > 0}
    in_degree = {node: degree for node, degree in G_year.in_degree() if degree > 0}
    
    # create path
    if not os.path.exists( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'CCDF') ):
        os.makedirs( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'CCDF') )
        
    # CCDF of out-degree distribution
    outdeg_distr = np.unique( np.array( list(out_degree.values()) ) )
    coeff = (powerlaw.Fit(outdeg_distr).power_law.alpha - 1)
    xmin = powerlaw.Fit(outdeg_distr).xmin
    powerlaw.plot_ccdf(outdeg_distr, marker='.', linewidth=0)
    textstr = '\n'.join((
        f'Tail exponent:  {coeff:.2f}',
        f'$x_{{min}}$ = {xmin}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.2, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.xlabel('Number of customers')
    plt.ylabel('CCDF')
    plt.grid(True)
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output', f'{year}', 'CCDF',f'outdeg_ccdf_{year}.png'), 
                dpi=300, bbox_inches='tight' )
    plt.close()

    # CCDF of in-degree distribution
    indeg_distr = np.unique( np.array( list(in_degree.values()) ) )
    coeff = (powerlaw.Fit(indeg_distr).power_law.alpha - 1)
    xmin = powerlaw.Fit(indeg_distr).xmin
    powerlaw.plot_ccdf(indeg_distr, marker='.', linewidth=0)
    textstr = '\n'.join((
        f'Tail exponent:  {coeff:.2f}',
        f'$x_{{min}}$ = {xmin}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.2, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.xlabel('Number of suppliers')
    plt.ylabel('CCDF')
    plt.grid(True)
    plt.savefig( os.path.join(abs_path, 'task2_network_statistics','output',f'{year}', 'CCDF',f'indeg_ccdf_{year}.png'), 
                dpi=300, bbox_inches='tight' )
    plt.close()
    
