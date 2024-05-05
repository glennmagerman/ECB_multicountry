import pandas as pd
import numpy as np
import networkx as nx

import os 
if 'abs_path' not in globals():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path,'..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% INPUT
dict_data = load_workspace( os.path.join( abs_path, 'task2_network_statistics', 'input','init_data.pkl') )
full_df = dict_data['full_df']

#%% 1. Full panel stats

G = nx.from_pandas_edgelist(full_df, 'vat_i', 'vat_j', create_using=nx.DiGraph())

# Total number of firms, sellers, and buyers in network panel
total_unique_sellers = len({seller for seller, degree in G.out_degree() if degree > 0})
total_unique_buyers = len({buyer for buyer, degree in G.in_degree() if degree > 0})
total_firms = len(G.nodes())

# Total number of firms with degree 0
out_zero = len({seller for seller, degree in G.out_degree() if degree == 0})
in_zero = len({buyer for buyer, degree in G.in_degree() if degree == 0})

full_panel_df = pd.DataFrame({
    'Statistics': ['Total sellers', 'Total buyers', 'Total firms', 'Zero outdegree', 'Zero indegree'],
    'Value': [total_unique_sellers, total_unique_buyers, total_firms, out_zero, in_zero]
    })
full_panel_df.to_csv(os.path.join(abs_path, 'task2_network_statistics','output', 
                                  'sum_stat_full.csv'), index=False)

#%% 2. Stats per year

summary_per_period = []

for year in full_df['year'].unique():
    df_year = full_df[full_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    unique_sellers = len({seller for seller, transactions in G_year.out_degree() if transactions > 0})
    unique_buyers = len({buyer for buyer, transactions in G_year.in_degree() if transactions > 0})
    links = G_year.size()
    
    summary_per_period.append({
        'year': year,
        'sellers': unique_sellers,
        'buyers': unique_buyers,
        'links': links
    })

summary_df = pd.DataFrame(summary_per_period).sort_values(by='year')

# Save table to csv
summary_df.to_csv( os.path.join(abs_path, 'task2_network_statistics','output', 
                                'sum_stat_year.csv'), index=False )

#%% 3. Stats per sector

summary_data_by_sector = []

# Creating a unique identifier for year and sector (of the seller) as these are the dimensions we'll aggregate on
full_df['year_nace'] = full_df.apply(lambda row: f"{row['year']}_{row['nace_i']}", axis=1)

for year_nace, group in full_df.groupby('year_nace'):
    year, nace = year_nace.split('_')
    G_sector = nx.from_pandas_edgelist(group, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    unique_sellers = len({seller for seller, degree in G_sector.out_degree() if degree > 0})
    unique_buyers = len({buyer for buyer, degree in G_sector.in_degree() if degree > 0})
    links = G_sector.size()
    unique_firms = len(G_sector.nodes())
    
    summary_data_by_sector.append({
        'year': year,
        'NACE': nace,
        'sellers': unique_sellers,
        'buyers': unique_buyers,
        'links': links,
        'firms': unique_firms
    })

summary_df_bysec = pd.DataFrame(summary_data_by_sector).sort_values(by=['year', 'NACE'])

# Save table to csv
summary_df_bysec.to_csv( os.path.join(abs_path, 'task2_network_statistics','output', 
                                      'sum_stat_bysec.csv'), index=False )


