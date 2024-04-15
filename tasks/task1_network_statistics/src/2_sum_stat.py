import pandas as pd
import numpy as np
import networkx as nx

import os 
abs_path = os.path.abspath(os.path.join('..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% INPUT
dict_data = load_workspace( os.path.join( 'tmp','init_data.pkl') )
B2B_df = dict_data['B2B_df']
firms_df = dict_data['firms_df']

#%% 1. Full panel stats

G = nx.from_pandas_edgelist(B2B_df, 'vat_i', 'vat_j', create_using=nx.DiGraph())

# Total number of firms, sellers, and buyers in network panel
total_unique_sellers = len({seller for seller, transactions in G.out_degree() if transactions > 0})
total_unique_buyers = len({buyer for buyer, transactions in G.in_degree() if transactions > 0})
total_firms = len(G.nodes())

#%% 2. Stats per year

summary_per_period = []

for year in B2B_df['year'].unique():
    df_year = B2B_df[B2B_df['year'] == year]
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

summary_df = pd.DataFrame(summary_per_period)

# Save table to csv
summary_df.to_csv( os.path.join(abs_path, 'task1_network_statistics','output', 'sum_stat_full.csv'), index=False )

#%% 3. Stats per sector

# This requires extending the B2B_df with sector information before creating the graph
B2B_df_extended = B2B_df.merge(firms_df.rename(columns={'vat': 'vat_i', 'nace': 'nace_i'}), on=['vat_i', 'year'], how='left')

summary_data_by_sector = []

# Creating a unique identifier for year and sector as these are the dimensions we'll aggregate on
B2B_df_extended['year_nace'] = B2B_df_extended.apply(lambda row: f"{row['year']}_{row['nace_i']}", axis=1)

for year_nace, group in B2B_df_extended.groupby('year_nace'):
    year, nace = year_nace.split('_')
    G_sector = nx.from_pandas_edgelist(group, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    unique_sellers = len({seller for seller, transactions in G_sector.out_degree() if transactions > 0})
    unique_buyers = len({buyer for buyer, transactions in G_sector.in_degree() if transactions > 0})
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
summary_df_bysec.to_csv( os.path.join(abs_path, 'task1_network_statistics','output', 'sum_stat_bysec.csv'), index=False )


