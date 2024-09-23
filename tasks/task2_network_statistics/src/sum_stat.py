import pandas as pd
import numpy as np
import networkx as nx
import os 

# 1. Full panel stats
def count_firms(G):

    # Number of firms, sellers, and buyers in network panel
    unique_sellers = len({seller for seller, degree in G.out_degree() if degree > 0})
    unique_buyers = len({buyer for buyer, degree in G.in_degree() if degree > 0})
    nfirms = len(G.nodes())

    return unique_sellers, unique_buyers, nfirms

def count_zero_degree(G):

    # Total number of firms with degree 0
    out_zero = len({seller for seller, degree in G.out_degree() if degree == 0})
    in_zero = len({buyer for buyer, degree in G.in_degree() if degree == 0})

    return out_zero, in_zero

def full_panel_stats(full_df, output_path):

    # Create the directed graph
    G = nx.from_pandas_edgelist(full_df, 'vat_i', 'vat_j', create_using=nx.DiGraph())

    # Total number of firms, sellers, and buyers in network panel
    total_unique_sellers, total_unique_buyers, total_firms = count_firms(G)

    # Total number of firms with degree 0
    out_zero, in_zero = count_zero_degree(G)

    full_panel_stats_df = pd.DataFrame({
        'Statistics': ['Total sellers', 'Total buyers', 'Total firms', 'Zero outdegree', 'Zero indegree'],
        'Value': [total_unique_sellers, total_unique_buyers, total_firms, out_zero, in_zero]
        })
    full_panel_stats_df.to_csv(os.path.join(output_path,'sum_stat_full.csv'), index=False)

# 2. Stats per year
def stats_per_year(full_df, output_path, start, end):

    summary_per_period = []

    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year]
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        unique_sellers, unique_buyers, nfirms = count_firms(G_year)
        links = G_year.size()
        
        summary_per_period.append({
            'year': year,
            'sellers': unique_sellers,
            'buyers': unique_buyers,
            'links': links
        })

    summary_df = pd.DataFrame(summary_per_period).sort_values(by='year')

    # Save table to csv
    summary_df.to_csv( os.path.join(output_path, 'sum_stat_year.csv'), index=False )

#%% 3. Stats per sector
def stats_per_sec(full_df, output_path):

    summary_data_by_sector = []

    # Creating a unique identifier for year and sector (of the seller) as these are the dimensions we'll aggregate on
    full_df['year_nace'] = full_df.apply(lambda row: f"{row['year']}_{row['nace_i']}", axis=1)

    for year_nace, group in full_df.groupby('year_nace'):
        year, nace = year_nace.split('_')
        G_sector = nx.from_pandas_edgelist(group, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        unique_sellers, unique_buyers, unique_firms = count_firms(G_sector)
        links = G_sector.size()
        
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
    summary_df_bysec.to_csv( os.path.join(output_path, 'sum_stat_bysec.csv'), index=False )


# Master function
def master_sum_stat(full_df, output_path, start, end):

    # 1. Full panel stats
    full_panel_stats(full_df, output_path)

    # 2. Stats per year
    stats_per_year(full_df, output_path, start, end)

    # 3. Stats per sector
    stats_per_sec(full_df, output_path)
    