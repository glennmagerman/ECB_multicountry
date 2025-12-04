import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
    full_df['year_nace'] = full_df.apply(lambda row: f"{row['year']}_{row['nace']}", axis=1)

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
    
#%% 4. Plots
def plots_descriptive(B2B_df, output_path, start, end):
    
    nfirms_array = []
    nlinks_array = []
    average_deg_array = []
    for year in range(start, end+1):
        df_year = B2B_df[B2B_df['year'] == year]
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        unique_sellers, unique_buyers, nfirms = count_firms(G_year)
        links = G_year.size()
    
        nfirms_array.append(nfirms)
        nlinks_array.append(links)
        
        average_deg = links / nfirms
        average_deg_array.append(average_deg)
        
    nlinks_array = np.array(nlinks_array) / 1_000_000 # in millions
    nfirms_array = np.array(nfirms_array) / 1_000 # in thousands
    
    years = np.arange(start, end+1)
    summary_df = pd.DataFrame({
        "year": years,
        "n_links (mln)": nlinks_array,
        "n_firms (thousands)": nfirms_array,
        "average_degree": average_deg_array,
    })
    csv_path = os.path.join(output_path, "descriptives_tab.csv")
    summary_df.to_csv(csv_path, index=False)
    
    # Plot 1: number of links and firms
    fig, ax1 = plt.subplots(figsize=(12,6))
    x = range(start, end+1)

    color1 = "tab:blue"

    # --- Left axis (nlinks)
    ax1.plot(x, nlinks_array, label="Number of links", linewidth=2, color=color1)
    ax1.set_ylabel("Links (millions)", color=color1)
    ax1.set_xlabel("Year")
    ax1.set_xticks(x[::2])
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Right axis (two series)
    ax2 = ax1.twinx()
    ax2.plot(x, nfirms_array, label="Number of firms", linestyle="--", color="tab:red")
    ax2.set_ylabel("Firms (thousands)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.grid(True, linestyle='--', alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'nfirms_links_plot.png'), dpi=300)
    plt.close()
    
    # Plot 2: average degree
    fig, ax = plt.subplots(figsize=(12,6))
    x = range(start, end+1)

    # --- Left axis (nlinks)
    ax.plot(x, average_deg_array, label="Average degree", linewidth=2)
    ax.set_ylabel("Average degree")
    ax.set_xlabel("Year")
    ax.set_xticks(x[::2])
    ax.tick_params(axis='y')
    ax.grid(True, linestyle='--', alpha=0.5)
    #plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'average_deg_plot.png'), dpi=300)
    plt.close()
    
    
# Master function
def master_sum_stat(B2B_df, firms_df, output_path):
    
    B2B_df = B2B_df.merge(firms_df[['vat', 'year', 'nace']].rename(columns={'vat':'vat_i'}), on=['vat_i', 'year'], how='left')
    start = B2B_df['year'].min()
    end = B2B_df['year'].max()

    # 1. Full panel stats
    full_panel_stats(B2B_df, output_path)

    # 2. Stats per year
    stats_per_year(B2B_df, output_path, start, end)

    # 3. Stats per sector
    stats_per_sec(B2B_df, output_path)
    
    # 4. Plots
    plots_descriptive(B2B_df, output_path, start, end)
    
    