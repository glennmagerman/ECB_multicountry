import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.ticker import FuncFormatter, MaxNLocator
from linearmodels import PanelOLS

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

#%% 1. Calculate distributions and moments per year

for year in pd.unique( full_df['year'] ):
    df_year = full_df[full_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # Degrees
    out_degree = {node: degree for node, degree in G_year.out_degree() if degree > 0}  # Number of buyers per seller
    in_degree = {node: degree for node, degree in G_year.in_degree() if degree > 0} # Number of sellers per buyer
    
    # Plot kernel density of out-degree and in-degree distributions
    if not os.path.exists( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'kernel_densities') ):
        os.makedirs( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'kernel_densities') )
    
    ## Out-degree plot
    log_outdeg = np.array( np.log(list(out_degree.values())) ).reshape(-1, 1)
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(log_outdeg)
    grid = np.linspace(log_outdeg.min(), log_outdeg.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Number of buyers per seller')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'out_degree.png'), dpi=300 )
    plt.close()
    
    # now de-mean variables
    df_year_out = df_year.copy()
    df_year_out['outdeg'] = df_year_out['vat_i'].map(out_degree)
    df_year_out['ln_outdeg'] = np.log(df_year_out['outdeg'])
    df_year_out = df_year_out.set_index(['nace_i', 'year'])
    mod = PanelOLS.from_formula('ln_outdeg ~ 1 + EntityEffects', df_year_out)
    res = mod.fit()
    loutdeg_dem = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(loutdeg_dem)
    grid = np.linspace(loutdeg_dem.min(), loutdeg_dem.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Number of buyers per seller, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'out_degree_demeaned.png'), dpi=300 )
    plt.close()
    
    ## In-degree plot
    log_indeg = np.array( np.log(list(in_degree.values())) ).reshape(-1, 1)
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(log_indeg)
    grid = np.linspace(log_indeg.min(), log_indeg.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Number of sellers per buyer')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'in_degree.png'), dpi=300 )
    plt.close()
    
    # now de-mean variables
    df_year_in = df_year.copy()
    df_year_in['indeg'] = df_year_in['vat_j'].map(in_degree)
    df_year_in['ln_indeg'] = np.log(df_year_in['indeg'])
    df_year_in = df_year_in.set_index(['nace_j', 'year'])
    mod = PanelOLS.from_formula('ln_indeg ~ 1 + EntityEffects', df_year_in)
    res = mod.fit()
    lindeg_dem = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lindeg_dem)
    grid = np.linspace(lindeg_dem.min(), lindeg_dem.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Number of sellers per buyer, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'in_degree_demeaned.png'), dpi=300 )
    plt.close()

#%% 2. Calculate distributions and moments per year by sector of the seller

# modify the data to include broad definition of industries
full_df['nace_2digit'] = full_df['nace_i'].astype(str).replace(
    '', np.nan).str[:2].apply(lambda x: int(x) if pd.notna(x) and x.isdigit() else np.nan)

# Apply the recategorize_industry function to the 2-digit NACE codes
full_df['industry_i'] = full_df['nace_2digit'].apply(recategorize_industry)

industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction', 
            'Market services','Non-market services']

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    ## 1a. Plot kernel density for out-degree by industry
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        log_outdeg = np.array( np.log(list(out_degree.values())) ).reshape(-1, 1)
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(log_outdeg)
        grid = np.linspace(log_outdeg.min(), log_outdeg.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Number of buyers per seller')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'out_degree_bysec.png'), dpi=300)
    plt.close()
    
    # Now with de-meaned variables
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        df_year_sec['outdeg'] = df_year_sec['vat_i'].map(out_degree)
        df_year_sec['ln_outdeg'] = np.log(df_year_sec['outdeg'])
        
        # now de-mean variables
        df_year_sec = df_year_sec.set_index(['nace_i', 'year'])
        mod = PanelOLS.from_formula('ln_outdeg ~ 1 + EntityEffects', df_year_sec)
        res = mod.fit()
        loutdeg_dem = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(loutdeg_dem)
        grid = np.linspace(loutdeg_dem.min(), loutdeg_dem.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities)
        
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Number of buyers per seller, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'out_degree_demeaned_bysec.png'), dpi=300 )
    plt.close()
    
    ## 1b. Plot kernel density for in-degree by industry
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_j'].unique()
        in_degree = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        log_indeg = np.array( np.log(list(in_degree.values())) ).reshape(-1, 1)
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(log_indeg)
        grid = np.linspace(log_indeg.min(), log_indeg.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Number of sellers per buyer')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities','in_degree_bysec.png'), dpi=300)
    plt.close()
    
    # Now with de-meaned variables
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_j'].unique()
        in_degree = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        df_year_sec['indeg'] = df_year_sec['vat_j'].map(in_degree)
        df_year_sec['ln_indeg'] = np.log(df_year_sec['indeg'])
        
        # now de-mean variables
        df_year_sec = df_year_sec.set_index(['nace_j', 'year'])
        mod = PanelOLS.from_formula('ln_indeg ~ 1 + EntityEffects', df_year_sec)
        res = mod.fit()
        lindeg_dem = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lindeg_dem)
        grid = np.linspace(lindeg_dem.min(), lindeg_dem.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities)
        
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Number of sellers per buyer, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'in_degree_demeaned_bysec.png'), dpi=300 )
    plt.close()

#%% 3. Construct summary tables

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    out_sum_bysec = []
    
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree_sec = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        out_moments_sec = calculate_distribution_moments(list(out_degree_sec.values()) )
        out_sum_bysec.append( out_moments_sec )
       
    out_degree_full = {node: degree for node, degree in G_year.out_degree() if degree > 0}
    out_moments_full = calculate_distribution_moments(list(out_degree_full.values()) )
    out_sum_bysec.append( out_moments_full )
    out_sum_bysec_tab = pd.DataFrame(out_sum_bysec, index=industries + ['All'])
    
    # Save table to csv
    if not os.path.exists( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'moments') ):
        os.makedirs( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'moments') )
    out_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task1_network_statistics', 'output', 
                                           f'{year}', 'moments','out_degree_bysec.csv') )
    
    in_sum_bysec = []
    
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        in_degree_sec = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        in_moments_sec = calculate_distribution_moments(list(in_degree_sec.values()) )
        in_sum_bysec.append( in_moments_sec )
       
    in_degree_full = {node: degree for node, degree in G_year.in_degree() if degree > 0}
    in_moments_full = calculate_distribution_moments(list(in_degree_full.values()) )
    in_sum_bysec.append( in_moments_full )
    in_sum_bysec_tab = pd.DataFrame(in_sum_bysec, index=industries + ['All'])
    
    #Save table to csv
    in_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task1_network_statistics', 'output', 
                                          f'{year}', 'moments','in_degree_bysec.csv') )
    
#%% #TODO: degree distribution by trade status
