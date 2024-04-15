import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.ticker import FuncFormatter, MaxNLocator
from linearmodels import PanelOLS

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
    columns={'vat': 'vat_i', 'corr_turnover':'turnover_i', 'nace':'nace_i'}), on=['vat_i','year'], how='inner' )
B2B_df = B2B_df.merge( firms_df[['year','vat','nace','corr_turnover']].rename(
    columns={'vat': 'vat_j', 'corr_turnover':'turnover_j', 'nace': 'nace_j'}), on=['vat_j','year'], how='inner' )

# some firms in vat_j do not report their turnover, so we need to impute it from the B2B sales
sales_sum_by_vat_i = B2B_df.groupby('vat_i')['sales_ij'].sum().reset_index()
sales_sum_by_vat_i.rename(columns={'vat_i': 'vat_j', 'sales_ij': 'sum_sales_ij'}, inplace=True)
B2B_df = B2B_df.merge(sales_sum_by_vat_i, on='vat_j', how='inner')
B2B_df['turnover_j'] = B2B_df.apply(
    lambda row: row['sum_sales_ij'] if pd.isna(row['turnover_j']) else row['turnover_j'], 
    axis=1)
B2B_df.drop(columns=['sum_sales_ij'], inplace=True)

#%% 1. Compute distributions of sales per seller and buyer (per year)

for year in pd.unique(B2B_df['year']):
    df_year = B2B_df[B2B_df['year'] == year]

    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    # Degrees
    out_degree = {node: degree for node, degree in G_year.out_degree() if degree > 0}  # Number of buyers per seller
    in_degree = {node: degree for node, degree in G_year.in_degree() if degree > 0} # Number of sellers per buyer
    
    ## 1a. Sales per customer
    df_year_seller = df_year.copy()
    
    df_year_seller['outdeg'] = df_year_seller['vat_i'].map(out_degree)
    df_year_seller['lsale_share_i'] = np.log(df_year_seller['turnover_i'] / df_year_seller['outdeg'])
    lsale_share_i = np.array( df_year_seller['lsale_share_i'] ).reshape(-1, 1)
    
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lsale_share_i)
    grid = np.linspace(lsale_share_i.min(), lsale_share_i.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Sales per customer')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'sales_share_perbuyer.png'), dpi=300 )
    plt.close()
    
    # now de-mean variables
    df_year_seller = df_year_seller.set_index(['nace_i', 'year'])
    mod = PanelOLS.from_formula('lsale_share_i ~ 1 + EntityEffects', df_year_seller)
    res = mod.fit()
    lsales_share_dem_i = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lsales_share_dem_i)
    grid = np.linspace(lsales_share_dem_i.min(), lsales_share_dem_i.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Sales per customer, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'sales_share_perbuyer_demeaned.png'), dpi=300 )
    plt.close()
    
    ## 1b. Sales per seller
    df_year_buyer = df_year.copy()
    
    df_year_buyer['indeg'] = df_year_buyer['vat_j'].map(in_degree)
    df_year_buyer['lsale_share_j'] = np.log(df_year_buyer['turnover_j'] / df_year_buyer['indeg'])
    lsale_share_j = np.array( df_year_buyer['lsale_share_j'] ).reshape(-1, 1)
    
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lsale_share_j)
    grid = np.linspace(lsale_share_j.min(), lsale_share_j.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Sales per seller')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'sales_share_perseller.png'), dpi=300 )
    plt.close()
    
    # now de-mean variables
    df_year_buyer = df_year_buyer.set_index(['nace_j', 'year'])
    mod = PanelOLS.from_formula('lsale_share_j ~ 1 + EntityEffects', df_year_buyer)
    res = mod.fit()
    lsales_shares_dem_j = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lsales_shares_dem_j)
    grid = np.linspace(lsales_shares_dem_j.min(), lsales_shares_dem_j.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Sales per seller, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'sales_share_perseller_demeaned.png'), dpi=300 )
    plt.close()


#%% 2. Calculate distributions and moments per year by sector of the seller

industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction', 
            'Market services','Non-market services']

# Apply the recategorize_industry function to the 2-digit NACE codes
B2B_df['nace_2digit'] = B2B_df['nace_i'].astype(str).replace(
    '', np.nan).str[:2].apply(lambda x: int(x) if pd.notna(x) and x.isdigit() else np.nan)
B2B_df['industry_i'] = B2B_df['nace_2digit'].apply(recategorize_industry)

for year in pd.unique(B2B_df['year']):
    df_year = B2B_df[B2B_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    ## 1a. Plot kernel density sales per buyer by industry
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        df_year_sec['outdeg'] = df_year_sec['vat_i'].map(out_degree)
        df_year_sec['lsale_share_i'] = np.log(df_year_sec['turnover_i'] / df_year_sec['outdeg'])
        lsale_share_i = np.array( df_year_sec['lsale_share_i'] ).reshape(-1, 1)
        
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lsale_share_i)
        grid = np.linspace(lsale_share_i.min(), lsale_share_i.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Sales per customer')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_share_perbuyer_bysec.png'), dpi=300)
    plt.close()
    
    # Now with de-meaned variables
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        # now de-mean variables
        df_year_sec = df_year_sec.set_index(['nace_i', 'year'])
        df_year_sec['outdeg'] = df_year_sec['vat_i'].map(out_degree)
        df_year_sec['lsale_share_i'] = np.log(df_year_sec['turnover_i'] / df_year_sec['outdeg'])
        
        mod = PanelOLS.from_formula('lsale_share_i ~ 1 + EntityEffects', df_year_sec)
        res = mod.fit()
        lsale_share_dem_i = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lsale_share_dem_i)
        grid = np.linspace(lsale_share_dem_i.min(), lsale_share_dem_i.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Sales per customer, demeaned')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_share_perbuyer_bysec_dem.png'), dpi=300)
    plt.close()
    
    ## 1b. Plot kernel density sales per seller by industry (of the seller)
    
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_j'].unique()
        in_degree = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        df_year_sec['indeg'] = df_year_sec['vat_j'].map(in_degree)
        df_year_sec['lsale_share_j'] = np.log(df_year_sec['turnover_j'] / df_year_sec['indeg'])
        lsale_share_j = np.array( df_year_sec['lsale_share_j'] ).reshape(-1, 1)
        
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lsale_share_j)
        grid = np.linspace(lsale_share_j.min(), lsale_share_j.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Sales per seller')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_share_perseller_bysec.png'), dpi=300)
    plt.close()
    
    # Now with de-meaned variables
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_j'].unique()
        in_degree = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        # now de-mean variables
        df_year_sec = df_year_sec.set_index(['nace_j', 'year'])
        df_year_sec['indeg'] = df_year_sec['vat_j'].map(in_degree)
        df_year_sec['lsale_share_j'] = np.log(df_year_sec['turnover_j'] / df_year_sec['indeg'])
        
        mod = PanelOLS.from_formula('lsale_share_j ~ 1 + EntityEffects', df_year_sec)
        res = mod.fit()
        lsale_share_dem_j = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lsale_share_dem_j)
        grid = np.linspace(lsale_share_dem_j.min(), lsale_share_dem_j.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Sales per seller, demeaned')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_share_perseller_bysec_dem.png'), dpi=300)
    plt.close()
    
#%% 3. Construct summary tables

for year in pd.unique(B2B_df['year']):
    df_year = B2B_df[B2B_df['year'] == year]
    G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
    
    sales_share_i_sum_bysec = []
    
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_i'].unique()
        out_degree = {node: degree for node, degree in G_year.out_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        df_year_sec['outdeg'] = df_year_sec['vat_i'].map(out_degree)

        sales_share_moments_sec = calculate_distribution_moments( df_year_sec['turnover_i'] / df_year_sec['outdeg'])
        sales_share_i_sum_bysec.append( sales_share_moments_sec )
      
    out_degree_full = dict( G_year.out_degree() )
    df_year['outdeg'] = df_year['vat_i'].map(out_degree_full)
    sales_share_i_moments_full = calculate_distribution_moments( df_year['turnover_i'] / df_year['outdeg'] )
    sales_share_i_sum_bysec.append( sales_share_i_moments_full )
    sales_share_i_sum_bysec_tab = pd.DataFrame(sales_share_i_sum_bysec, index=industries + ['All'])
    
    # Save table to csv
    if not os.path.exists( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'moments') ):
        os.makedirs( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'moments') )
    sales_share_i_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task1_network_statistics', 'output', 
                                           f'{year}', 'moments','sales_shares_perbuyer_bysec.csv') )
    
    sales_share_j_sum_bysec = []
    
    for industry in industries:
        industry_nodes = df_year[df_year['industry_i'] == industry]['vat_j'].unique()
        in_degree = {node: degree for node, degree in G_year.in_degree(industry_nodes) if degree > 0}
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        df_year_sec['indeg'] = df_year_sec['vat_j'].map(in_degree)

        sales_share_moments_sec = calculate_distribution_moments( df_year_sec['turnover_j'] / df_year_sec['indeg'])
        sales_share_j_sum_bysec.append( sales_share_moments_sec )
      
    in_degree_full = dict( G_year.in_degree() )
    df_year['indeg'] = df_year['vat_j'].map(in_degree_full)
    sales_share_j_moments_full = calculate_distribution_moments( df_year['turnover_j'] / df_year['indeg'] )
    sales_share_j_sum_bysec.append( sales_share_j_moments_full )
    sales_share_j_sum_bysec_tab = pd.DataFrame(sales_share_j_sum_bysec, index=industries + ['All'])
    
    # Save table to csv
    if not os.path.exists( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'moments') ):
        os.makedirs( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'moments') )
    sales_share_j_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task1_network_statistics', 'output', 
                                           f'{year}', 'moments','sales_shares_perseller_bysec.csv') ) 
