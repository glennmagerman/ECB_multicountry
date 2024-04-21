import pandas as pd
import numpy as np
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

#%% 1. Compute distributions of sales for each seller and buyer (per year)

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year]
    
    ## Sales for each seller
    df_year_seller = df_year.copy()
    df_year_seller['ln_turnover_i'] = np.log(df_year_seller['turnover_i'])
    ln_turnover_i = np.array( df_year_seller['ln_turnover_i'] ).reshape(-1, 1)
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(ln_turnover_i)
    grid = np.linspace(ln_turnover_i.min(), ln_turnover_i.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Total sales')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'sales_seller.png'), dpi=300 )
    plt.close()
    
    # now de-mean variables
    df_year_seller = df_year_seller.set_index(['nace_i', 'year'])
    mod = PanelOLS.from_formula('ln_turnover_i ~ 1 + EntityEffects', df_year_seller)
    res = mod.fit()
    lturnover_dem_i = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lturnover_dem_i)
    grid = np.linspace(lturnover_dem_i.min(), lturnover_dem_i.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Total sales, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'sales_seller_demeaned.png'), dpi=300 )
    plt.close()
    
    ## Sales for each buyer
    df_year_buyer = df_year.copy()
    df_year_buyer['ln_turnover_j'] = np.log(df_year_buyer['turnover_j'])
    ln_turnover_j = np.array( df_year_buyer['ln_turnover_j'] ).reshape(-1, 1)
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(ln_turnover_j)
    grid = np.linspace(ln_turnover_j.min(), ln_turnover_j.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Total sales')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'sales_buyer.png'), dpi=300 )
    plt.close()
    
    # now de-mean variables
    df_year_buyer = df_year_buyer.set_index(['nace_j', 'year'])
    mod = PanelOLS.from_formula('ln_turnover_j ~ 1 + EntityEffects', df_year_buyer)
    res = mod.fit()
    lturnover_dem_j = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
    kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lturnover_dem_j)
    grid = np.linspace(lturnover_dem_i.min(), lturnover_dem_i.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    plt.plot(grid, kde_densities)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Total sales, demeaned')
    plt.ylabel('Density')
    plt.savefig( os.path.join(abs_path, 'task1_network_statistics','output', 
                              f'{year}', 'kernel_densities', 'sales_buyer_demeaned.png'), dpi=300 )
    plt.close()


#%% 2. Calculate distributions and moments per year by sector of the seller

industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction', 
            'Market services','Non-market services']

# Apply the recategorize_industry function to the 2-digit NACE codes
full_df['nace_2digit'] = full_df['nace_i'].astype(str).replace(
    '', np.nan).str[:2].apply(lambda x: int(x) if pd.notna(x) and x.isdigit() else np.nan)
full_df['industry_i'] = full_df['nace_2digit'].apply(recategorize_industry)

for year in pd.unique(full_df['year']):
    df_year_seller = full_df[full_df['year'] == year]
    
    ## 2a. Plot kernel density sales of the seller by industry
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year_seller[df_year_seller['industry_i'] == industry].copy()
        
        df_year_sec['ln_turnover_i'] = np.log(df_year_sec['turnover_i'])
        ln_turnover_i = np.array(df_year_sec['ln_turnover_i']).reshape(-1, 1)
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(ln_turnover_i)
        grid = np.linspace(ln_turnover_i.min(), ln_turnover_i.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Total sales')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_seller_bysec.png'), dpi=300)
    plt.close()
    
    # Now with de-meaned variables
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year_seller[df_year_seller['industry_i'] == industry].copy()
        
        # now de-mean variables
        df_year_sec = df_year_sec.set_index(['nace_i', 'year'])
        df_year_sec['ln_turnover_i'] = np.log(df_year_sec['turnover_i'])
        mod = PanelOLS.from_formula('ln_turnover_i ~ 1 + EntityEffects', df_year_sec)
        res = mod.fit()
        lturnover_dem_i = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lturnover_dem_i)
        grid = np.linspace(lturnover_dem_i.min(), lturnover_dem_i.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Total sales, demeaned')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_seller_bysec_demeaned.png'), dpi=300)
    plt.close()
    
    ## 2b. Plot kernel density sales of the buyer by industry (of the seller)
    df_year_buyer = full_df[full_df['year'] == year]
    
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year_buyer[df_year_buyer['industry_i'] == industry].copy()
        
        df_year_sec['ln_turnover_j'] = np.log(df_year_sec['turnover_j'])
        ln_turnover_j = np.array(df_year_sec['ln_turnover_j']).reshape(-1, 1)
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(ln_turnover_j)
        grid = np.linspace(ln_turnover_j.min(), ln_turnover_j.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Total sales')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_buyer_bysec.png'), dpi=300)
    plt.close()
    
    # Now with de-meaned variables
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year_buyer[df_year_buyer['industry_i'] == industry].copy()
        
        # now de-mean variables
        df_year_sec = df_year_sec.set_index(['nace_j', 'year'])
        df_year_sec['ln_turnover_j'] = np.log(df_year_sec['turnover_j'])
        mod = PanelOLS.from_formula('ln_turnover_j ~ 1 + EntityEffects', df_year_sec)
        res = mod.fit()
        lturnover_dem_j = np.array(res.resids).reshape(-1,1) # The residuals are the demeaned log variables
        kde = KernelDensity(kernel='epanechnikov', bandwidth=1).fit(lturnover_dem_j)
        grid = np.linspace(lturnover_dem_j.min(), lturnover_dem_j.max(), 1000).reshape(-1, 1)
        kde_scores = kde.score_samples( grid )
        kde_densities = np.exp(kde_scores)
        plt.plot(grid, kde_densities, label=f'{industry}')
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(val)}}}$'))
    plt.xlabel('Total sales, demeaned')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(abs_path, 'task1_network_statistics', 'output', 
                             f'{year}', 'kernel_densities', 'sales_buyer_bysec_demeaned.png'), dpi=300)
    plt.close()
    
#%% 3. Construct summary tables

for year in pd.unique(full_df['year']):
    df_year = full_df[full_df['year'] == year]
    
    turnover_i_sum_bysec = []
    
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()

        turnover_moments_sec = calculate_distribution_moments( df_year_sec['turnover_i'] / 1000000 ) # report values in mln euros
        turnover_i_sum_bysec.append( turnover_moments_sec )
       
    turnover_i_moments_full = calculate_distribution_moments( df_year['turnover_i'] / 1000000 )
    turnover_i_sum_bysec.append( turnover_i_moments_full )
    turnover_i_sum_bysec_tab = pd.DataFrame(turnover_i_sum_bysec, index=industries + ['All'])
    
    # Save table to csv
    if not os.path.exists( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'moments') ):
        os.makedirs( os.path.join(abs_path, 'task1_network_statistics','output', f'{year}', 'moments') )
    turnover_i_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task1_network_statistics', 'output', 
                                           f'{year}', 'moments','sales_seller_bysec.csv') )
    
    turnover_j_sum_bysec = []
    
    for industry in industries:
        
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()

        turnover_moments_sec = calculate_distribution_moments( df_year_sec['turnover_j']/ 1000000 ) # report values in mln euros
        turnover_j_sum_bysec.append( turnover_moments_sec )
       
    turnover_j_moments_full = calculate_distribution_moments( df_year['turnover_j'] / 1000000)
    turnover_j_sum_bysec.append( turnover_j_moments_full )
    turnover_j_sum_bysec_tab = pd.DataFrame(turnover_j_sum_bysec, index=industries + ['All'])
    
    #Save table to csv
    turnover_j_sum_bysec_tab.to_csv( os.path.join(abs_path, 'task1_network_statistics', 'output', 
                                          f'{year}', 'moments','sales_buyer_bysec.csv') )  
