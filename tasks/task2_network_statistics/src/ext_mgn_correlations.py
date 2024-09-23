import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
import statsmodels.api as sm
import os 
from common.utilities import demean_variable_in_df, set_ticks_log_scale, aggregate_and_bin

#%% 1. Compute correlations between network sales vs outdegree
def transform_in_log_and_demean(df_year, x, y, nace_column):
     
    # Create log-transformed variables
    df_year[f'log_{x}'] = np.log(df_year[x])
    df_year[f'log_{y}'] = np.log(df_year[y])
    
    # drop possible NaNs
    df_year = df_year.dropna(subset=[x,y])

    # demean the variables
    df_year = df_year.groupby(f'{nace_column}').filter(lambda w: len(w) >= 5) # drop NACE codes with less than 5 observations
    df_year[f'log_{x}_dem'] = demean_variable_in_df(x, f'{nace_column}', df_year)
    df_year[f'log_{y}_dem'] = demean_variable_in_df(y, f'{nace_column}', df_year)
    
    return df_year

def regress_degree_vs_netvar(df_year, nace_column, nickname, network_var):
    
    # regression on the (demeaned) underlying data
    df_year.set_index([f'{nace_column}','year'],inplace=True)
    Y = df_year[f'ln_network_{network_var}_dem']
    X = sm.add_constant(df_year[[f'ln_{nickname}deg_dem']])  # Adds a constant term to the model
    model = PanelOLS(Y, X, entity_effects=True)
    res = model.fit(cov_type='robust')
    
    return res

def plot_binned_data(binned_data, res, x, y):
    
    # Plotting binned data
    plt.scatter(binned_data[x], binned_data[y])
    textstr = '\n'.join((
        f'Linear slope: {res.params[x]:.2f} ({res.std_errors[x]:.2f})',
        f'R-squared: {res.rsquared:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    set_ticks_log_scale(np.array(binned_data[x]), step=2)
    set_ticks_log_scale(np.array(binned_data[y]), step=2, axis='y')

def process_degree_correlations(df_year, G_year, degree: str, output_path, year):
    """
    General function to process and analyze correlations between degrees (out-degree/in-degree) and network sales/purchases.
    
    Parameters:
        df_year (pd.DataFrame): The data for the corresponding year.
        degree (str): Either 'outdegree' or 'indegree' to specify the type of degree.
        abs_path (str): Path to save the output files.
        year (int): The year for which the correlations are being generated.
    """

    if degree == 'outdegree':
        vat_column = 'vat_i'
        nace_column = 'nace_i'
        nickname = 'out'
        network_var = 'sales'
        ylabel = 'Network sales, demeaned'
        degree_type = 'customers'
        degree_func = G_year.out_degree

    elif degree == 'indegree':
        vat_column = 'vat_j'
        nace_column = 'nace_j'
        nickname = 'in'
        network_var = 'purch'
        ylabel = 'Network purchases, demeaned'
        degree_type = 'suppliers'
        degree_func = G_year.in_degree

    # Find out-degree and add it to the dataframe
    degree = {node: degree for node, degree in degree_func()}
    df_year[f'{nickname}deg'] = df_year[f'{vat_column}'].map(degree)
    
    # Create log-transformed variables
    df_year[f'ln_{nickname}deg'] = np.log(df_year[f'{nickname}deg'])
    df_year[f'ln_network_{network_var}_i'] = np.log(df_year.groupby(f'{vat_column}')['sales_ij'].transform('sum'))
    
    # Estimate the FE model
    df_year = df_year.groupby(f'{nace_column}').filter(lambda x: len(x) >= 5) # drop NACE codes with less than 5 observations
    # demean the variables
    df_year[f'ln_network_{network_var}_dem'] = demean_variable_in_df(f'ln_network_{network_var}_i', f'{nace_column}', df_year)
    df_year[f'ln_{nickname}deg_dem'] = demean_variable_in_df(f'ln_{nickname}deg', f'{nace_column}', df_year)

    # regression on the (demeaned) underlying data
    res = regress_degree_vs_netvar(df_year, nace_column, nickname, network_var)
    
    # aggregate data into 20 bins for network sales and calculate mean of degree for each bin
    binned_data = aggregate_and_bin(df_year, f'ln_{nickname}deg_dem', f'ln_network_{network_var}_dem')
    
    # Plotting
    plot_binned_data(binned_data, res, f'ln_{nickname}deg_dem', f'ln_network_{network_var}_dem')
    plt.ylabel(ylabel)
    plt.xlabel(f'Number of {degree_type}, demeaned')
    plt.savefig( os.path.join(output_path,f'{year}', 'correlations', f'outdeg_net_{network_var}_dem.png'), dpi=300, bbox_inches='tight' )
    plt.close()

def corr_degree_vs_network_var(full_df, output_path, start, end):
    
    for year in range(start, end+1):

        df_year = full_df[full_df['year'] == year].copy()
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())

        # Outdegree vs network sales
        process_degree_correlations(df_year, G_year, 'outdegree', output_path, year)
        # Indegree vs network purchases
        process_degree_correlations(df_year, G_year, 'indegree', output_path, year)

# 3. Compute correlations between outdegree and indegree

def regress_degree(df_year):
        
        # run the regression to print the results on the plot
        df_year = df_year.set_index(['nace_i', 'year'])
        X = sm.add_constant(df_year['log_outdeg_dem'])  # Adds a constant term to the model
        model = PanelOLS(df_year['log_indeg_dem'], X, entity_effects=True)
        res = model.fit(cov_type='robust')

        return res

def corr_outdeg_indeg(full_df, output_path, start, end):

    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year].copy()
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        out_degree = {node: degree for node, degree in G_year.out_degree() }
        in_degree = {node: degree for node, degree in G_year.in_degree() if degree >0}
        df_year['outdeg'] = df_year['vat_i'].map(out_degree)
        df_year['indeg'] = df_year['vat_i'].map(in_degree)

        # transform variables in logs
        df_year['log_indeg'] = np.log(df_year['indeg'])
        df_year['log_outdeg'] = np.log(df_year['outdeg'])
        
        # de-mean the variables
        df_year = df_year.dropna(subset=['log_indeg']) # drop NaNs
        df_year = df_year.groupby('nace_i').filter(lambda x: len(x) >= 5) # drop NACE codes with less than 5 observations
        df_year['log_indeg_dem'] = demean_variable_in_df('log_indeg', 'nace_i', df_year)
        df_year['log_outdeg_dem'] = demean_variable_in_df('log_outdeg', 'nace_i', df_year)
        
        # regress the variables
        res = regress_degree(df_year)
        
        # bin the data
        binned_data = aggregate_and_bin(df_year, 'log_outdeg_dem', 'log_indeg_dem')
        
        # Now plot the binned data
        plot_binned_data(binned_data, res, 'log_outdeg_dem', 'log_indeg_dem')
        plt.xlabel('Number of customers, demeaned')
        plt.ylabel('Number of suppliers, demeaned')
        plt.savefig( os.path.join(output_path,f'{year}', 'correlations', 'indeg_outdeg.png'), dpi=300, bbox_inches='tight' )
        plt.close()


# Master function
def master_ext_mgn_correlations(full_df, output_path, start, end):
        
        # Compute correlations between degree and respective network variables
        corr_degree_vs_network_var(full_df, output_path, start, end)

        # Compute correlation between outdegree and indegree
        corr_outdeg_indeg(full_df, output_path, start, end)
        
    

