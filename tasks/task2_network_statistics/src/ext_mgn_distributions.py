import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os 
from common.utilities import set_ticks_log_scale, demean_variable_in_df, calculate_distribution_moments, kernel_density_plot, find_kernel_densities

def calculate_degrees(G):
    
    # Number of buyers per seller
    out_degree = {node: degree for node, degree in G.out_degree() if degree > 0}
    
    # Number of sellers per buyer
    in_degree = {node: degree for node, degree in G.in_degree() if degree > 0}
    
    return out_degree, in_degree

def degree_plots(array, degree: str, output_path, year, df_year):
    """
    General function to generate degree plots for both out-degree and in-degree based on degree_name.
    
    Parameters:
        array: The degree values (either in-degree or out-degree).
        degree: The degree name (either 'outdegree' or 'indegree').
        output_path: Path to save the plots.
        year: The year for which the plots are being generated.
        df_year: The data for the corresponding year.
    """
    # Determine degree type (customers or suppliers) and columns based on degree_name
    if degree == 'outdegree':
        degree_type = "customers"
        vat_column = 'vat_i'
        nace_column = 'nace_i'
        nickname = 'out'
    elif degree == 'indegree':
        degree_type = "suppliers"
        vat_column = 'vat_j'
        nace_column = 'nace_j'
        nickname = 'in'
    else:
        raise ValueError("degree must be either out_degree or in_degree")

    # Degree plot
    log_degree = np.array(np.log(list(array.values())))
    kernel_density_plot(log_degree, f'Number of {degree_type}', 'Density', f'{nickname}_degree.png', output_path, year)
    
    # Now de-mean variables
    df_degree = df_year.copy()
    df_degree[f'{nickname}deg'] = df_degree[vat_column].map(array)
    df_degree[f'ln_{nickname}deg'] = np.log(df_degree[f'{nickname}deg'])
    
    df_degree = df_degree.groupby(nace_column).filter(lambda x: len(x) >= 5)  # Filter out sectors with less than 5 firms
    demeaned_degree = demean_variable_in_df(f'ln_{nickname}deg', nace_column, df_degree)
    
    kernel_density_plot(demeaned_degree, f'Number of {degree_type}, demeaned', 'Density', f'{nickname}_degree_demeaned.png', output_path, year)
        
# 1. Calculate distributions and moments per year
def calculate_distributions_per_year(full_df, output_path, start, end):

    for year in range(start, end+1):

        df_year = full_df[full_df['year'] == year]
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        # Degrees
        out_degree, in_degree = calculate_degrees(G_year)
        
        # Plot kernel density of out-degree and in-degree distributions
        ## Out-degree plot
        degree_plots(out_degree, 'outdegree',output_path, year, df_year)
        
        ## In-degree plot
        degree_plots(in_degree, 'indegree',output_path, year, df_year)

def degree_plots_by_ind(df_year, G_year, year, degree: str, output_path, demean=False):
    """
    General function to plot kernel densities for degree (either out-degree or in-degree) by industry.
    
    Parameters:
        G_year: The graph for the given year.
        array (dict): The degree values (either in-degree or out-degree).
        degree (str): The degree name (either 'outdegree' or 'indegree').
        output_path: Path to save the plots.
        df_year: The data for the corresponding year.
        year: The year for which the plots are being generated.
        demean (bool): Whether to demean the variables before plotting.
    """
    # Determine degree type (customers or suppliers) and columns based on degree_name
    if degree == 'outdegree':
        degree_type = "customers"
        vat_column = 'vat_i'
        nace_column = 'nace_i'
        nickname = 'out'
        degree_func = G_year.out_degree
        industry_index = 'industry_i'
    elif degree == 'indegree':
        degree_type = "suppliers"
        vat_column = 'vat_j'
        nace_column = 'nace_j'
        nickname = 'in'
        degree_func = G_year.in_degree
        industry_index = 'industry_j'
    else:
        raise ValueError("degree must be either 'outdegree' or 'indegree'")
    
    ## Plot kernel density for degree by industry
    x_min = np.inf
    x_max = -np.inf

    industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction', 
                'Market services','Non-market services']
    for industry in industries:
        industry_nodes = df_year[df_year[industry_index] == industry][f'{vat_column}'].unique()
        degree_sec = {node: degree for node, degree in degree_func(industry_nodes) if degree > 0}

        if demean:
            # Filter the data for the current year and sector
            df_year_sec = df_year[df_year[industry_index] == industry].copy()
            df_year_sec[f'{nickname}deg'] = df_year_sec[f'{vat_column}'].map(degree_sec)
            df_year_sec[f'ln_{nickname}deg'] = np.log(df_year_sec[f'{nickname}deg'])
        
            # now de-mean variables
            df_year_sec = df_year_sec.groupby(f'{nace_column}').filter(lambda x: len(x) >= 5) # Filter out sectors with less than 5 firms
            log_deg = demean_variable_in_df(f'ln_{nickname}deg', f'{nace_column}', df_year_sec)
        else:
            log_deg = np.array( np.log(list(degree_sec.values())) )

        grid, kde_densities = find_kernel_densities(log_deg)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)
    
    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel(f'Number of {degree_type}')
    plt.ylabel('Density')
    plt.legend()
    if demean:
        plt.savefig(os.path.join(output_path,f'{year}', 'kernel_densities', f'{nickname}_degree_demeaned_bysec.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_path,f'{year}', 'kernel_densities', f'{nickname}_degree_bysec.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 2. Calculate distributions and moments per year by sector of the seller
def calculate_distributions_per_year_by_sector(full_df, output_path, start, end):
    
    for year in range(start, end+1):
        
        df_year = full_df[full_df['year'] == year]
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        ## Out-degree
        # raw data
        degree_plots_by_ind(df_year, G_year, year, 'outdegree', output_path, demean=False)
        # demeaned data
        degree_plots_by_ind(df_year, G_year, year, 'outdegree', output_path, demean=True)
        
        ## In-degree
        # raw data
        degree_plots_by_ind(df_year, G_year, year, 'indegree', output_path, demean=False)
        # demeaned data
        degree_plots_by_ind(df_year, G_year, year, 'indegree', output_path, demean=True)


#%% 3. Construct summary tables
def calculate_degree_summary(G_year, df_year, degree: str, industries, output_path, year):
    """
    General function to calculate degree summary tables for both out-degree and in-degree.
    
    Parameters:
        G_year (nx.DiGraph): The graph for the given year.
        df_year (pd.DataFrame): The data for the corresponding year.
        degree_type (str): Either 'outdegree' or 'indegree' to specify the type of degree.
        industries (list): List of industry names.
        output_path (str): Path to save the output CSV.
        year (int): The year for which the summary tables are being generated.
    """
    if degree == 'outdegree':
        degree_func = G_year.out_degree
        vat_column = 'vat_i'
        nickname = 'out'
        industry_index = 'industry_i'
    elif degree == 'indegree':
        degree_func = G_year.in_degree
        vat_column = 'vat_j'
        nickname = 'in'
        industry_index = 'industry_j'
    else:
        raise ValueError("degree_type must be either 'outdegree' or 'indegree'")
    
    # Initialize summary list
    sum_bysec = []
    
     # Calculate degree moments for each industry
    for industry in industries:
        industry_nodes = df_year[df_year[industry_index] == industry][vat_column].unique()
        degree_sec = {node: degree for node, degree in degree_func(industry_nodes) if degree > 0}
        moments_sec = calculate_distribution_moments(list(degree_sec.values()))
        sum_bysec.append(moments_sec)
    
    # Calculate degree moments for the full graph
    degree_full = {node: degree for node, degree in degree_func() if degree > 0}
    moments_full = calculate_distribution_moments(list(degree_full.values()))
    sum_bysec.append(moments_full)

    # Create a summary table and save it to CSV
    sum_bysec_tab = pd.DataFrame(sum_bysec, index=industries + ['All'])
    sum_bysec_tab.to_csv(os.path.join(output_path, f'{year}', 'moments', f'{nickname}_degree_bysec.csv'))

def summary_tables(full_df, output_path, start, end):
    
    industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction',
                    'Market services','Non-market services']
    
    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year]
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        # Out-degree
        calculate_degree_summary(G_year, df_year, 'outdegree', industries, output_path, year)
        
        # In-degree
        calculate_degree_summary(G_year, df_year, 'indegree', industries, output_path, year)


# Master function
def master_ext_mgn_distributions(full_df, output_path, start, end):

    # Calculate distributions per year
    calculate_distributions_per_year(full_df, output_path, start, end)

    # Calculate distributions per year by sector
    calculate_distributions_per_year_by_sector(full_df, output_path, start, end)

    # Construct summary tables
    summary_tables(full_df, output_path, start, end)
    