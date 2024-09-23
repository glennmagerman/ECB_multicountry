import os
import numpy as np
import pandas as pd
from common.utilities import demean_variable_in_df, kernel_density_plot, find_kernel_densities, set_ticks_log_scale, calculate_distribution_moments
import matplotlib.pyplot as plt

## Plots by year
def compute_distributions(df_year, variable_type, distribution_type, output_path, year):
    """
    General function to compute and plot distributions (either sales or purchases) for total or network values.
    
    Parameters:
        df_year (pd.DataFrame): DataFrame for the current year.
        variable_type (str): Either 'sales' or 'purchases' to specify which distribution to compute.
        distribution_type (str): Either 'total' or 'network' to specify which type of distribution.
        output_path (str): Path to save the output plots.
        year (int): Year for which the computations are made.
    """
    # Define columns and labels based on the variable and distribution type
    if variable_type == 'sales':
        if distribution_type == 'total':
            column = 'ln_turnover_i'
            label = 'Total sales'
            file_label = 'turnover'
            nace_column = 'nace_i'
        elif distribution_type == 'network':
            column = 'ln_network_sales_i'
            label = 'Network sales'
            file_label = 'network_sales'
            nace_column = 'nace_i'
        else:
            raise ValueError("Invalid distribution_type. Must be either 'total' or 'network'.")
    elif variable_type == 'purchases':
        if distribution_type == 'total':
            column = 'ln_inputs_j'
            label = 'Total purchases'
            file_label = 'inputs'
            nace_column = 'nace_j'
        elif distribution_type == 'network':
            column = 'ln_network_inputs_j'
            label = 'Network purchases'
            file_label = 'network_purchases'
            nace_column = 'nace_j'
        else:
            raise ValueError("Invalid distribution_type. Must be either 'total' or 'network'.")
    else:
        raise ValueError("Invalid variable_type. Must be either 'sales' or 'purchases'.")

    # Perform the kernel density plot for the given column
    data = np.array(df_year[column])
    kernel_density_plot(data, label, 'Density', f'{file_label}.png', output_path, year)
    
    # Now de-mean the variables by sector (grouped by group_column)
    df_year = df_year.groupby(nace_column).filter(lambda x: len(x) >= 5)
    demeaned_data = demean_variable_in_df(column, nace_column, df_year)
    kernel_density_plot(demeaned_data, f'{label}, demeaned', 'Density', f'{file_label}_demeaned.png', output_path, year)

def plot_densities_sales_or_purchases(full_df, output_path, variable_type: str, distribution_type: str, start, end):
    """
    Process either sales or purchases for a specific distribution type (total or network).
    
    Parameters:
        full_df (pd.DataFrame): Full DataFrame with all years.
        output_path (str): Path to save the plots.
        variable_type (str): Either 'sales' or 'purchases'.
        distribution_type (str): Either 'total' or 'network'.
    """
    for year in range(start, end + 1):
        df_year = full_df[full_df['year'] == year].copy()

        # Apply log transformations for the necessary columns based on the type of variable
        if variable_type == 'sales':
            df_year['ln_turnover_i'] = np.log(df_year['turnover_i'])
            df_year['ln_network_sales_i'] = np.log(df_year.groupby('vat_i')['sales_ij'].transform('sum'))
        elif variable_type == 'purchases':
            df_year['ln_inputs_j'] = np.log(df_year['inputs_j'].replace(0, np.nan))
            df_year.dropna(subset=['ln_inputs_j'], inplace=True)
            df_year['ln_network_inputs_j'] = np.log(df_year.groupby('vat_j')['sales_ij'].transform('sum'))
        
        # Compute distributions based on the provided type
        compute_distributions(df_year, variable_type, distribution_type, output_path, year)

## Plots by industry

def plots_by_ind(df_year, year, var: str, variable_type: str, output_path, demean=False):
    """
    Generalized function to plot kernel density by industry for both sales and purchases.
    
    Parameters:
        df_year (pd.DataFrame): The dataframe for the given year.
        year (int): The current year.
        var (str): The variable to plot (e.g., 'turnover', 'network_sales', 'inputs', 'network_purchases').
        variable_type (str): Either 'sales' or 'purchases' to determine the column suffix.
        output_path (str): Path to save the output plots.
        demean (bool): Whether to demean the variables before plotting.
    """
    # Define the xlabel and the necessary column transformations
    if variable_type == 'sales':
        if var == 'turnover':
            xlabel = 'Total sales'
        elif var == 'network_sales':
            xlabel = 'Network sales'
            df_year['network_sales_i'] = df_year.groupby('vat_i')['sales_ij'].transform('sum')
        column_suffix = '_i'
    elif variable_type == 'purchases':
        if var == 'inputs':
            xlabel = 'Total purchases'
        elif var == 'network_purchases':
            xlabel = 'Network purchases'
            df_year['network_purchases_j'] = df_year.groupby('vat_j')['sales_ij'].transform('sum')
        column_suffix = '_j'
    else:
        raise ValueError("variable_type must be either 'sales' or 'purchases'.")

    # List of industries
    industries = ['Primary and extraction', 'Manufacturing', 'Utilities', 'Construction',
                  'Market services', 'Non-market services']

    # Plot kernel density by industry
    x_min = np.inf
    x_max = -np.inf
    for industry in industries:
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()
        
        if demean:
            df_year_sec[f'ln_{var}{column_suffix}'] = np.log(df_year_sec[f'{var}{column_suffix}'])
            ln_var = demean_variable_in_df(f'ln_{var}{column_suffix}', f'nace{column_suffix}', df_year_sec)
        else:
            ln_var = np.array(np.log(df_year_sec[f'{var}{column_suffix}']))

        grid, kde_densities = find_kernel_densities(ln_var)
        plt.plot(grid, kde_densities, label=f'{industry}')
        x_min = min(min(grid), x_min)
        x_max = max(max(grid), x_max)

    set_ticks_log_scale([x_min, x_max], step=2)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.legend()

    # Save plot
    if demean:
        plt.savefig(os.path.join(output_path, f'{year}', 'kernel_densities', f'{var}_bysec_demeaned.png'),
                    dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_path, f'{year}', 'kernel_densities', f'{var}_bysec.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()

def kernel_densities_bysec(full_df, output_path, variable_type: str, start, end):
    """
    General function to process kernel densities for both sales and purchases.

    Parameters:
        full_df (pd.DataFrame): The full dataframe containing all the data.
        output_path (str): Path to save the output plots.
        variable_type (str): Either 'sales' or 'purchases' to specify what is being processed.
    """
    for year in range(start, end + 1):
        df_year = full_df[full_df['year'] == year].copy()

        if variable_type == 'sales':
            # Plot turnover and network sales kernel densities by industry
            plots_by_ind(df_year, year, 'turnover', variable_type, output_path)
            plots_by_ind(df_year, year, 'turnover', variable_type, output_path, demean=True)
            plots_by_ind(df_year, year, 'network_sales', variable_type, output_path)
            plots_by_ind(df_year, year, 'network_sales', variable_type, output_path, demean=True)

        elif variable_type == 'purchases':
            # Plot total purchases and network purchases kernel densities by industry
            plots_by_ind(df_year, year, 'inputs', variable_type, output_path)
            plots_by_ind(df_year, year, 'inputs', variable_type, output_path, demean=True)
            plots_by_ind(df_year, year, 'network_purchases', variable_type, output_path)
            plots_by_ind(df_year, year, 'network_purchases', variable_type, output_path, demean=True)

## Summary tables

def summary_table_by_year(df_year, year, variable_type: str, output_path):
    """
    Generalized function to compute summary tables for sales or purchases by industry.
    
    Parameters:
        df_year (pd.DataFrame): DataFrame for the given year.
        year (int): The year for which the summary is being generated.
        variable_type (str): Either 'sales' or 'purchases' to determine the column.
        output_path (str): Path to save the output summary table.
    """
    # List of industries
    industries = ['Primary and extraction', 'Manufacturing', 'Utilities', 'Construction',
                  'Market services', 'Non-market services']

    # Determine the column name and file name based on the variable_type
    if variable_type == 'sales':
        column = 'turnover_i'
        file_label = 'turnover_bysec.csv'
    elif variable_type == 'purchases':
        column = 'inputs_j'
        file_label = 'purchases_bysec.csv'
    else:
        raise ValueError("variable_type must be either 'sales' or 'purchases'.")

    sum_bysec = []
    for industry in industries:
        # Filter the data for the current year and sector
        df_year_sec = df_year[df_year['industry_i'] == industry].copy()

        # Calculate moments for the industry and append to list
        moments_sec = calculate_distribution_moments(df_year_sec[column] / 1000000)  # Report values in mln euros
        sum_bysec.append(moments_sec)

    # Calculate moments for the full dataset (all industries)
    moments_full = calculate_distribution_moments(df_year[column] / 1000000)
    sum_bysec.append(moments_full)

    # Create a DataFrame with the results
    sum_bysec_tab = pd.DataFrame(sum_bysec, index=industries + ['All'])

    # Save the table to CSV
    sum_bysec_tab.to_csv(os.path.join(output_path, f'{year}', 'moments', file_label))

def summary_tables(full_df, output_path, variable_type: str, start, end):
    """
    General function to generate summary tables for sales or purchases for all years.
    
    Parameters:
        full_df (pd.DataFrame): Full DataFrame containing data for all years.
        output_path (str): Path to save the summary tables.
        variable_type (str): Either 'sales' or 'purchases'.
    """
    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year].copy()
        summary_table_by_year(df_year, year, variable_type, output_path)