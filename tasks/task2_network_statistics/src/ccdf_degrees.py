import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import powerlaw
import os 

def plot_CCDF_degree_distr(degree, degree_type: str, output_path, year):
    """
    General function to plot the CCDF of degree distribution for both out-degree and in-degree.
    
    Parameters:
        degree: The degree values (either in_degree or out_degree).
        degree_type (str): Either 'outdegree' or 'indegree' to specify the type of degree.
        output_path: Path to save the output plot.
        year: The year for which the plot is being generated.
    """
    # Determine labels and filenames based on degree_type
    if degree_type == 'outdegree':
        label = 'customers'
        nickname = 'out'
    elif degree_type == 'indegree':
        label = 'suppliers'
        nickname = 'in'
    else:
        raise ValueError("degree_type must be either 'outdegree' or 'indegree'")

    # CCDF of the degree distribution
    degree_distr = np.unique(np.array(list(degree.values())))
    fit = powerlaw.Fit(degree_distr)
    coeff = (fit.power_law.alpha - 1)
    xmin = fit.xmin
    powerlaw.plot_ccdf(degree_distr, marker='.', linewidth=0)
    
    # Add text box with fit information
    textstr = '\n'.join((
        f'Tail exponent:  {coeff:.2f}',
        f'$x_{{min}}$ = {xmin}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.2, textstr, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    # Set plot labels and save figure
    plt.xlabel(f'Number of {label}')
    plt.ylabel('CCDF')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f'{year}', 'CCDF', f'{nickname}deg_ccdf_{year}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# CCDF of degree distributions per year
def CCDF_degree_per_year(full_df, output_path, start, end):

    for year in range(start, end+1):

        df_year = full_df[full_df['year'] == year].copy()
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        # Find out-degree and in-degree and add it to the dataframe
        out_degree = {node: degree for node, degree in G_year.out_degree() if degree > 0}
        in_degree = {node: degree for node, degree in G_year.in_degree() if degree > 0}
            
        # CCDF of out-degree distribution
        plot_CCDF_degree_distr(out_degree, 'outdegree', output_path, year)

        # CCDF of in-degree distribution
        plot_CCDF_degree_distr(in_degree, 'indegree', output_path, year)
    

# Master function
def master_CCDF_degrees(full_df, output_path, start, end):

    # Generate the CCDF of degree distributions for each year
    CCDF_degree_per_year(full_df, output_path, start, end)