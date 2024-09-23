import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse.linalg as spsl
import scipy.sparse as sps
import os 

#%% 1. Upstreamness

def calculate_technical_coeffs(df_year):

    # calculate allocation coefficients
    df_year['coef'] =  - df_year['sales_ij'] / df_year['turnover_i']

    return df_year['coef']

def define_graph(df_year):

    edge_list = df_year[['vat_i', 'vat_j', 'coef']].values.tolist()
    G = nx.DiGraph()
    G.add_weighted_edges_from(edge_list)

    return G

def create_leontief_mat(G):

    # create Ghosh matrix as a sparse matrix
    I_A = nx.to_scipy_sparse_array(G, dtype=np.float64, format='csc')
    
    I_A=sps.lil_matrix(I_A)
    for firm in range(0,len(G)):
        I_A[firm,firm] = 1
    I_A=sps.csc_matrix(I_A)

    return I_A

def solve_linear_system(I_A, b, year):

    U, info = spsl.lgmres(I_A, b, x0=b, atol=1e-15)
    print(f'Year {year}, info = {info}')

    return U

def construct_upstreamness_df(full_df, output_path, start, end):

    results = []

    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year].copy()

        # calculate technical coefficients
        df_year['coef'] = calculate_technical_coeffs(df_year)

        # define the graph
        G = define_graph(df_year)

        # create Ghosh matrix as a sparse matrix
        I_A = create_leontief_mat(G)

        # define the b vector
        b = np.ones((len(G),1))

        # Solve the linear system
        U = solve_linear_system(I_A, b, year)

        # create the dataframe for single year
        for vat, upstreamness in zip([node for node in G.nodes()], U):
            results.append({
                'year': year,
                'vat': vat,
                'upstreamness': upstreamness
            })
        
    final_df = pd.DataFrame(results)
    final_df.to_csv(os.path.join(output_path,'upstreamness.csv'), index=False)


# Master function
def upstreamness(full_df, output_path, start, end):

    # Construct the upstreamness dataframe
    construct_upstreamness_df(full_df, output_path, start, end)

