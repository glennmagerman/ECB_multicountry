import os
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse.linalg as spsl
import scipy.sparse as sps

def calculate_technical_coeffs(df_year, variable: str):

    # calculate allocation coefficients
    df_year['coef'] =  - df_year['sales_ij'] / df_year[variable]

    return df_year['coef']

def define_graph(df_year, weighted=True):

    G = nx.DiGraph()
    if weighted:
        edge_list = df_year[['vat_i', 'vat_j', 'coef']].values.tolist()
        G.add_weighted_edges_from(edge_list)
    else:
        edge_list = df_year[['vat_i', 'vat_j']].values.tolist()
        G.add_edges_from(edge_list)

    return G

def create_leontief_mat(G):

    # create matrix as a sparse matrix
    I_A = nx.to_scipy_sparse_array(G, dtype=np.float64, format='csc')
    
    I_A=sps.lil_matrix(I_A)
    for firm in range(0,len(G)):
        I_A[firm,firm] = 1
    I_A=sps.csc_matrix(I_A)

    return I_A

def solve_linear_system(I_A, b, measure, year, x0=None):

    U, info = spsl.lgmres(I_A, b, x0=x0)
    print(f'{measure}: Year {year}, info = {info}')

    return U

def upstreamness(full_df, start, end):
    
    df_up = full_df.copy()
    df_up['turnover_i'] = np.where( (df_up['network_sales_i'] == 
                                        df_up['turnover_i']) , df_up['turnover_i'] + 1, df_up['turnover_i']  )

    # add some final demand (1 euro) to the sales if network sales = turnover of i
    df_up['turnover_j'] = np.where( (df_up['network_sales_j'] == 
                                        df_up['turnover_j']) , df_up['turnover_j'] + 1, df_up['turnover_j']  )
    

    results = []
    
    for year in range(start, end+1):
        
        df_year = df_up[df_up['year'] == year].copy()
        
        # calculate technical coefficients
        df_year['coef'] = calculate_technical_coeffs(df_year, 'turnover_i')

        # define the graph
        G = define_graph(df_year)

        # create Ghosh matrix as a sparse matrix
        I_A = create_leontief_mat(G)

        # define the b vector
        b = np.ones((len(G),1))

        # Solve the linear system
        U = solve_linear_system(I_A, b, 'Upstreamness', year, x0=b)
        
        # create the dataframe for single year
        for vat, upstreamness in zip([node for node in G.nodes()], U):
            results.append({
                'year': year,
                'vat': vat,
                'upstreamness': upstreamness
            })
            
    upstreamness_df = pd.DataFrame(results)
    
    return upstreamness_df

def downstreamness(full_df, start, end):
    
    df_down = full_df.copy()
    df_down['turnover_j'] = np.where(df_down['network_purch_j'] > df_down['turnover_j'],df_down['network_purch_j'],df_down['turnover_j'])
    df_down['inputs_j'] = np.where(df_down['inputs_j'] >= df_down['turnover_j'], df_down['turnover_j'], df_down['inputs_j'] )

    results = []
    
    for year in range(start, end+1):
        
        df_year = df_down[df_down['year'] == year].copy()
        
        # remove firms with missing turnover
        df_year = df_year[~df_year['turnover_j'].isna()].copy()
        
        # calculate technical coefficients
        df_year['coef'] = calculate_technical_coeffs(df_year, 'turnover_j')

        # define the graph
        G = define_graph(df_year)

        # create Ghosh matrix as a sparse matrix
        I_A = create_leontief_mat(G)

        # define the b vector
        b = np.ones((len(G),1))

        # Solve the linear system
        U = solve_linear_system(I_A.transpose(), b, 'Downstreamness', year, x0=b)
        
        # create the dataframe for single year
        for vat, downstreamness in zip([node for node in G.nodes()], U):
            results.append({
                'year': year,
                'vat': vat,
                'downstreamness': downstreamness
            })
            
    downstreamness_df = pd.DataFrame(results)
    
    return downstreamness_df

def domar_weights(full_df, start, end):
    
    results = []
    
    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year].copy()
        
        # 1. Build the graph
        G = define_graph(df_year, weighted=False)
        
        # 2. Build firm-level table using BOTH i and j roles
        
        # Supplier side
        firms_i = (
            df_year[['vat_i', 'sales_to_fd_i', 'turnover_i']]
            .rename(columns={
                'vat_i': 'vat',
                'sales_to_fd_i': 'sales_to_fd',
                'turnover_i': 'turnover'
            })
        )
        
        # Customer side
        firms_j = (
            df_year[['vat_j', 'sales_to_fd_j', 'turnover_j']]
            .rename(columns={
                'vat_j': 'vat',
                'sales_to_fd_j': 'sales_to_fd',
                'turnover_j': 'turnover'
            })
        )
        
        # Combine
        firms = pd.concat([firms_i, firms_j], ignore_index=True)
        
        # 3. Aggregate (firm-level)
        firms = (
            firms
            .groupby('vat', as_index=False)
            .agg({
                'sales_to_fd': 'first',
                'turnover': 'first'
            })
        )
        
        # 4. Attach node attributes
        nx.set_node_attributes(G, dict(zip(firms['vat'], firms['sales_to_fd'])), 'sales_to_fd')
        nx.set_node_attributes(G, dict(zip(firms['vat'], firms['turnover'])), 'turnover')
        
        # 5. GDP
        gdp = firms['sales_to_fd'].sum()
        
        # 6. Domar and turnover in output
        for node in G.nodes():
            turnover = G.nodes[node].get('turnover', np.nan)
            fd = G.nodes[node].get('sales_to_fd', np.nan)
            
            if pd.isna(turnover):
                domar = np.nan
            else:
                domar = turnover / gdp
            
            results.append({
                'year': year,
                'vat': node,
                'turnover': turnover,
                'sales_to_fd': fd,
                'domar': domar
            })
        print(f'Domar weights: Year {year}')
    
    domar_df = pd.DataFrame(results)
    return domar_df

def centrality(
    full_df,
    start,
    end,
    alpha_const = False         
):
    results = []

    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year].copy()
        
        if alpha_const:
            alpha = 0.2
            df_year['coef'] = calculate_technical_coeffs(df_year, 'inputs_j') * (1-alpha)
            #measure = 'Centrality (constant alpha)'
        else:
            df_year['coef'] = calculate_technical_coeffs(df_year, 'turnover_j')
            #measure = 'Centrality'
            
        # define the graph
        G = define_graph(df_year)

        # create Leontief matrix as a sparse matrix
        I_A = create_leontief_mat(G)

        # build per‐firm inputs and turnover
        df_i = df_year.groupby('vat_i')[['inputs_i','turnover_i', 'sales_to_fd_i']] \
              .first() \
              .rename(columns={'inputs_i':'inputs','turnover_i':'turnover', 'sales_to_fd_i':'sales_to_fd'})
        df_j = df_year.groupby('vat_j')[['inputs_j','turnover_j', 'sales_to_fd_j']] \
                    .first() \
                    .rename(columns={'inputs_j':'inputs','turnover_j':'turnover', 'sales_to_fd_j':'sales_to_fd'})

        # combine, so each vat appears once with whatever data it has
        firm_stats = pd.concat([df_i, df_j]).groupby(level=0).first()
        firm_stats = firm_stats[~firm_stats['sales_to_fd'].isna()]
        
        # push them into G
        #  create dicts of values keyed by vat
        salesfd_dict = firm_stats['sales_to_fd'].to_dict()
        nx.set_node_attributes(G, salesfd_dict, name='b')
        
        node_order = sorted(G.nodes())
        if not alpha_const:
            inputs_dict   = firm_stats['inputs'].to_dict()
            turnover_dict = firm_stats['turnover'].to_dict()
            nx.set_node_attributes(G, inputs_dict,   name='inputs')
            nx.set_node_attributes(G, turnover_dict, name='turnover')
            # compute alpha = 1 - inputs/turnover
            alpha_dict = {
                vat: (1 - inputs_dict.get(vat) / turnover_dict.get(vat))
                for vat in G.nodes()
            }
            nx.set_node_attributes(G, alpha_dict, name='alpha')
            alpha = pd.Series(
                nx.get_node_attributes(G, 'alpha'),
                index=node_order
            )
        
        # compute GDP
        b_dict   = nx.get_node_attributes(G, 'b')
        b_series = pd.Series([b_dict.get(node, 1) for node in node_order], index=node_order).fillna(0)
        b = b_series.values.reshape(-1, 1)
        gdp  = np.nansum(b)
        
        # Solve the linear system
        r = solve_linear_system(I_A, b, 'Centrality', year)
        r_series = pd.Series(r.flatten(), index=node_order)
        #r_dict = nx.get_node_attributes(G, 'turnover')
        #r_series = pd.Series([r_dict.get(node, 1) for node in node_order], index=node_order)
        
        # compute centrality
        C = alpha/gdp * r_series
        
        #if alpha_const:
        #    print(f'Centrality (constant alpha): Year {year}')
        #else:
        #    print(f'Centrality: Year {year}')
            
        # create the dataframe for single year
        for vat, centrality in zip([node for node in G.nodes()], C):
            results.append({
                'year': year,
                'vat': vat,
                'centrality': centrality
            })

    return pd.DataFrame(results)

def centrality_katz(full_df, start, end, alpha=0.2, katz_tol=1e-6, katz_max_iter=1000):
    
    results = []

    for year in range(start, end + 1):
        df_year = full_df[full_df['year'] == year].copy()

        # --- 1. Build coefficients and graph ---
        #df_year['coef'] = calculate_technical_coeffs(df_year, 'inputs_j')
        df_year['coef'] =  df_year['sales_ij'] / df_year['inputs_j']
        G = define_graph(df_year)  # must set edge weights = 'coef'

        # --- 2. Build firm-level stats (inputs, turnover, sales_to_fd) ---
        df_i = (
            df_year.groupby('vat_i')[['inputs_i', 'turnover_i', 'sales_to_fd_i']]
                  .first()
                  .rename(columns={
                      'inputs_i': 'inputs',
                      'turnover_i': 'turnover',
                      'sales_to_fd_i': 'sales_to_fd'
                  })
        )

        df_j = (
            df_year.groupby('vat_j')[['inputs_j', 'turnover_j', 'sales_to_fd_j']]
                  .first()
                  .rename(columns={
                      'inputs_j': 'inputs',
                      'turnover_j': 'turnover',
                      'sales_to_fd_j': 'sales_to_fd'
                  })
        )

        firm_stats = pd.concat([df_i, df_j]).groupby(level=0).first()
        firm_stats = firm_stats[~firm_stats['sales_to_fd'].isna()]

        # Only keep nodes that exist in G
        node_order = sorted(G.nodes())
        firm_stats = firm_stats.loc[firm_stats.index.intersection(node_order)]

        # --- 3. Construct β_i = sales_to_fd_i ---
        beta_series = firm_stats['sales_to_fd'].reindex(node_order).fillna(0.0)
        beta_dict   = beta_series.to_dict()

        # --- 4. Compute Katz/BONACICH centrality with node-specific β ---
        katz_dict = nx.katz_centrality(
            G,
            alpha=alpha,
            beta=beta_dict,
            weight='weight',      
            normalized=False,
            max_iter=katz_max_iter,
            tol=katz_tol
        )

        gdp  = np.nansum(beta_series)
        # --- 5. Store results ---
        for vat, c_val in katz_dict.items():
            results.append({
                'year': year,
                'vat': vat,
                'centrality': 0.2 * c_val/gdp
            })

    return pd.DataFrame(results)


def net_sales_inputs(full_df, start, end):
    """
    Build firm-year averages of network sales and net purchases
    using info from both supplier (i) and customer (j) roles.

    Expects columns:
      - 'year'
      - 'vat_i', 'network_sales_i', 'net_purch_i'
      - 'vat_j', 'net_sales_j', 'net_purch_j'
    """
    results = []

    for year in range(start, end + 1):
        df_year = full_df[full_df['year'] == year].copy()

        # Supplier side (i)
        firms_i = (
            df_year[['vat_i', 'year','network_sales_i', 'network_purch_i','inputs_i']]
            .dropna(subset=['vat_i'])
            .rename(columns={
                'vat_i': 'vat',
                'network_sales_i': 'network_sales',
                'network_purch_i': 'network_purch',
                'inputs_i': 'inputs'
            })
        )

        # Customer side (j)
        firms_j = (
            df_year[['vat_j', 'year','network_sales_j', 'network_purch_j','inputs_j']]
            .dropna(subset=['vat_j'])
            .rename(columns={
                'vat_j': 'vat',
                'network_sales_j': 'network_sales',
                'network_purch_j': 'network_purch',
                'inputs_j': 'inputs'
            })
        )

        # Stack i and j so firms that only appear as j are included
        firms = pd.concat([firms_i, firms_j], ignore_index=True)

        # Aggregate to firm-year
        firm_year = (
            firms
            .groupby(['vat', 'year'], as_index=False)
            .agg(
                network_sales=('network_sales', 'mean'),
                network_purch=('network_purch', 'mean'),
                inputs=('inputs', 'mean')
            )
        )

        results.append(firm_year)
        print(f'Network aggregates: Year {year}')

    if not results:
        return pd.DataFrame(columns=['vat', 'year', 'network_sales', 'network_purch', 'inputs'])

    avg_df = pd.concat(results, ignore_index=True)
    return avg_df
    

def degrees(full_df, start, end):
    
    results = []
    for year in range(start, end+1):
        
        df_year = full_df[full_df['year'] == year].copy()
        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())
        
        # Extract outdegree and indegree as dictionaries
        outdeg = dict(G_year.out_degree())
        indeg = dict(G_year.in_degree())
        
        # Get all nodes in the current year's graph
        nodes = set(G_year.nodes())
        
        # For each node, record its degrees along with the firm (vat) and year
        for node in nodes:
            results.append({
                'vat': node,
                'year': year,
                'outdeg': outdeg.get(node, 0),
                'indeg': indeg.get(node, 0)
            })
        print(f'Degrees: Year {year}')
        
    degrees_df = pd.DataFrame(results)
    
    return degrees_df

def wavg_mkt_share(full_df, start, end):
    
    results = []
    
    for year in range(start, end + 1):
        df_year = full_df[full_df['year'] == year].copy()
        
        df_year['mkt_share'] = df_year['sales_ij'] / df_year['network_purch_j']
        df_year['sales_share'] = df_year['sales_ij'] / df_year['network_sales_i']
        df_year['avg_mkt_share'] = df_year['mkt_share'] ** df_year['sales_share']
        
        agg = (
            df_year
            .groupby(['vat_i', 'year'], as_index=False)['avg_mkt_share']
            .prod()
            .rename(columns={
                'vat_i': 'vat'
            })
        )
        results.append(agg)
        print(f'Weighted average market share: Year {year}')
        
    if not results:
        return pd.DataFrame(columns=['vat', 'year', 'avg_mkt_share'])

    wavg_df = pd.concat(results, ignore_index=True)
    return wavg_df

def firm_nace_from_edges(full_df, start, end):
    """
    Build a firm-year NACE table using both supplier (i) and customer (j) roles.
    """

    results = []

    for year in range(start, end + 1):
        df_year = full_df[full_df['year'] == year]

        if df_year.empty:
            continue

        # Supplier side
        nace_i = (
            df_year[['vat_i', 'nace_i']]
            .rename(columns={'vat_i': 'vat', 'nace_i': 'nace'})
            .dropna(subset=['vat'])
        )

        # Customer side
        nace_j = (
            df_year[['vat_j', 'nace_j']]
            .rename(columns={'vat_j': 'vat', 'nace_j': 'nace'})
            .dropna(subset=['vat'])
        )

        # Combine
        nace_all = pd.concat([nace_i, nace_j], ignore_index=True)

        # Deduplicate: firms might appear multiple times in edges
        firm_nace = (
            nace_all
            .groupby('vat')['nace']
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .reset_index()
        )

        firm_nace['year'] = year
        results.append(firm_nace)

    if not results:
        return pd.DataFrame(columns=['vat', 'year', 'nace'])

    return pd.concat(results, ignore_index=True)

def create_panel(tmp_path, output_path):
    
    full_df = pd.read_parquet(os.path.join(tmp_path, 'full_data_cleaned.parquet'))
    start = full_df['year'].min()
    end = full_df['year'].max()
    
    upstreamness_df = upstreamness(full_df, start, end)
    
    downstreamness_df = downstreamness(full_df, start, end)
    
    centrality_df = centrality(full_df, start, end, alpha_const=True)
    #centrality_df = centrality_katz(full_df, start, end, alpha=0.8)
    
    degrees_df = degrees(full_df, start, end)
    
    domar_df = domar_weights(full_df, start, end)
    
    net_df = net_sales_inputs(full_df, start, end)
    
    wavg_df = wavg_mkt_share(full_df, start, end)
    
    panel_df = upstreamness_df.merge(downstreamness_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(centrality_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df  = panel_df.merge(degrees_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(domar_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(net_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(wavg_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(firm_nace_from_edges(full_df, start, end), on=['vat', 'year'], how='left',validate='1:1')
    panel_df['nace2d'] = panel_df['nace'].str[:2]
    
    # save
    panel_df.to_parquet(os.path.join(output_path, 'panel.parquet'),engine='pyarrow')
    full_df.drop(columns=['network_sales_i', 'network_purch_i', 'network_sales_j', 'network_purch_j'])
    full_df.to_parquet(os.path.join(output_path, 'full_data_cleaned.parquet'),engine='pyarrow')
