import gc
import os
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


def calculate_technical_coeffs(df_year, variable: str):

    # calculate allocation coefficients
    df_year['coef'] = -df_year['sales_ij'] / df_year[variable]

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


def create_leontief_mat(G, node_order=None):

    if node_order is None:
        node_order = list(G.nodes())

    I_A = nx.to_scipy_sparse_array(
        G,
        nodelist=node_order,
        dtype=np.float64,
        format='csc'
    )

    I_A = sps.lil_matrix(I_A)
    I_A.setdiag(1.0)
    I_A = sps.csc_matrix(I_A)

    return I_A


def solve_linear_system(I_A, b, measure, year, x0=None):

    U, info = spsl.lgmres(I_A, b, x0=x0, maxiter=2000)
    print(f'{measure}: Year {year}, info = {info}')

    return U


def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def preprocess_partitioned_input(tmp_path):
    input_file = os.path.join(tmp_path, 'full_data_cleaned.parquet')
    partitioned_dir = os.path.join(tmp_path, 'full_data_by_year')

    ensure_clean_dir(partitioned_dir)
    full_df = pd.read_parquet(input_file)
    full_df.to_parquet(partitioned_dir, engine='pyarrow', partition_cols=['year'], index=False)
    del full_df
    gc.collect()

    return partitioned_dir


def get_available_years(input_base_path):
    years = []
    for name in os.listdir(input_base_path):
        if name.startswith('year='):
            year_str = name.split('=', 1)[1]
            if year_str.isdigit():
                years.append(int(year_str))

    if not years:
        raise ValueError(f'No year partitions found in {input_base_path}')

    years.sort()
    return years


def read_year_partition(input_base_path, year, columns):
    year_path = os.path.join(input_base_path, f'year={year}')
    if not os.path.exists(year_path):
        return pd.DataFrame(columns=columns)
    df_year = pd.read_parquet(year_path, columns=columns)
    if 'year' not in df_year.columns:
        df_year['year'] = year
    return df_year


def write_metric_year(df_year_metric, metric_path, year):
    os.makedirs(metric_path, exist_ok=True)
    df_year_metric.to_parquet(os.path.join(metric_path, f'year={year}.parquet'), engine='pyarrow', index=False)


def read_window(input_base_path, years, columns):
    frames = []
    for y in years:
        part = read_year_partition(input_base_path, y, columns)
        if not part.empty:
            frames.append(part)

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def upstreamness(input_base_path, start, end, metric_path, window=1):

    col_name = 'upstreamness' if window == 1 else 'avg_upstreamness'

    for year in range(start + window - 1, end + 1):
        cols = ['vat_i', 'vat_j', 'sales_ij', 'turnover_i', 'network_sales_i', 'turnover_j', 'network_sales_j']
        if window == 1:
            df_year = read_year_partition(input_base_path, year, cols)
            df_year['turnover_i'] = np.where(
                df_year['network_sales_i'] == df_year['turnover_i'],
                df_year['turnover_i'] + 1,
                df_year['turnover_i'],
            )
            df_year['turnover_j'] = np.where(
                df_year['network_sales_j'] == df_year['turnover_j'],
                df_year['turnover_j'] + 1,
                df_year['turnover_j'],
            )
        else:
            window_years = list(range(year - window + 1, year + 1))
            df_window = read_window(input_base_path, window_years, cols)
            df_window['turnover_i'] = np.where(
                df_window['network_sales_i'] == df_window['turnover_i'],
                df_window['turnover_i'] + 1,
                df_window['turnover_i'],
            )
            df_window['turnover_j'] = np.where(
                df_window['network_sales_j'] == df_window['turnover_j'],
                df_window['turnover_j'] + 1,
                df_window['turnover_j'],
            )
            df_sales = df_window.groupby(['vat_i', 'vat_j'], as_index=False)['sales_ij'].sum()
            df_turnover_unique = df_window[['year', 'vat_i', 'turnover_i']].drop_duplicates()
            df_turnover = df_turnover_unique.groupby('vat_i', as_index=False)['turnover_i'].sum()
            df_year = pd.merge(df_sales, df_turnover, on='vat_i', how='left')

        if df_year.empty:
            write_metric_year(pd.DataFrame(columns=['year', 'vat', col_name]), metric_path, year)
            continue

        df_year['coef'] = calculate_technical_coeffs(df_year, 'turnover_i')
        G = define_graph(df_year)
        I_A = create_leontief_mat(G)
        b = np.ones((len(G), 1))

        label = 'Average upstreamness' if window > 1 else 'Upstreamness'
        U = solve_linear_system(I_A, b, label, year, x0=b)

        year_result = pd.DataFrame({'vat': list(G.nodes()), col_name: U.flatten()})
        year_result['year'] = year
        year_result = year_result[['year', 'vat', col_name]]
        write_metric_year(year_result, metric_path, year)

        del df_year, G, I_A, b, U, year_result
        if window > 1:
            del df_window, df_sales, df_turnover_unique, df_turnover
        gc.collect()


def downstreamness(input_base_path, start, end, metric_path, window=1):

    col_name = 'downstreamness' if window == 1 else 'avg_downstreamness'

    for year in range(start + window - 1, end + 1):
        cols = ['vat_i', 'vat_j', 'sales_ij', 'turnover_j', 'network_purch_j']
        if window == 1:
            df_year = read_year_partition(input_base_path, year, cols)
            df_year['turnover_j'] = np.where(
                df_year['turnover_j'].isna(),
                df_year['network_purch_j'] + 1,
                df_year['turnover_j'],
            )
            df_year['turnover_j'] = np.where(
                df_year['network_purch_j'] > df_year['turnover_j'],
                df_year['network_purch_j'] + 1,
                df_year['turnover_j'],
            )
        else:
            window_years = list(range(year - window + 1, year + 1))
            df_window = read_window(input_base_path, window_years, cols)
            df_window['turnover_j'] = np.where(
                df_window['turnover_j'].isna(),
                df_window['network_purch_j'] + 1,
                df_window['turnover_j'],
            )
            df_window['turnover_j'] = np.where(
                df_window['network_purch_j'] > df_window['turnover_j'],
                df_window['network_purch_j'] + 1,
                df_window['turnover_j'],
            )
            df_sales = df_window.groupby(['vat_i', 'vat_j'], as_index=False)['sales_ij'].sum()
            df_turnover_unique = df_window[['year', 'vat_j', 'turnover_j']].drop_duplicates()
            df_turnover = df_turnover_unique.groupby('vat_j', as_index=False)['turnover_j'].sum()
            df_year = pd.merge(df_sales, df_turnover, on='vat_j', how='left')

        if df_year.empty:
            write_metric_year(pd.DataFrame(columns=['year', 'vat', col_name]), metric_path, year)
            continue

        df_year = df_year[(df_year['turnover_j'].notna()) & (df_year['turnover_j'] != 0)].copy()
        df_year['coef'] = calculate_technical_coeffs(df_year, 'turnover_j')
        G = define_graph(df_year)
        I_A = create_leontief_mat(G)
        b = np.ones((len(G), 1))

        label = 'Average downstreamness' if window > 1 else 'Downstreamness'
        U = solve_linear_system(I_A.transpose(), b, label, year, x0=b)

        year_result = pd.DataFrame({'vat': list(G.nodes()), col_name: U.flatten()})
        year_result['year'] = year
        year_result = year_result[['year', 'vat', col_name]]
        write_metric_year(year_result, metric_path, year)

        del df_year, G, I_A, b, U, year_result
        if window > 1:
            del df_window, df_sales, df_turnover_unique, df_turnover
        gc.collect()


def domar_weights(input_base_path, start, end, metric_path):

    for year in range(start, end + 1):
        cols = ['vat_i', 'vat_j', 'sales_to_fd_i', 'turnover_i', 'sales_to_fd_j', 'turnover_j']
        df_year = read_year_partition(input_base_path, year, cols)

        if df_year.empty:
            write_metric_year(pd.DataFrame(columns=['year', 'vat', 'turnover', 'sales_to_fd', 'domar']), metric_path, year)
            continue

        # 1. Build the graph
        G = define_graph(df_year, weighted=False)

        # 2. Build firm-level table using BOTH i and j roles

        # Supplier side
        firms_i = (
            df_year[['vat_i', 'sales_to_fd_i', 'turnover_i']]
            .rename(
                columns={
                    'vat_i': 'vat',
                    'sales_to_fd_i': 'sales_to_fd',
                    'turnover_i': 'turnover',
                }
            )
        )

        # Customer side
        firms_j = (
            df_year[['vat_j', 'sales_to_fd_j', 'turnover_j']]
            .rename(
                columns={
                    'vat_j': 'vat',
                    'sales_to_fd_j': 'sales_to_fd',
                    'turnover_j': 'turnover',
                }
            )
        )

        # Combine
        firms = pd.concat([firms_i, firms_j], ignore_index=True)

        # 3. Aggregate (firm-level)
        firms = firms.groupby('vat', as_index=False).agg({'sales_to_fd': 'first', 'turnover': 'first'})

        # 4. Attach node attributes
        nx.set_node_attributes(G, dict(zip(firms['vat'], firms['sales_to_fd'])), 'sales_to_fd')
        nx.set_node_attributes(G, dict(zip(firms['vat'], firms['turnover'])), 'turnover')

        # 5. GDP
        gdp = firms['sales_to_fd'].sum()

        year_rows = []

        # 6. Domar and turnover in output
        for node in G.nodes():
            turnover = G.nodes[node].get('turnover', np.nan)
            fd = G.nodes[node].get('sales_to_fd', np.nan)

            if pd.isna(turnover):
                domar = np.nan
            else:
                domar = turnover / gdp

            year_rows.append({'year': year, 'vat': node, 'turnover': turnover, 'sales_to_fd': fd, 'domar': domar})
        print(f'Domar weights: Year {year}')

        write_metric_year(pd.DataFrame(year_rows), metric_path, year)
        del df_year, G, firms_i, firms_j, firms, year_rows
        gc.collect()


def centrality(input_base_path, start, end, metric_path, alpha_const=False, window=1):

    col_name = 'centrality' if window == 1 else 'avg_centrality'

    for year in range(start + window - 1, end + 1):
        cols = [
            'vat_i', 'vat_j', 'sales_ij',
            'inputs_i', 'turnover_i', 'sales_to_fd_i',
            'inputs_j', 'turnover_j', 'sales_to_fd_j',
        ]

        if window == 1:
            df_year = read_year_partition(input_base_path, year, cols)
        else:
            window_years = list(range(year - window + 1, year + 1))
            df_window = read_window(input_base_path, window_years, cols)
            df_sales = df_window.groupby(['vat_i', 'vat_j'], as_index=False)['sales_ij'].sum()

            # Aggregate turnover and inputs for both i and j sides
            for side, vat_col in [('i', 'vat_i'), ('j', 'vat_j')]:
                side_cols = [f'inputs_{side}', f'turnover_{side}', f'sales_to_fd_{side}']
                df_unique = df_window[['year', vat_col] + side_cols].drop_duplicates()
                df_agg_side = df_unique.groupby(vat_col, as_index=False)[side_cols].sum()
                df_sales = pd.merge(df_sales, df_agg_side, on=vat_col, how='left')

            df_year = df_sales

        if df_year.empty:
            write_metric_year(pd.DataFrame(columns=['year', 'vat', col_name]), metric_path, year)
            continue

        if alpha_const:
            alpha = 0.2
            df_year['coef'] = calculate_technical_coeffs(df_year, 'inputs_j') * (1 - alpha)
        else:
            df_year['coef'] = calculate_technical_coeffs(df_year, 'turnover_j')

        G = define_graph(df_year)
        
        node_order = sorted(G.nodes())
        I_A = create_leontief_mat(G, node_order)

        # build per-firm stats
        df_i = (
            df_year.groupby('vat_i')[['inputs_i', 'turnover_i', 'sales_to_fd_i']]
            .first()
            .rename(columns={'inputs_i': 'inputs', 'turnover_i': 'turnover', 'sales_to_fd_i': 'sales_to_fd'})
        )
        df_j = (
            df_year.groupby('vat_j')[['inputs_j', 'turnover_j', 'sales_to_fd_j']]
            .first()
            .rename(columns={'inputs_j': 'inputs', 'turnover_j': 'turnover', 'sales_to_fd_j': 'sales_to_fd'})
        )

        firm_stats = pd.concat([df_i, df_j]).groupby(level=0).first()
        firm_stats = firm_stats[~firm_stats['sales_to_fd'].isna()]

        gdp = firm_stats['sales_to_fd'].sum()

        salesfd_dict = firm_stats['sales_to_fd'].to_dict()
        nx.set_node_attributes(G, salesfd_dict, name='b')

        node_order = sorted(G.nodes())

        if not alpha_const:
            inputs_dict = firm_stats['inputs'].to_dict()
            turnover_dict = firm_stats['turnover'].to_dict()
            nx.set_node_attributes(G, inputs_dict, name='inputs')
            nx.set_node_attributes(G, turnover_dict, name='turnover')
            alpha_dict = {
                vat: (1 - inputs_dict.get(vat, 0) / turnover_dict.get(vat, 1))
                for vat in G.nodes()
            }
            nx.set_node_attributes(G, alpha_dict, name='alpha')
            alpha = pd.Series(nx.get_node_attributes(G, 'alpha'), index=node_order)

        b_dict = nx.get_node_attributes(G, 'b')
        b_series = pd.Series([b_dict.get(node, 0) for node in node_order], index=node_order).fillna(0)
        b = b_series.values.reshape(-1, 1)

        label = 'Average centrality' if window > 1 else 'Centrality'
        r = solve_linear_system(I_A, b, label, year)
        r_series = pd.Series(r.flatten(), index=node_order)

        C = alpha / gdp * r_series

        year_result = pd.DataFrame({'vat': node_order, col_name: C.values})
        year_result['year'] = year
        year_result = year_result[['year', 'vat', col_name]]
        write_metric_year(year_result, metric_path, year)

        del df_year, G, I_A, df_i, df_j, firm_stats, b, r, r_series, C, year_result
        if window > 1:
            del df_window, df_sales, df_unique, df_agg_side
        gc.collect()


def net_sales_inputs(input_base_path, start, end, metric_path):
    """
    Build firm-year averages of network sales and net purchases
    using info from both supplier (i) and customer (j) roles.
    """

    for year in range(start, end + 1):
        cols = ['vat_i', 'vat_j', 'network_sales_i', 'network_purch_i', 'inputs_i', 'network_sales_j', 'network_purch_j', 'inputs_j']
        df_year = read_year_partition(input_base_path, year, cols)

        if df_year.empty:
            write_metric_year(pd.DataFrame(columns=['vat', 'year', 'network_sales', 'network_purch', 'inputs']), metric_path, year)
            continue

        # Supplier side (i)
        firms_i = (
            df_year[['vat_i', 'network_sales_i', 'network_purch_i', 'inputs_i']]
            .dropna(subset=['vat_i'])
            .rename(
                columns={
                    'vat_i': 'vat',
                    'network_sales_i': 'network_sales',
                    'network_purch_i': 'network_purch',
                    'inputs_i': 'inputs',
                }
            )
        )
        firms_i['year'] = year

        # Customer side (j)
        firms_j = (
            df_year[['vat_j', 'network_sales_j', 'network_purch_j', 'inputs_j']]
            .dropna(subset=['vat_j'])
            .rename(
                columns={
                    'vat_j': 'vat',
                    'network_sales_j': 'network_sales',
                    'network_purch_j': 'network_purch',
                    'inputs_j': 'inputs',
                }
            )
        )
        firms_j['year'] = year

        # Stack i and j so firms that only appear as j are included
        firms = pd.concat([firms_i, firms_j], ignore_index=True)

        # Aggregate to firm-year
        firm_year = firms.groupby(['vat', 'year'], as_index=False).agg(
            network_sales=('network_sales', 'mean'),
            network_purch=('network_purch', 'mean'),
            inputs=('inputs', 'mean'),
        )

        write_metric_year(firm_year, metric_path, year)
        print(f'Network aggregates: Year {year}')
        del df_year, firms_i, firms_j, firms, firm_year
        gc.collect()


def degrees(input_base_path, start, end, metric_path):

    for year in range(start, end + 1):
        cols = ['vat_i', 'vat_j']
        df_year = read_year_partition(input_base_path, year, cols)

        if df_year.empty:
            write_metric_year(pd.DataFrame(columns=['vat', 'year', 'outdeg', 'indeg']), metric_path, year)
            continue

        G_year = nx.from_pandas_edgelist(df_year, 'vat_i', 'vat_j', create_using=nx.DiGraph())

        # Extract outdegree and indegree as dictionaries
        outdeg = dict(G_year.out_degree())
        indeg = dict(G_year.in_degree())

        # Get all nodes in the current year's graph
        nodes = set(G_year.nodes())

        year_rows = []

        # For each node, record its degrees along with the firm (vat) and year
        for node in nodes:
            year_rows.append({'vat': node, 'year': year, 'outdeg': outdeg.get(node, 0), 'indeg': indeg.get(node, 0)})
        print(f'Degrees: Year {year}')

        write_metric_year(pd.DataFrame(year_rows), metric_path, year)
        del df_year, G_year, outdeg, indeg, nodes, year_rows
        gc.collect()


def wavg_mkt_share(input_base_path, start, end, metric_path):

    for year in range(start, end + 1):
        cols = ['vat_i', 'sales_ij', 'network_purch_j', 'network_sales_i']
        df_year = read_year_partition(input_base_path, year, cols)

        if df_year.empty:
            write_metric_year(pd.DataFrame(columns=['vat', 'year', 'avg_mkt_share']), metric_path, year)
            continue

        df_year['mkt_share'] = df_year['sales_ij'] / df_year['network_purch_j']
        df_year['sales_share'] = df_year['sales_ij'] / df_year['network_sales_i']
        df_year['avg_mkt_share'] = df_year['mkt_share'] ** df_year['sales_share']

        agg = (
            df_year.groupby(['vat_i'], as_index=False)['avg_mkt_share']
            .prod()
            .rename(columns={'vat_i': 'vat'})
        )
        agg['year'] = year
        agg = agg[['vat', 'year', 'avg_mkt_share']]

        write_metric_year(agg, metric_path, year)
        print(f'Weighted average market share: Year {year}')
        del df_year, agg
        gc.collect()


def firm_nace_from_edges(input_base_path, start, end, metric_path):
    """
    Build a firm-year NACE table using both supplier (i) and customer (j) roles.
    """

    for year in range(start, end + 1):
        cols = ['vat_i', 'nace_i', 'vat_j', 'nace_j']
        df_year = read_year_partition(input_base_path, year, cols)

        if df_year.empty:
            write_metric_year(pd.DataFrame(columns=['vat', 'year', 'nace']), metric_path, year)
            continue

        # Supplier side
        nace_i = df_year[['vat_i', 'nace_i']].rename(columns={'vat_i': 'vat', 'nace_i': 'nace'}).dropna(subset=['vat'])

        # Customer side
        nace_j = df_year[['vat_j', 'nace_j']].rename(columns={'vat_j': 'vat', 'nace_j': 'nace'}).dropna(subset=['vat'])

        # Combine
        nace_all = pd.concat([nace_i, nace_j], ignore_index=True)

        # Deduplicate: firms might appear multiple times in edges
        firm_nace = (
            nace_all.groupby('vat')['nace']
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            .reset_index()
        )

        firm_nace['year'] = year
        write_metric_year(firm_nace, metric_path, year)
        del df_year, nace_i, nace_j, nace_all, firm_nace
        gc.collect()


def load_metric(metric_path):
    files = [
        os.path.join(metric_path, f)
        for f in os.listdir(metric_path)
        if f.endswith('.parquet') and f.startswith('year=')
    ]
    if not files:
        return pd.DataFrame()
    files.sort()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def create_panel(tmp_path, output_path):

    input_base_path = preprocess_partitioned_input(tmp_path)
    years = get_available_years(input_base_path)
    start = min(years)
    end = max(years)

    metric_root = os.path.join(tmp_path, 'metric_by_year')
    ensure_clean_dir(metric_root)

    upstreamness(input_base_path, start, end, os.path.join(metric_root, 'upstreamness'), window=1)
    upstreamness(input_base_path, start, end, os.path.join(metric_root, 'avg_upstreamness'), window=3)

    downstreamness(input_base_path, start, end, os.path.join(metric_root, 'downstreamness'), window=1)
    downstreamness(input_base_path, start, end, os.path.join(metric_root, 'avg_downstreamness'), window=3)

    centrality(input_base_path, start, end, os.path.join(metric_root, 'centrality'), alpha_const=True)
    centrality(input_base_path, start, end, os.path.join(metric_root, 'avg_centrality'), alpha_const=True, window=3)

    degrees(input_base_path, start, end, os.path.join(metric_root, 'degrees'))

    domar_weights(input_base_path, start, end, os.path.join(metric_root, 'domar'))

    net_sales_inputs(input_base_path, start, end, os.path.join(metric_root, 'net_sales_inputs'))

    wavg_mkt_share(input_base_path, start, end, os.path.join(metric_root, 'wavg_mkt_share'))

    firm_nace_from_edges(input_base_path, start, end, os.path.join(metric_root, 'firm_nace'))

    upstreamness_df = load_metric(os.path.join(metric_root, 'upstreamness'))
    avg_upstreamness_df = load_metric(os.path.join(metric_root, 'avg_upstreamness'))
    downstreamness_df = load_metric(os.path.join(metric_root, 'downstreamness'))
    avg_downstreamness_df = load_metric(os.path.join(metric_root, 'avg_downstreamness'))
    centrality_df = load_metric(os.path.join(metric_root, 'centrality'))
    avg_centrality_df = load_metric(os.path.join(metric_root, 'avg_centrality'))
    degrees_df = load_metric(os.path.join(metric_root, 'degrees'))
    domar_df = load_metric(os.path.join(metric_root, 'domar'))
    net_df = load_metric(os.path.join(metric_root, 'net_sales_inputs'))
    wavg_df = load_metric(os.path.join(metric_root, 'wavg_mkt_share'))
    nace_df = load_metric(os.path.join(metric_root, 'firm_nace'))

    panel_df = upstreamness_df.merge(downstreamness_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(avg_upstreamness_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(avg_downstreamness_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(centrality_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(avg_centrality_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(degrees_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(domar_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(net_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(wavg_df, on=['vat', 'year'], how='outer', validate='1:1')
    panel_df = panel_df.merge(nace_df, on=['vat', 'year'], how='left', validate='1:1')
    panel_df['nace2d'] = panel_df['nace'].str[:2]

    # save
    panel_df.to_parquet(os.path.join(output_path, 'panel.parquet'), engine='pyarrow', index=False)
    shutil.copy2(
        os.path.join(tmp_path, 'full_data_cleaned.parquet'),
        os.path.join(output_path, 'full_data_cleaned.parquet'),
    )
