import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_moments(cv_df, col='cv_pct'):
    s = cv_df[col].dropna()
    return pd.Series({
        'mean': s.mean(),
        'SD': s.std(ddof=1),
        'p10': s.quantile(0.10),
        'p25': s.quantile(0.25),
        'p50': s.quantile(0.50),
        'p75': s.quantile(0.75),
        'p90': s.quantile(0.90),
    })
    
def summarize_cv(df, group_cols, min_n=2, eps=1e-12):
    """
    Group by `group_cols`, compute n, mean, sd, and CV% = 100*sd/|mean|.
    Returns a DataFrame with those columns plus `group_cols`.
    """
    g = df.groupby(group_cols)['coef']
    out = pd.DataFrame({
        'n': g.size(),
        'mean_coef': g.mean(),
        'sd_coef': g.std(ddof=1)
    }).reset_index()

    out['cv_pct'] = np.where(
        (out['n'] >= min_n) & (out['mean_coef'].abs() > eps),
         out['sd_coef'] / out['mean_coef'].abs(),
        np.nan
    )
    return out

def plot_cdf(values, output_path, year, coefficient, label, country,xlabel="Coefficient of Variation", logx=False):
    s = np.sort(values[~np.isnan(values)])   # drop NaNs & sort
    y = np.linspace(0, 1, len(s))

    plt.figure(figsize=(10,6))
    plt.plot(s, y, lw=2)

    if logx:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{year}', 'coefficients_of_variation',f'CDF_CV_{coefficient}_shares_{label}_{country}.png'), dpi=300)
    plt.close()

def compute_cv(full_df, coefficient ,output_path, start, end, country):
    
    if coefficient == 'output':
        vars = ['network_sales_i', 'turnover_i']
    elif coefficient == 'input':
        vars = ['network_purch_j', 'inputs_j']
        
    full_df['nace_i_2d'] = full_df['nace_i'].str[:2]
    full_df['nace_j_2d'] = full_df['nace_j'].str[:2]
    
    for year in range(start, end+1):
        
        for var in vars:
            
            if var == 'network_purch_j':
                label = 'net_purch'
            else:
                label = 'tot_inputs'
                
            df_year = full_df[full_df['year'] == year].copy()
            df_year['coef'] =  df_year['sales_ij'] / df_year[var]
            
            summary_4d = summarize_cv(df_year, ['nace_i', 'nace_j'])
            summary_2d = summarize_cv(df_year, ['nace_i_2d', 'nace_j_2d'])
            plot_cdf(summary_4d['cv_pct'], output_path, year, coefficient, label, country)
            plot_cdf(summary_2d['cv_pct'], output_path, year, coefficient, label, country)
            
            row_4d = compute_moments(summary_4d)
            row_2d = compute_moments(summary_2d)
            cv_table = pd.DataFrame([row_4d, row_2d])
            cv_table.index = ['CV (4-digit pairs)', 'CV (2-digit pairs)']
            cv_table = cv_table[['mean','SD','p10','p25','p50','p75','p90']]  # order columns
            cv_table.to_csv(os.path.join(output_path, f'{year}', 'coefficients_of_variation',f'cv_{coefficient}_shares_{label}_{country}.csv'))
            
            moments_mean_4d = compute_moments(summary_4d, col='mean_coef')
            moments_mean_2d = compute_moments(summary_2d, col='mean_coef')
            mean_table = pd.DataFrame([moments_mean_4d, moments_mean_2d],
                          index=['Avg. share (4-digit pairs)', 'Avg. share (2-digit pairs)'])
            mean_table = mean_table[['mean','SD','p10','p25','p50','p75','p90']]
            mean_table.to_csv(os.path.join(output_path, f'{year}', 'coefficients_of_variation',f'avg_{coefficient}_shares_{label}_{country}.csv'))
            
def create_micro_IO(full_df, output_path, start, end, country):
    
    for year in range(start, end+1):
        
        df_year = full_df[full_df['year'] == year].copy()
        
        sales_j_4d = (
            df_year.groupby('nace_j')['turnover_j']
            .sum()
            .rename('sector_sales_j')
            .reset_index()
        )
        
        io_4d = (
            df_year
            .groupby(['nace_i', 'nace_j'], as_index=False)['sales_ij']
            .sum()
            .rename(columns={'sales_ij': 'value'})
        )
        io_4d = io_4d.merge(sales_j_4d, on='nace_j', how='left')
        io_4d['a_ij'] = io_4d['value'] / io_4d['sector_sales_j']
        
        io_4d_mat = io_4d.pivot(index='nace_i', columns='nace_j', values='a_ij') #.fillna(0.0)
        io_4d_mat.to_csv(os.path.join(output_path, str(year),f'io_nace4d_mat_{country}.csv'))
        
        sales_j_2d = (
            df_year.groupby('nace_j_2d')['turnover_j']
            .sum()
            .rename('sector_sales_j')
            .reset_index()
        )
        
        io_2d = (
            df_year
            .groupby(['nace_i_2d', 'nace_j_2d'], as_index=False)['sales_ij']
            .sum()
            .rename(columns={'sales_ij': 'value'})
        )
        io_2d = io_2d.merge(sales_j_2d, on='nace_j_2d', how='left')
        io_2d['a_ij'] = io_2d['value'] / io_2d['sector_sales_j']
        
        io_2d_mat = io_2d.pivot(index='nace_i_2d', columns='nace_j_2d', values='a_ij') #.fillna(0.0)
        io_2d_mat.to_csv(os.path.join(output_path, str(year),f'io_nace2d_mat_{country}.csv'))
        

# Master function
def master_cv(full_df, output_path, start, end, country):
    
    compute_cv(full_df, 'input', output_path, start, end, country)
    compute_cv(full_df, 'output', output_path, start, end, country)
    
    create_micro_IO(full_df, output_path, start, end, country)
