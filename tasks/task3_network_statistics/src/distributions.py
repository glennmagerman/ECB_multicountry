import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
from common.utilities import set_ticks_log_scale, demean_variable_in_df, calculate_distribution_moments, kernel_density_plot, find_kernel_densities, recategorize_industry

    
def calculate_distributions_per_year(panel_df, output_path, start, end, vars_to_loop, label_map, country):
    
    for var in vars_to_loop:
        
        xlabel = label_map.get(var, var)
        
        for year in range(start, end+1):

            df_year = panel_df[panel_df['year'] == year]
            
            var_np = np.array(df_year[var].dropna())
            var_np_pos = var_np[var_np>0]
            log_var = np.log(var_np_pos)
            
            kernel_density_plot(log_var, xlabel, 'Density', f'{var}_{country}.png', output_path, year)
            
            # Now de-mean variables by nace (both 4 digits and 2 digits)
            df_demean = df_year.copy()
            df_demean[f'ln_{var}'] = np.log(df_demean[var].where(df_demean[var] > 0)).dropna()
            
            df_demean = df_demean.groupby('nace').filter(lambda x: len(x) >= 5).dropna(subset=[f'ln_{var}', 'nace'])    # Filter out sectors with less than 5 firms
            demeaned_var = demean_variable_in_df(f'ln_{var}', 'nace', df_demean)
            
            kernel_density_plot(demeaned_var, f'{xlabel}, demeaned', 'Density', f'{var}_demeaned_nace4d_{country}.png', output_path, year)
            
            df_demean = df_demean.groupby('nace2d').filter(lambda x: len(x) >= 5).dropna(subset=[f'ln_{var}', 'nace2d'])    # Filter out sectors with less than 5 firms
            demeaned_var = demean_variable_in_df(f'ln_{var}', 'nace2d', df_demean)
            
            kernel_density_plot(demeaned_var, f'{xlabel}, demeaned', 'Density', f'{var}_demeaned_nace2d_{country}.png', output_path, year)
            
def calculate_distributions_per_year_by_ind(panel_df, output_path, start, end, vars_to_loop, label_map, country, demean=None):
    
    industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction', 
                        'Market services','Non-market services']
    
    for var in vars_to_loop:
        
        xlabel = label_map.get(var, var)
        
        for year in range(start, end+1):

            x_min = np.inf
            x_max = -np.inf
            df_year = panel_df[panel_df['year'] == year]

            for industry in industries:
                
                df_year_sec = df_year[df_year['industry'] == industry].copy()
                
                if demean is not None:
                    df_demean = df_year_sec.copy()
                    df_demean[f'ln_{var}'] = np.log(df_demean[var].where(df_demean[var] > 0)).dropna()
                    df_demean = df_demean.groupby(demean).filter(lambda x: len(x) >= 5).dropna(subset=[f'ln_{var}', demean])  # Filter out sectors with less than 5 firms
                    log_var = demean_variable_in_df(f'ln_{var}', demean, df_demean)
                else:
                    var_np = np.array(df_year_sec[var].dropna())
                    var_np_pos = var_np[var_np>0]
                    log_var = np.log(var_np_pos)
                
                grid, kde_densities = find_kernel_densities(log_var)
                plt.plot(grid, kde_densities, label=f'{industry}')
                x_min = min(min(grid), x_min)
                x_max = max(max(grid), x_max)
                
            set_ticks_log_scale([x_min, x_max], step=2)
            plt.ylabel('Density')
            plt.legend()
            if demean is not None:
                plt.xlabel(f'{xlabel}, demeaned')
                plt.savefig(os.path.join(output_path,f'{year}', 'kernel_densities', f'{var}_bysec_demeaned_{demean}_{country}.png'), dpi=300, bbox_inches='tight')
            else:
                plt.xlabel(xlabel)
                plt.savefig(os.path.join(output_path,f'{year}', 'kernel_densities', f'{var}_bysec_{country}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def summary_tables(full_df, output_path, start, end, country, var):
    
    industries = ['Primary and extraction','Manufacturing', 'Utilities', 'Construction',
                    'Market services','Non-market services']
    
    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year]
        
        sum_bysec = []
            
        # Calculate degree moments for each industry
        for industry in industries:
            df_year_sec = df_year[df_year['industry'] == industry].copy()
            var_np_sec = np.array(df_year_sec[var]) 
            moments_sec = calculate_distribution_moments(var_np_sec)
            sum_bysec.append(moments_sec)
        
        # Calculate degree moments for the full graph
        var_np = np.array(df_year[var])
        moments_full = calculate_distribution_moments(var_np)
        sum_bysec.append(moments_full)

        # Create a summary table and save it to CSV
        sum_bysec_tab = pd.DataFrame(sum_bysec, index=industries + ['All'])
        sum_bysec_tab.to_csv(os.path.join(output_path, f'{year}', 'moments', f'{var}_bysec_{country}.csv')) 
        
def vars_correlation_summary(panel_df, output_path, start, end, vars_to_loop, label_map, country):
    
    labels = [label_map.get(v, v) for v in vars_to_loop]
    
    nace = ['nace', 'nace2d']
    vars_in_logs = []
    vars_in_logs_dem_nace = []
    vars_in_logs_dem_nace2d = []
    for var in vars_to_loop:
        panel_df[f'ln_{var}'] = np.log(panel_df[var].where(panel_df[var] > 0))
        vars_in_logs.append(f'ln_{var}')
        for nace_type in nace:
            panel_df = panel_df.groupby(nace_type).filter(lambda x: len(x) >= 5).dropna(subset=[f'ln_{var}', nace_type])  # Filter out sectors with less than 5 firms
            panel_df[f'ln_{var}_dem_{nace_type}'] = demean_variable_in_df(f'ln_{var}', nace_type, panel_df)

    vars_in_logs_dem_nace = [f'ln_{v}_dem_nace' for v in vars_to_loop]
    vars_in_logs_dem_nace2d = [f'ln_{v}_dem_nace2d' for v in vars_to_loop]
    
    vars_list = [vars_in_logs, vars_in_logs_dem_nace, vars_in_logs_dem_nace2d]
    
    for year in range(start, end+1):
        
        for var_list in vars_list:
            df_year = panel_df[panel_df['year'] == year]
            corr_df = df_year[var_list].corr()
            
            corr_df.to_csv(os.path.join(output_path, f'{year}', 'correlations', f'correlation_matrix_{country}.csv'))
            corr_df.to_latex(os.path.join(output_path, f'{year}', 'correlations', f'correlation_matrix_{country}.tex'), float_format="%.3f")
            
            fig, ax = plt.subplots(figsize=(14, 12))

            cax = ax.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)

            # ticks
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)

            # loop to annotate each cell
            for i in range(len(vars_to_loop)):
                for j in range(len(vars_to_loop)):
                    ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                            ha="center", va="center",
                            color="black" if abs(corr_df.iloc[i,j]) < 0.5 else "white",
                            fontsize=9)

            # colorbar
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            if var_list[0].endswith('nace2d'):
                nace_cat = '_nace2d'
            elif var_list[0].endswith('nace'):
                nace_cat = '_nace'
            else:
                nace_cat = ''
                
            plt.savefig(os.path.join(output_path, f'{year}', 'correlations',f'correlation_heatmap_{country}{nace_cat}.png'), dpi=300)
            plt.close()

def master_distributions(panel_df, output_path, start, end, country):
    
    panel_df['avg_network_sales'] = (
        (panel_df['network_sales'] / panel_df['outdeg'])
        .where(panel_df['outdeg'] != 0)
    )

    panel_df['avg_network_purch'] = (
        (panel_df['network_purch'] / panel_df['indeg'])
        .where(panel_df['indeg'] != 0)
    )
    
    panel_df['avg_turnover'] = (
        (panel_df['turnover'] / panel_df['outdeg'])
        .where(panel_df['outdeg'] != 0)
    )

    panel_df['avg_inputs'] = (
        (panel_df['inputs'] / panel_df['indeg'])
        .where(panel_df['indeg'] != 0)
    )
    
    vars_to_loop = [
        'turnover',
        'inputs',
        'network_sales',
        'network_purch',
        'outdeg',
        'indeg',
        'avg_network_sales',
        'avg_network_purch',
        'avg_turnover',
        'avg_inputs',
        'avg_mkt_share',
        'domar',
        'upstreamness',
        'downstreamness',
        'centrality'
    ]
    
    label_map = {
        'turnover': 'Total sales',
        'inputs': 'Total inputs',
        'network_sales': 'Network sales',
        'network_purch': 'Network purchases',
        'outdeg': 'Number of customers',
        'indeg': 'Number of suppliers',
        'avg_network_sales': 'Network sales per customer',
        'avg_network_purch': 'Network purchases per supplier',
        'avg_turnover': 'Total sales per customer',
        'avg_inputs': 'Total purchases per supplier',
        'avg_mkt_share': 'Average market share',
        'domar': 'Domar weight',
        'upstreamness': 'Upstreamness',
        'downstreamness': 'Downstreamness',
        'centrality': 'Bonacich centrality'
    }
    
    panel_df['industry'] = panel_df['nace2d'].apply(
        lambda x: recategorize_industry(int(x)) if pd.notna(x) else np.nan
    )
    
    # Distributions by year
    calculate_distributions_per_year(panel_df, output_path, start, end, vars_to_loop, label_map, country)
    
    # Distributions by year and industry
    calculate_distributions_per_year_by_ind(panel_df, output_path, start, end, vars_to_loop, label_map, country)
    calculate_distributions_per_year_by_ind(panel_df, output_path, start, end, vars_to_loop, label_map, country, demean='nace')
    calculate_distributions_per_year_by_ind(panel_df, output_path, start, end, vars_to_loop, label_map, country, demean='nace2d')
    
    # Summary tables of outdegree and indegree
    summary_tables(panel_df, output_path, start, end, country, var='outdeg')
    summary_tables(panel_df, output_path, start, end, country, var='indeg')
    
    # Correlation tables
    vars_correlation_summary(panel_df, output_path, start, end, vars_to_loop, label_map, country)
    
    
    
    