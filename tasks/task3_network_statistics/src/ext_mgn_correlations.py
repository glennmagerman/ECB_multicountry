import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os 
import re
from common.utilities import demean_variable_in_df, set_ticks_log_scale, aggregate_and_bin

def plot_binned_data(binned_data, res, x, y):
    
    # Plotting binned data
    plt.figure(figsize=(6, 4))
    plt.scatter(binned_data[x], binned_data[y], color='navy')
    textstr = '\n'.join((
        f'Linear slope: {res.coef()[x]:.2f} ({res.se()[x]:.2f})',
        f'R-squared: {res._r2:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=8,
                   verticalalignment='top', bbox=props)
    set_ticks_log_scale(np.array(binned_data[x]), step=1)
    set_ticks_log_scale(np.array(binned_data[y]), step=1, axis='y')

# 3. Compute correlations between outdegree and indegree

def corr_outdeg_indeg(full_df, output_path, start, end, country):

    for year in range(start, end+1):
        df_year = full_df[full_df['year'] == year].copy()

        # transform variables in logs
        df_year['log_indeg'] = np.log(df_year['indeg'].where(df_year['indeg'] > 0))
        df_year['log_outdeg'] = np.log(df_year['outdeg'].where(df_year['outdeg'] > 0))
        
        # de-mean the variables
        df_year = df_year.dropna(subset=['log_indeg']) # drop NaNs
        df_year = df_year.groupby('nace').filter(lambda x: len(x) >= 5) # drop NACE codes with less than 5 observations
        df_year['log_indeg_dem'] = demean_variable_in_df('log_indeg', 'nace', df_year)
        df_year = df_year.dropna(subset=['log_outdeg'])
        df_year['log_outdeg_dem'] = demean_variable_in_df('log_outdeg', 'nace', df_year)
        
        # regress the variables
        res = pf.feols('log_indeg_dem ~ log_outdeg_dem | nace', data=df_year)
        
        # bin the data
        binned_data = aggregate_and_bin(df_year, 'log_outdeg_dem', 'log_indeg_dem')
        
        # Now plot the binned data
        plot_binned_data(binned_data, res, 'log_outdeg_dem', 'log_indeg_dem')
        plt.xlabel('Number of customers, demeaned')
        plt.ylabel('Number of suppliers, demeaned')
        fig_path = os.path.join(
                        output_path, f"{year}", "correlations", 'binned_scatters'
                    )
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig( os.path.join(output_path,f'{year}', 'correlations', 'binned_scatters',f'indeg_outdeg_{country}.png'), dpi=300, bbox_inches='tight' )
        plt.close()
        
def all_vars_reg(panel_df, output_path, start, end, country):
    
    vars = ['outdeg', 'indeg', 'avg_network_sales', 'avg_network_purch', 'avg_mkt_share', 'domar', 'upstreamness', 'downstreamness']
    fe_parts = {
        'none'  : '',          # no FE
        'nace2' : 'nace2d',    # NACE2 FE
        'nace4' : 'nace',      # NACE4 FE
    }

    y_vars = ['ln_net_sales', 'ln_turnover', 'ln_net_purch', 'ln_inputs']
    results = {}   # results[year][ln_var][spec_name] = feols result

    for year in range(start, end + 1):
        df_year = panel_df[panel_df['year'] == year].copy()

        # logs for this year (safe)
        df_year['ln_net_sales'] = np.log(df_year['network_sales'].where(df_year['network_sales'] > 0))
        df_year['ln_turnover']  = np.log(df_year['turnover'].where(df_year['turnover'] > 0))
        df_year['ln_net_purch'] = np.log(df_year['network_purch'].where(df_year['network_purch'] > 0))
        df_year['ln_inputs']  = np.log(df_year['inputs'].where(df_year['inputs'] > 0))

        results.setdefault(year, {})

        for yvar in y_vars:
            for var in vars:
                ln_var = f'ln_{var}'
                df_year[ln_var] = np.log(df_year[var].where(df_year[var] > 0))
                results[year].setdefault(ln_var, {})

                # drop rows with NaNs in either y or x
                df_use = df_year.dropna(subset=[yvar, ln_var]).copy()
                if df_use.empty:
                    continue
                
                models = []
                for spec_name, fe_str in fe_parts.items():
                    # build pyfixest formula
                    if fe_str == '':
                        fml = f'{yvar} ~ {ln_var}'              # no FE
                    else:
                        fml = f'{yvar} ~ {ln_var} | {fe_str}'  # with FE

                    # pyfixest estimation
                    res = pf.feols(fml, df_use)
                    results[year][ln_var][spec_name] = res
                    models.append(res)
                    
                # ---- save LaTeX table ----
                save_dir = os.path.join(output_path, str(year), "correlations", "tables")
                os.makedirs(save_dir, exist_ok=True)

                tex_path = os.path.join(
                    save_dir,
                    f"reg_{yvar}_{ln_var}_{country}.tex"
                )
                with open(tex_path, "w") as f:
                        f.write(pf.etable(models, type='tex'))  
                        
def binned_plots_demeaned_vars(panel_df, output_path, country, start, end):
    
    vars = ['network_sales', 'avg_network_sales', 'avg_mkt_share']
    nace_type = ['nace', 'nace2d']
    
    label_map = {
        'ln_network_sales_dem': 'Network Sales (demeaned)',
        'ln_outdeg_dem': 'Number of customers (demeaned)',
        'ln_avg_network_sales_dem': 'Network sales per customer (demeaned)',
        'ln_avg_mkt_share_dem': 'Weighted avg. market share (demeaned)'
    }
    
    for year in range(start, end+1):
        
        df_year = panel_df[panel_df['year'] == year].copy()
        
        for nace in nace_type:
            
            df_year['ln_outdeg'] = np.log(df_year['outdeg'].where(df_year['outdeg'] > 0))
            df_demean = df_year.groupby(nace).filter(lambda x: len(x) >= 5).dropna(subset=['ln_outdeg', nace])    # Filter out sectors with less than 5 firms
            df_demean['ln_outdeg_dem'] = demean_variable_in_df('ln_outdeg', nace, df_demean)
            
            for yvar in vars:
                
                df_demean[f'ln_{yvar}'] = np.log(df_demean[yvar].where(df_demean[yvar] > 0))
                df_demean = df_demean.groupby(nace).filter(lambda x: len(x) >= 5).dropna(subset=[yvar, nace])    # Filter out sectors with less than 5 firms
                df_demean[f'ln_{yvar}_dem'] = demean_variable_in_df(f'ln_{yvar}', nace, df_demean)
                
                # estimation
                df_use = df_demean.dropna(subset=[f'ln_{yvar}_dem', 'ln_outdeg_dem']).copy()
                if df_use.empty:
                    continue
                res = pf.feols(f'ln_{yvar}_dem ~ ln_outdeg_dem', df_use)
                
                # ---- binned scatter plot ----
                binned_data = aggregate_and_bin(df_use, 'ln_outdeg_dem', f'ln_{yvar}_dem')
                plot_binned_data(binned_data, res, 'ln_outdeg_dem',  f'ln_{yvar}_dem')

                ylabel = label_map.get( f'ln_{yvar}_dem',  f'ln_{yvar}_dem')
                xlabel = label_map.get('ln_outdeg_dem', 'ln_outdeg_dem')
                plt.ylabel(ylabel)
                plt.xlabel(xlabel)

                fig_path = os.path.join(
                        output_path, f"{year}", "correlations", 'binned_scatters'
                    )
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)
                plt.savefig(os.path.join(fig_path, f"binned_ln_{yvar}_outdeg_dem{nace}_{country}.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
                        
def gdp_reg(panel_df, gdp_df, gdp_pc_df, output_path, country):
    
    vars = ['outdeg', 'indeg', 'avg_network_sales', 'avg_network_purch', 'avg_mkt_share', 'domar', 'upstreamness', 'downstreamness']
    for var in vars:
        panel_df[f'ln_{var}'] = np.log(panel_df[var].where(panel_df[var] > 0))

        
    agg_means = panel_df.groupby('year')[[f'ln_{v}' for v in vars]].mean().add_suffix('_mean')
    agg_stds  = panel_df.groupby('year')[[f'ln_{v}' for v in vars]].std().add_suffix('_std')
    if not os.path.exists(os.path.join(output_path, 'all_years')):
        os.mkdir(os.path.join(output_path, 'all_years'))
    agg_means.to_csv(os.path.join(output_path, 'all_years',f'vars_for_gdp_df_means_{country}.csv'))
    agg_stds.to_csv(os.path.join(output_path, 'all_years', f'vars_for_gdp_df_stds_{country}.csv'))
    
    year_cols = [col for col in gdp_df.columns if col.isdigit()]
    year_cols_int = list(map(int, year_cols))
    min_year = panel_df["year"].min()
    max_year = panel_df["year"].max()
    selected_years = [str(y) for y in year_cols_int if min_year <= y <= max_year]
    gdp_country = gdp_df[gdp_df['Country Code'] == country]
    gdp = np.log(gdp_country[selected_years]).T
    gdp.columns = ["gdp"]
    gdp.index = gdp.index.astype(int)
    
    gdp_pc_country = gdp_pc_df[gdp_pc_df['Country Code'] == country]
    gdp_pc = np.log(gdp_pc_country[selected_years]).T
    gdp_pc.columns = ["gdp_pc"]
    gdp_pc.index = gdp_pc.index.astype(int)
    
    stats_df = pd.concat([gdp, gdp_pc, agg_means, agg_stds], axis=1)
    
    label_map = {
        'ln_net_sales': 'Network Sales',
        'ln_turnover': 'Total sales',
        'ln_outdeg': 'Number of customers',
        'ln_indeg': 'Number of suppliers',
        'ln_avg_network_sales': 'Network sales per customer',
        'ln_avg_network_purch': 'Network purchases per supplier',
        'ln_avg_mkt_share': 'Average Market Share',
        'ln_domar': 'Domar Weight',
        'ln_upstreamness': 'Upstreamness',
        'ln_downstreamness': 'Downstreamness'
    }

    extended_label_map = {}
    for key, base_label in label_map.items():
        extended_label_map[f"{key}_mean"] = f"{base_label} (mean, in logs)"
        extended_label_map[f"{key}_std"]  = f"{base_label} (std, in logs)"

    # --- run the same loop for GDP and GDP per capita ---
    for x_col, x_label, prefix in [
        ("gdp", "GDP", "scatter_gdp"),
        ("gdp_pc", "GDP per capita", "scatter_gdp_pc")
    ]:
        for y_col in stats_df.columns:
            # skip both GDP and GDP per capita on the y-axis
            if y_col in ["gdp", "gdp_pc"]:
                continue

            # Drop missing values for this pair
            df_plot = stats_df[[x_col, y_col]].dropna()
            if df_plot.empty:
                continue

            x = df_plot[x_col]
            y = df_plot[y_col]

            # --- OLS regression y = a + b x ---
            X = sm.add_constant(x.values)      # use values for regression
            model = sm.OLS(y.values, X).fit()

            beta = model.params[1]             # slope
            se   = model.bse[1]                # std. error of slope
            r2   = model.rsquared

            # Fitted line for plotting
            x_line = np.linspace(x.min(), x.max(), 100)
            X_line = sm.add_constant(x_line)
            y_line = model.predict(X_line)

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(10, 4))

            # scatter
            ax.scatter(x, y, s=40, alpha=0.7, color='navy')

            # label each point with the year (index of df_plot)
            for year, x_val, y_val in zip(df_plot.index, x, y):
                ax.annotate(
                    str(year),
                    (x_val, y_val),
                    xytext=(3, 3),              # small offset so text not exactly on the dot
                    textcoords="offset points",
                    fontsize=8
                )

            # regression line
            ax.plot(x_line, y_line, linewidth=1, color='black', linestyle='--')

            # labels and title
            ax.set_xlabel(x_label)
            ylabel = extended_label_map.get(y_col, y_col)
            ax.set_ylabel(ylabel)

            # stats box in top-left corner
            textstr = (
                rf"Coef. = ${beta:.3f} ({se:.3f})$" + "\n"
                rf"$R^2 = {r2:.3f}$"
            )

            ax.text(
                0.05, 0.95, textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            # grid (optional)
            ax.grid(alpha=0.3)
            ax.grid(alpha=0.3)

            # save
            fname = f"{prefix}_{y_col}_{country}.png"
            fig.savefig(os.path.join(output_path,'all_years',fname), dpi=300, bbox_inches='tight')
            plt.close(fig)

def clean_gdp(gdp_df_full):
    
    gdp_df = gdp_df_full[gdp_df_full["Series Name"] == "GDP (current US$)"].reset_index(drop=True)
    gdp_pc_df = gdp_df_full[gdp_df_full["Series Name"] == "GDP per capita (current US$)"].reset_index(drop=True)
    
    cols_to_drop = ["Country Name", "Series Name", "Series Code"]
    gdp_df = gdp_df.drop(columns=cols_to_drop)
    gdp_pc_df = gdp_pc_df.drop(columns=cols_to_drop)

    # Clean year column names: keep only the year
    def clean_columns(df):
        new_cols = {}
        for col in df.columns:
            match = re.match(r"(\d{4})", col)
            if match:
                new_cols[col] = match.group(1) 
        df = df.rename(columns=new_cols)
        return df

    gdp_df = clean_columns(gdp_df)
    gdp_pc_df = clean_columns(gdp_pc_df)

    # Reset index
    gdp_df = gdp_df.reset_index(drop=True)
    gdp_pc_df = gdp_pc_df.reset_index(drop=True)
    
    return gdp_df, gdp_pc_df

# Master function
def master_ext_mgn_correlations(panel_df, gdp_df_full, output_path, start, end, country):
    
    panel_df['avg_network_sales'] = (
        (panel_df['network_sales'] / panel_df['outdeg'])
        .where(panel_df['outdeg'] != 0)
    )

    panel_df['avg_network_purch'] = (
        (panel_df['network_purch'] / panel_df['indeg'])
        .where(panel_df['indeg'] != 0)
    )
    panel_df['ln_net_sales'] = np.log(panel_df['network_sales'])
    
    all_vars_reg(panel_df, output_path, start, end, country)
    
    binned_plots_demeaned_vars(panel_df, output_path, country, start, end)
    
    # Compute correlation between outdegree and indegree
    corr_outdeg_indeg(panel_df, output_path, start, end, country)
    
    # correlations with GDP and GDP_pc
    gdp_df, gdp_pc_df = clean_gdp(gdp_df_full)
    gdp_reg(panel_df, gdp_df, gdp_pc_df, output_path, country)

    
        
    

