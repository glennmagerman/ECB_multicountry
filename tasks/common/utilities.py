import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
from sklearn.neighbors import KernelDensity
import shutil
import yaml

################################
# Functions for task maintenance
################################

def initialize_task(abs_path):

    # Define the folders to manage
    folders = ['tmp', 'output', 'input']
    
    # Remove and recreate folders
    for folder in folders:
        folder_path = os.path.join(abs_path, folder)
        
        # Remove folder if it exists
        if os.path.exists(folder_path):
            # Apply input folder condition
            if folder == 'input' and os.path.basename(abs_path) == 'task1_clean_data': # do not remove data from task1 
                print(f"Skipped removing '{os.path.basename(abs_path)}'/'{folder}' as it's in 'task1_clean_data'.")
            else:
                shutil.rmtree(folder_path)
                print(f"Folder '{os.path.basename(abs_path)}'/'{folder}' removed successfully.")
        
        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{os.path.basename(abs_path)}'/'{folder}' created successfully.")
        else:
            print(f"Folder '{os.path.basename(abs_path)}'/'{folder}' already exists.")

def maintenance(abs_path):

    # Maintenance (remove temporary folder)
    if os.path.exists( os.path.join(abs_path,'tmp') ):
        shutil.rmtree( os.path.join(abs_path,'tmp') )


# Copy output from previous task
def copy_output_from_previous_task(abs_path, previous_task_name):

    dir1 = os.path.abspath(os.path.join(abs_path, '..', previous_task_name,'output'))
    target_dir = os.path.join(abs_path,'input') 

    for filename in os.listdir(dir1):
        src_path = os.path.join(dir1, filename)  # Source file path
        dest_path = os.path.join(target_dir, filename)  # Destination file path
        shutil.copy2(src_path, dest_path)  # Copy and replace file

# Create folders for storing output for each year
def create_folders_for_years(abs_path, output_path):
    
    start, end = extract_start_end_years(abs_path)

    for year in range(start, end + 1):
        
        # create the 'year' folder
        if not os.path.exists( os.path.join(output_path, f'{year}') ):
            os.makedirs( os.path.join(output_path, f'{year}') )
            
        # create folder for kernel densities
        if not os.path.exists( os.path.join(output_path, f'{year}', 'kernel_densities') ):
            os.makedirs( os.path.join(output_path, f'{year}', 'kernel_densities') )
            
        # create folder for moments tables
        if not os.path.exists( os.path.join(output_path, f'{year}', 'moments') ):
            os.makedirs( os.path.join(output_path, f'{year}', 'moments') )
        
        # create folder for correlations
        if not os.path.exists( os.path.join(output_path, f'{year}', 'correlations') ):
            os.makedirs( os.path.join(output_path, f'{year}', 'correlations') )
            
        # create folder for CCDFs
        if not os.path.exists( os.path.join(output_path, f'{year}', 'CCDF') ):
            os.makedirs( os.path.join(output_path, f'{year}', 'CCDF') )

################################
# Functions for loading data
################################

def save_workspace(data: dict, file_name: str):
    # to save session
    if os.path.exists(os.path.join(os.path.abspath(".."), file_name)):
        os.remove(os.path.join(os.path.abspath(".."), file_name))
    path_file = open(os.path.join(os.path.abspath(".."), file_name), 'wb')
    pickle.dump(data, path_file, protocol=pickle.HIGHEST_PROTOCOL)
    path_file.close()
    
def load_workspace(file_name: str):
    path_file = open(os.path.join(os.path.abspath(".."), file_name), 'rb')
    data = pickle.load(path_file)
    path_file.close()
    return data

def read_data(file_path):
    # Extract the file extension
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.csv':
        return pd.read_csv(file_path, dtype = {'nace': str})
    elif file_extension.lower() == '.dta':
        return pd.read_stata(file_path)
    else:
        raise ValueError("Unsupported file format")
    

def extract_start_end_years(abs_path):

    # Load configuration from YAML file
    path_config_file = os.path.abspath(os.path.join(abs_path, '..', 'config', 'config.yaml'))
    with open(path_config_file, 'r') as file:
        config = yaml.safe_load(file)

    # start and end year
    start = config['years']['start_year']
    end = config['years']['end_year']

    return start, end

def macros_from_config(abs_path):
    
    # Load configuration from YAML file
    path_config_file = os.path.abspath(os.path.join(abs_path, '..', 'config', 'config.yaml'))
    with open(path_config_file, 'r') as file:
        config = yaml.safe_load(file)

    # extract configuration parameters
    start, end = extract_start_end_years(abs_path)
    nfirms = config['data_generation']['nfirms']
    nlinks = config['data_generation']['nlinks']

    return start, end, nfirms, nlinks

################################
# Other functions
################################

def calculate_distribution_moments(degrees):
    """
    Calculates and returns the desired moments from a given distribution of degrees.
    """
    moments = {
        'N': len(degrees),
        'Mean': round(np.mean(degrees),2),
        'SD': round(np.std(degrees),2),
        'p1': round(np.percentile(degrees, 1),2),
        'p5': round(np.percentile(degrees, 5),2),
        'p10': round(np.percentile(degrees, 10),2),
        'p25': round(np.percentile(degrees, 25),2),
        'p50': round(np.percentile(degrees, 50),2),
        'p75': round(np.percentile(degrees, 75),2),
        'p90': round(np.percentile(degrees, 90),2),
        'p95': round(np.percentile(degrees, 95),2),
        'p99': round(np.percentile(degrees, 99),2)
    }
    
    return moments

# Define broad industries
def recategorize_industry(nace2):
    
    if nace2 <= 9:
        return 'Primary and extraction'
    elif 10 <= nace2 <= 33:
        return 'Manufacturing'
    elif 35 <= nace2 <= 39:
        return 'Utilities'
    elif 41 <= nace2 <= 43:
        return 'Construction'
    elif 45 <= nace2 <= 82:
        return 'Market services'
    elif nace2 >= 84 or np.isnan(nace2):
        return 'Non-market services'
    

def set_ticks_log_scale(x_value, step=1, axis='x'):

    log10_ticks = np.floor(np.log10(np.exp(x_value)))
    minx = min(log10_ticks)
    maxx = max(log10_ticks)
    maxx = maxx + 1 # set ceiling
    
    xticks_new = np.arange(minx, maxx + 1, step)
    
    if axis == 'x':
        return plt.xticks(xticks_new*2.3, [f'$10^{{{int(x)}}}$' for x in xticks_new])
    elif axis == 'y':
        return plt.yticks(xticks_new*2.3, [f'$10^{{{int(x)}}}$' for x in xticks_new])
    
    
def demean_variable_in_df(var_name, FE_name, df):
    
    # regress the variable on its NACE-4digits code fixed effect
    df = df.set_index([FE_name, 'year'])
    mod = PanelOLS.from_formula(f'{var_name} ~ 1 + EntityEffects', df)
    res = mod.fit()
    var_demeaned = np.array(res.resids)
    
    return var_demeaned

def aggregate_and_bin(df_year, x, y):

    df_year[f'{x}_bin'] = pd.qcut(df_year[x], q=20, duplicates='drop')
    binned_data = df_year.groupby(f'{x}_bin', observed=True).agg({
        y: 'mean',
        x: 'mean'
    }).reset_index()

    return binned_data

def find_kernel_densities(vec, bdwidth=1):
    # transpose the vector
    vec = vec.reshape(-1,1)
    
    # create densities
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bdwidth).fit(vec)
    grid = np.linspace(vec.min(), vec.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    
    return grid, kde_densities

def kernel_density_plot(array, xlabel: str, ylabel: str, name_plot: str, output_path, year):

    grid, kde_densities = find_kernel_densities(array)
    plt.plot(grid, kde_densities)
    set_ticks_log_scale(grid, step=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig( os.path.join(output_path, f'{year}', 'kernel_densities', name_plot), dpi=300, bbox_inches='tight' )
    plt.close()
