import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
from sklearn.neighbors import KernelDensity

"""Functions"""

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
        return pd.read_csv(file_path)
    elif file_extension.lower() == '.dta':
        return pd.read_stata(file_path)
    else:
        raise ValueError("Unsupported file format")


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

def kernel_density_plot(vec, bdwidth=1):
    # transpose the vector
    vec = vec.reshape(-1,1)
    
    # create densities
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bdwidth).fit(vec)
    grid = np.linspace(vec.min(), vec.max(), 1000).reshape(-1, 1)
    kde_scores = kde.score_samples( grid )
    kde_densities = np.exp(kde_scores)
    
    return grid, kde_densities