import os
import pickle
import pandas as pd
import numpy as np

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
        'Mean': np.mean(degrees),
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
