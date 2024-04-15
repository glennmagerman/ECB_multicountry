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

def load_large_dta( file_path , y):
    
    # Initialize an empty DataFrame to store rows for the year 2002
    df_year = pd.DataFrame()

    # Specify the chunk size
    chunk_size = 1000000  # Adjust based on your system's memory capacity

    # Read the Stata file in chunks
    for chunk in pd.read_stata(file_path, chunksize=chunk_size):
        # Filter the chunk for rows where the year equals 2002
        filtered_chunk = chunk[chunk['year'] == y]  # Assuming 'year' is the column name
    
        # Append the filtered chunk to the df_2002 DataFrame
        df_year = pd.concat([df_year, filtered_chunk], ignore_index=True)
        
    return df_year


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