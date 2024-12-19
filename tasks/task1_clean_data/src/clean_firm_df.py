import pandas as pd
import os 
from common.utilities import recategorize_industry


# Adjust NACE sectors
def extract_nace_4_digits(firms_df):

    firms_df['nace'] = firms_df['nace'].astype(str)
    
    # Determine the maximum length of NACE codes
    max_length = firms_df['nace'].str.len().max()
    # add a leading 0 to NACE codes with less than 4 digits (csv file reads them as integers)
    firms_df['nace'] = firms_df['nace'].apply(lambda x: x.zfill(max_length))

    #extract NACE 4-digits codes, replace NACE 5 (or more) digits
    firms_df['nace'] = firms_df['nace'].astype(str).str.slice(start=0, stop=4)

    return firms_df

def define_industry(firms_df):

    # extract 2-digits NACE code
    firms_df['nace_2digits'] = firms_df['nace'].astype(str).str.slice(start=0, stop=2)
    firms_df['nace_2digits'] = pd.to_numeric(firms_df['nace_2digits'], errors='coerce')

    # include broad definition of industries
    firms_df['industry'] = firms_df['nace_2digits'].apply(lambda x: recategorize_industry(x) )
    firms_df.drop(columns=['nace_2digits'], inplace=True)
    firms_df['industry'] = firms_df['industry'].fillna('Unknown')

    return firms_df

# Clean data
def remove_zeros(firms_df):
    firms_df = firms_df[firms_df['turnover'] != 0]
    return firms_df

# save data
def save_data(firms_df, tmp_path):

    # Save the cleaned data
    firms_df.to_parquet(os.path.join(tmp_path, 'firms_data_cleaned.parquet'), engine='pyarrow')

    return None

# Master function
def clean_firm_df(tmp_path, firms_df):

    firms_df = extract_nace_4_digits(firms_df)
    firms_df = define_industry(firms_df)
    firms_df = remove_zeros(firms_df)

    save_data(firms_df, tmp_path)

    return firms_df