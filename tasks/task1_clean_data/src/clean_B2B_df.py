import pandas as pd
import os 

# Clean data
def remove_zeros(B2B_df):
    B2B_df = B2B_df[B2B_df['sales_ij'] != 0]
    return B2B_df

# save data
def save_data(B2B_df, tmp_path):

    # Save the cleaned data
    B2B_df.to_parquet(os.path.join(tmp_path, 'B2B_data_cleaned.parquet'), engine='pyarrow')

    return None

# Master function
def clean_B2B_df(tmp_path, B2B_df):
    
        B2B_df = remove_zeros(B2B_df)
    
        save_data(B2B_df, tmp_path)

