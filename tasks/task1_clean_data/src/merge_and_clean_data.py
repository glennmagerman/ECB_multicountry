import pandas as pd
import numpy as np
import os 

def upload_data(tmp_path):
    # Load the data
    #firms_df = pd.read_csv(os.path.join(tmp_path, 'firms_data_cleaned.csv'))
    #B2B_df = pd.read_csv(os.path.join(tmp_path, 'B2B_data_cleaned.csv'))
    firms_df = pd.read_parquet(os.path.join(tmp_path, 'firms_data_cleaned.parquet'), engine='pyarrow')
    B2B_df = pd.read_parquet(os.path.join(tmp_path, 'B2B_data_cleaned.parquet'), engine='pyarrow')

    return firms_df, B2B_df

# Merge B2B to firm dataset
def merge_data(firms_df, B2B_df):

    # merge network and firm data on vat_i
    full_df = B2B_df.merge( firms_df[['year','vat','nace','turnover','industry']].rename(
        columns={'vat': 'vat_i', 'turnover':'turnover_i', 'nace':'nace_i','industry':'industry_i'}), 
        on=['vat_i','year'], how='inner' )

    # merge network and firm data on vat_j
    full_df = full_df.merge( firms_df[['year','vat','nace','inputs_total']].rename(
        columns={'vat': 'vat_j', 'inputs_total':'inputs_j', 'nace': 'nace_j'}), on=['vat_j','year'], how='inner' )

    return full_df

# Clean data
def adjust_turnover(full_df):

    # 1. impute turnover from network sales
    # it may happen that some firms in the B2B do not report their turnover, 
    # so we need to impute it from the B2B sales
    # correct sales of i
    full_df['turnover_i'] = np.where( pd.isna(full_df['turnover_i']), full_df['sum_sales_ij'], full_df['turnover_i'] ).T

    return full_df

def adjust_sales_ij(full_df):
    # 2. correct sales from i to j larger than turnover of i
    full_df['sales_ij'] = np.where([full_df['sales_ij'] > 
                                        full_df['turnover_i']],full_df['turnover_i'],full_df['sales_ij']).T

    return full_df

def add_final_demand(full_df):
    # 3. add some final demand (1 euro) to the sales if bilateral sales between i and j = turnover of i
    full_df['turnover_i'] = np.where( [full_df['sales_ij'] == 
                                        full_df['turnover_i']], full_df['turnover_i'] + 1, full_df['turnover_i']  ).T

    # 4 add some final demand (1 euro) to the sales if network sales = turnover of i
    full_df['turnover_i'] = np.where( [full_df['sum_sales_ij'] == 
                                        full_df['turnover_i']], full_df['turnover_i'] + 1, full_df['turnover_i']  ).T
    
    return full_df

def adjust_inputs(full_df):

    # impute total inputs from network purchases
    # it may happen that some firms do not report input use but there are reported purchases in the B2B network
    full_df['sum_purch_ij'] = full_df.groupby('vat_j')['sales_ij'].transform('sum')
    full_df['inputs_j'] = np.where( pd.isna(full_df['inputs_j']), full_df['sum_purch_ij'], full_df['inputs_j'] ).T
    full_df['inputs_j'] = np.where(full_df['inputs_j']==0, full_df['sum_purch_ij'], full_df['inputs_j'] ).T
    full_df.drop(columns=['sum_purch_ij'], inplace=True)

    return full_df

def clean_merged_data(full_df):

    # calculate total network sales for each i
    full_df['sum_sales_ij'] = full_df.groupby('vat_i')['sales_ij'].transform('sum')

    # 1. impute turnover from network sales
    full_df = adjust_turnover(full_df)

    # 2. correct sales from i to j larger than turnover of i
    full_df = adjust_sales_ij(full_df)

    # 3. add some final demand (1 euro) to the sales if bilateral sales between i and j = turnover of i
    full_df = add_final_demand(full_df)

    # drop the column for total network sales
    full_df.drop(columns=['sum_sales_ij'], inplace=True)

    # 4. adjust inputs from network purchases
    full_df = adjust_inputs(full_df)

    return full_df

def save_data(full_df, output_path):

    # Save the cleaned data
    #full_df.to_csv(os.path.join(output_path, 'full_data_cleaned.csv'), index=False)
    full_df.to_parquet(os.path.join(output_path, 'full_data_cleaned.parquet'), engine='pyarrow')

    return None

# Master function
def merge_and_clean_data(tmp_path, output_path):

    firm_df, B2B_df = upload_data(tmp_path)

    full_df = merge_data(firm_df, B2B_df)

    full_df = clean_merged_data(full_df)

    save_data(full_df, output_path)
