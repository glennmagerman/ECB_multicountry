import pandas as pd
import numpy as np
import os 

def upload_data(tmp_path):
    # Load the data
    firms_df = pd.read_parquet(os.path.join(tmp_path, 'firms_data_cleaned.parquet'), engine='pyarrow')
    B2B_df = pd.read_parquet(os.path.join(tmp_path, 'B2B_data_cleaned.parquet'), engine='pyarrow')
    
    return firms_df, B2B_df

# Merge B2B to firm dataset
def merge_data(firms_df, B2B_df):

    # merge network and firm data on vat_i
    full_df = B2B_df.merge( firms_df[['year','vat','nace','turnover', 'inputs_total','industry']].rename(
        columns={'vat': 'vat_i', 'turnover':'turnover_i', 'inputs_total': 'inputs_i', 'nace':'nace_i','industry':'industry_i'}), 
        on=['vat_i','year'], how='left' )

    # merge network and firm data on vat_j
    full_df = full_df.merge( firms_df[['year','vat','nace','inputs_total', 'turnover', 'industry']].rename(
        columns={'vat': 'vat_j', 'inputs_total':'inputs_j', 'nace': 'nace_j', 'turnover': 'turnover_j', 'industry':'industry_j'}), on=['vat_j','year'], how='left' )

    return full_df

# Clean data
def compute_network_agg(full_df):
    
    network_sales = full_df.groupby(['vat_i', 'year'])['sales_ij'].sum().reset_index(name='network_sales').rename(columns={'vat_i': 'vat'})
    network_purch = full_df.groupby(['vat_j', 'year'])['sales_ij'].sum().reset_index(name='network_purch').rename(columns={'vat_j': 'vat'})
    network_agg = pd.merge(network_sales, network_purch, on=['vat', 'year'], how='outer')
    
    # merge back with the full data
    full_df = pd.merge(full_df, network_agg.rename(columns={'vat': 'vat_i', 'network_sales': 'network_sales_i', 'network_purch': 'network_purch_i'}), on=['vat_i', 'year'], how='left')
    full_df = pd.merge(full_df, network_agg.rename(columns={'vat': 'vat_j', 'network_sales': 'network_sales_j', 'network_purch': 'network_purch_j'}), on=['vat_j', 'year'], how='left')
    
    return full_df

def adjust_turnover(full_df):

    # impute turnover from network sales
    # it may happen that some firms in the B2B do not report their turnover, 
    # so we need to impute it from the B2B sales
    # correct sales of i
    full_df['turnover_i'] = np.where( full_df['turnover_i'].isna(), full_df['network_sales_i'], full_df['turnover_i'] )#.T
    
    # correct sales of i if network sales are larger
    full_df['turnover_i'] = np.where(full_df['network_sales_i'] > 
                                        full_df['turnover_i'],full_df['network_sales_i'],full_df['turnover_i'])#.T
    
    # correct sales of j
    full_df['turnover_j'] = np.where( full_df['turnover_j'].isna(), full_df['network_sales_j'], full_df['turnover_j'] )#.T
    
    # correct sales of j if network sales are larger
    full_df['turnover_j'] = np.where(full_df['network_sales_j'] > 
                                        full_df['turnover_j'],full_df['network_sales_j'],full_df['turnover_j'])#.T

    return full_df

def adjust_inputs(full_df):

    # impute total inputs from network purchases
    # it may happen that some firms do not report input use but there are reported purchases in the B2B network
    full_df['inputs_j'] = np.where( full_df['inputs_j'].isna(), full_df['network_purch_j'], full_df['inputs_j'] ).T
    full_df['inputs_j'] = np.where(full_df['inputs_j']<=0, full_df['network_purch_j'], full_df['inputs_j'] ).T # correct negative purchases
    
    # same for firm i
    full_df['inputs_i'] = np.where( full_df['inputs_i'].isna(), full_df['network_purch_i'], full_df['inputs_i'] ).T
    full_df['inputs_i'] = np.where(full_df['inputs_i']<=0, full_df['network_purch_i'], full_df['inputs_i'] ).T
    
    # correct purchases of j if network purchases are larger
    full_df['inputs_j'] = np.where([full_df['network_purch_j'] > 
                                        full_df['inputs_j']],full_df['network_purch_j'],full_df['inputs_j']).T
    
    # same for firm i
    full_df['inputs_i'] = np.where([full_df['network_purch_i'] > 
                                        full_df['inputs_i']],full_df['network_purch_i'],full_df['inputs_i']).T

    return full_df
    

def clean_merged_data(full_df):
    
    # compute network sales and purchases
    full_df = compute_network_agg(full_df)

    # impute turnover from network sales
    full_df = adjust_turnover(full_df)

    # adjust inputs from network purchases
    full_df = adjust_inputs(full_df)
    
    full_df['sales_to_fd_i'] = full_df['turnover_i'] - full_df['network_sales_i'].fillna(0)
    full_df['sales_to_fd_j'] = full_df['turnover_j'] - full_df['network_sales_j'].fillna(0)

    return full_df

# Master function
def merge_and_clean_data(tmp_path, output_path):

    firm_df, B2B_df = upload_data(tmp_path)

    full_df = merge_data(firm_df, B2B_df)

    full_df = clean_merged_data(full_df)

    full_df.to_parquet(os.path.join(tmp_path, 'full_data_cleaned.parquet'), engine='pyarrow')
