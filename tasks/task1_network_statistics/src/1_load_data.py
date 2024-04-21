import pandas as pd

import os 
import shutil
if 'abs_path' not in globals():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path,'..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% 1. Upload data

flag = os.getenv('data')
name_real_data_B2B = os.getenv('name_real_B2B')
name_real_data_firm = os.getenv('name_real_firm')

if flag == 'random':
    
    # copy the random data from the previous task in the input folder
    dir1 = os.path.join(abs_path, 'task0_random_data','output')
    target_dir = os.path.join(abs_path, 'task1_network_statistics','input') 
    for filename in os.listdir(dir1):
        src_path = os.path.join(dir1, filename)  # Source file path
        dest_path = os.path.join(target_dir, filename)  # Destination file path
        shutil.copy2(src_path, dest_path)  # Copy and replace file
    
    B2B_df = pd.read_stata( os.path.join(abs_path, 'task1_network_statistics','input','B2B_network_2002_2022.dta') )
    firms_df = pd.read_stata( os.path.join(abs_path, 'task1_network_statistics','input','firm_data_2002_2022.dta'))
    
elif flag == 'real':
    
    # in this case the initial real data should be manually included in the input folder of task1
    B2B_df = read_data( os.path.join(abs_path, 'task1_network_statistics','input',f'{name_real_data_B2B}'))
    firms_df = read_data( os.path.join(abs_path, 'task1_network_statistics','input',f'{name_real_data_firm}') )
    
#%% 2. Merge B2B to firm dataset

# merge network and firm data
full_df = B2B_df.merge( firms_df[['year','vat','nace','turnover']].rename(
    columns={'vat': 'vat_i', 'turnover':'turnover_i', 'nace':'nace_i'}), on=['vat_i','year'], how='inner' )
full_df = full_df.merge( firms_df[['year','vat','nace','turnover']].rename(
    columns={'vat': 'vat_j', 'turnover':'turnover_j', 'nace': 'nace_j'}), on=['vat_j','year'], how='inner' )

# some firms in vat_j do not report their turnover, so we need to impute it from the B2B sales
sales_sum_by_vat_i = full_df.groupby('vat_i')['sales_ij'].sum().reset_index()
sales_sum_by_vat_i.rename(columns={'vat_i': 'vat_j', 'sales_ij': 'sum_sales_ij'}, inplace=True)
full_df = full_df.merge(sales_sum_by_vat_i, on='vat_j', how='inner')
full_df['turnover_j'] = full_df.apply(
    lambda row: row['sum_sales_ij'] if pd.isna(row['turnover_j']) else row['turnover_j'], 
    axis=1)
full_df.drop(columns=['sum_sales_ij'], inplace=True)

#%% 3. Save to pickles

data_dict = {'B2B_df': B2B_df, 'firms_df': firms_df, 'full_df': full_df}
save_workspace(data_dict, os.path.join( abs_path, 'task1_network_statistics','tmp','init_data.pkl'))

