import pandas as pd
import numpy as np

import os 
import shutil
if 'abs_path' not in globals():
    abs_path = os.path.abspath(os.path.dirname(__file__))
    abs_path = os.path.abspath(os.path.join(abs_path,'..','..'))

import sys
sys.path.append( abs_path )
from functions_network import*

#%% INPUT
dict_data = load_workspace( os.path.join( abs_path, 'task1_clean_data', 'tmp','input_data.pkl') )
B2B_df = dict_data['B2B_df']
firms_df = dict_data['firms_df']

#%% 1. Adjust NACE sectors

#extract NACE 4-digits codes, replace NACE 5-digits
firms_df['nace'] = firms_df['nace'].astype(str).str.slice(start=0, stop=4)

# include broad definition of industries
firms_df['nace_2digits'] = firms_df['nace'].astype(str).str.slice(start=0, stop=2)
firms_df['nace_2digits'] = pd.to_numeric(firms_df['nace_2digits'], errors='coerce')
firms_df['industry'] = firms_df['nace_2digits'].apply(lambda x: recategorize_industry(x) )
firms_df.drop(columns=['nace_2digits'], inplace=True)

#%% 2. Merge B2B to firm dataset

# merge network and firm data on vat_i
full_df = B2B_df.merge( firms_df[['year','vat','nace','turnover','industry']].rename(
    columns={'vat': 'vat_i', 'turnover':'turnover_i', 'nace':'nace_i','industry':'industry_i'}), 
    on=['vat_i','year'], how='inner' )

# merge network and firm data on vat_j
full_df = full_df.merge( firms_df[['year','vat','nace','inputs_total']].rename(
    columns={'vat': 'vat_j', 'inputs_total':'inputs_j', 'nace': 'nace_j'}), on=['vat_j','year'], how='inner' )

#%% 3. Adjust turnover

# 3.1. impute turnover from network sales
# it may happen that some firms in the B2B do not report their turnover, 
# so we need to impute it from the B2B sales
# correct sales of i
full_df['sum_sales_ij'] = full_df.groupby('vat_i')['sales_ij'].transform('sum')
full_df['turnover_i'] = np.where( pd.isna(full_df['turnover_i']), full_df['sum_sales_ij'], full_df['turnover_i'] ).T

# some firms may have B2B sales equal to 0: we need to remove those firms
full_df = full_df[full_df['sales_ij'] != 0]

# 3.2. correct sales from i to j larger than turnover of i
full_df['sales_ij'] = np.where([full_df['sales_ij'] > 
                                      full_df['turnover_i']],full_df['turnover_i'],full_df['sales_ij']).T

# 3.3. add some final demand (1 euro) to the sales if bilateral sales = total sales
full_df['turnover_i'] = np.where( [full_df['sales_ij'] == 
                                      full_df['turnover_i']], full_df['turnover_i'] + 1, full_df['turnover_i']  ).T

# 3.4 add some final demand (1 euro) to the sales if network sales = total sales
full_df['turnover_i'] = np.where( [full_df['sum_sales_ij'] == 
                                      full_df['turnover_i']], full_df['turnover_i'] + 1, full_df['turnover_i']  ).T
full_df.drop(columns=['sum_sales_ij'], inplace=True)

#%% 4. Adjust total inputs

# 4.1. impute total inputs from network purchases
# it may happen that some firms report 0 input use but there are reported purchases
# in the B2B network
full_df['sum_purch_ij'] = full_df.groupby('vat_j')['sales_ij'].transform('sum')
full_df['inputs_j'] = np.where( pd.isna(full_df['inputs_j']), full_df['sum_purch_ij'], full_df['inputs_j'] ).T

#%% 5. Save to pickles

data_dict = {'full_df': full_df}
save_workspace(data_dict, os.path.join( abs_path, 'task1_clean_data','output','init_data.pkl'))

