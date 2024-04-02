import csv
from numpy import dtype
import pandas as pd
import os
from os.path import join,exists

USER_DATA_PATH = '/code/RiskEval/data/raw/risk-ratio5.0%_trans-weight2/user_data_risk-ratio5.0%.csv'
TRANSACTION_DATA_PATH = '/code/RiskEval/data/raw/risk-ratio5.0%_trans-weight2/transaction_risk-ratio5.0%_trans-weight2.csv'

def show_basic_info(user_df,transaction_df):
    print(user_df.head())
    print('-'*150)
    print(transaction_df.head())

if __name__ == '__main__':
    # step1: load dataframe
    user_df = pd.read_csv(USER_DATA_PATH)
    transaction_df = pd.read_csv(TRANSACTION_DATA_PATH)
    user_df = user_df.drop(user_df.columns[0],axis=1)
    transaction_df = transaction_df.drop(transaction_df.columns[0],axis=1)
    # step2: Class feature transformation
    for key in user_df:
        if user_df[key].dtypes == dtype('O'):
            encoded_df = pd.get_dummies(user_df[key], prefix=key)
            user_df = pd.concat([user_df, encoded_df], axis=1)
            user_df = user_df.drop(key,axis=1)
        elif user_df[key].dtypes == dtype('int64'):
            pass
    print(user_df.keys())
    # step3: Write the processed data back to the csv file
    TARGET_USER_DATA_PATH = USER_DATA_PATH.replace('raw','processed')
    folder_path = os.path.dirname(TARGET_USER_DATA_PATH)
    file_name = os.path.basename(TARGET_USER_DATA_PATH)
    os.makedirs(folder_path, exist_ok=True)
    user_df.to_csv(join(folder_path,file_name),index=False)
