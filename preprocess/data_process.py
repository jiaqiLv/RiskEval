import os
import csv
import glob
from numpy import dtype
import pandas as pd

RAW_DATA_LIST = glob.glob('./data/raw/**', recursive=True)
USER_DATA_LIST = [file_name for file_name in RAW_DATA_LIST if 'user_data' in file_name]
TRANSACTION_DATA_LIST = [file_name for file_name in RAW_DATA_LIST if 'transaction' in file_name]

if __name__ == '__main__':
    # step1: load dataframe
    for user_data in USER_DATA_LIST:
        user_df = pd.read_csv(user_data)
        user_df = user_df.drop(user_df.columns[0],axis=1)
        # step2: Class feature transformation
        for key in user_df:
            if user_df[key].dtypes == dtype('O'):
                encoded_df = pd.get_dummies(user_df[key], prefix=key)
                user_df = pd.concat([user_df, encoded_df], axis=1)
                user_df = user_df.drop(key,axis=1)
            elif user_df[key].dtypes == dtype('int64'):
                pass
        # print(user_df.keys())
        # step3: Write the processed data back to the csv file
        processed_user_data = user_data.replace('raw','processed')
        folder_path = os.path.dirname(processed_user_data)
        file_name = os.path.basename(processed_user_data)
        os.makedirs(folder_path, exist_ok=True)
        user_df.to_csv(os.path.join(folder_path,file_name),index=False)
