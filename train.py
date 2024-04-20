import pandas as pd

# 读取tsv文件
df = pd.read_csv('data/blockchair_ethereum_transactions_20160218.tsv', delimiter='\t')

# 统计每一列的值分布
for column in df.columns:
    print(f"Column: {column}")
    print(df[column].value_counts())
