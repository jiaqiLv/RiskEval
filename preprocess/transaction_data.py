import pandas as pd

transaction_data = pd.read_csv("./data/transaction.csv")
print(transaction_data.shape)
# 根据用户ID分组(只取大于20条交易记录的用户)
filtered_groups = transaction_data.groupby('CUSTOMER_ID').filter(lambda x: len(x) > 20).groupby('CUSTOMER_ID')
# 计算每个CUSTOMER_ID的总数和TX_FRAUD为1的数量
customer_stats = filtered_groups.agg(
    total_transactions=('TX_FRAUD', 'count'),
    fraud_transactions=('TX_FRAUD', lambda x: (x==1).sum())
)
# 计算fraud_transactions占total_transactions的比例
customer_stats['fraud_ratio'] = customer_stats['fraud_transactions'] / customer_stats['total_transactions']

# 按照fraud_ratio列进行排序
customer_stats_sorted = customer_stats.sort_values(by='fraud_ratio', ascending=False)

# 选前1000个用户作为风险用户的交易数据,这些用户的风险交易占比30%-60%
fraud_transaction_customer = customer_stats_sorted.iloc[:1000].sample(n=300)
fraud_transaction_data = transaction_data[transaction_data['CUSTOMER_ID'].isin(fraud_transaction_customer.index)].groupby('CUSTOMER_ID')

# 选后3000-5000个用户作为正常用户的交易数据,这些用户的风险交易占比小于1.3%
normal_transation_customer = customer_stats_sorted.iloc[3000:].sample(n=700)
normal_transaction_data = transaction_data[transaction_data['CUSTOMER_ID'].isin(normal_transation_customer.index)].groupby('CUSTOMER_ID')

data_list = []
for idx, (customer_id, data) in enumerate(normal_transaction_data, start=1):
    # 对每个组，取最多 30 条数据
    sampled_data = data.sample(min(len(data), 30))
    data_list.append(sampled_data.assign(user_id=idx))

for idx, (customer_id, data) in enumerate(fraud_transaction_data, start=701):
    # 对每个组，取最多 30 条数据
    sampled_data = data.sample(min(len(data), 30))
    data_list.append(sampled_data.assign(user_id=idx))


final_data = pd.concat(data_list, ignore_index=True)

final_data.to_csv('transaction_data.csv')