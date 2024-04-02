import pandas as pd
import numpy as np
import random
import argparse
import os


# 设置命令行参数
parser = argparse.ArgumentParser(description='处理用户和交易数据')
parser.add_argument('--risk_ratio', type=float, default=10.0,
                    help='风险用户比例，默认为10%')
parser.add_argument('--transfer_weight', type=int, default=3,
                    help='转账权重，默认为3')

# 解析命令行参数
args = parser.parse_args()

# 使用命令行参数
risk_user_ratio = args.risk_ratio
transfer_weight = args.transfer_weight

# 构造文件夹名称
folder_name = f"risk-ratio{risk_user_ratio}%_trans-weight{transfer_weight}"
# 创建文件夹
os.makedirs(folder_name, exist_ok=True)

# 计算风险用户和正常用户的数量
risk_user_count = int(1000 * (risk_user_ratio / 100))
normal_user_count = 1000 - risk_user_count

print(risk_user_count, normal_user_count)

# 处理user_data表
user_info_data = pd.read_csv("../data/train.csv")
print(user_info_data.shape)

# 先随机找出700个正常用户和300个风险用户
normal_user_data = user_info_data[user_info_data['label'] == 0].sample(n=normal_user_count)
fraud_user_data = user_info_data[user_info_data['label'] == 1].sample(n=risk_user_count)
user_data = pd.concat([normal_user_data, fraud_user_data], ignore_index=True)
user_data['id'] = range(1, len(user_data) + 1)

# 保存用户数据，文件名中包含风险用户比例
user_data_filename = os.path.join(folder_name, f"user_data_risk-ratio{risk_user_ratio}%.csv")
user_data.to_csv(user_data_filename)

# 处理transaction_data表
transaction_data = pd.read_csv("../archive/Final Transactions.csv",index_col=0)
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
# 随机选取100个风险用户
fraud_transaction_customer = customer_stats_sorted.iloc[:1000].sample(n=risk_user_count)
fraud_transaction_data = transaction_data[transaction_data['CUSTOMER_ID'].isin(fraud_transaction_customer.index)].groupby('CUSTOMER_ID')

# 选后3000-5000个用户作为正常用户的交易数据,这些用户的风险交易占比小于1.3%
# 随机选取900个正常用户
normal_transation_customer = customer_stats_sorted.iloc[3000:].sample(n=normal_user_count)
normal_transaction_data = transaction_data[transaction_data['CUSTOMER_ID'].isin(normal_transation_customer.index)].groupby('CUSTOMER_ID')

# 向数据中添加交易类型
# 正常用户：取款：存款：转账=1:1:1
# 风险用户：取款：存款：转账=1:1:3
# 转账、取款、存款可以分别使用1 、 2 、 3来表示
data_list = []
for idx, (customer_id, data) in enumerate(normal_transaction_data, start=1):
    # 对每个组，取全部数据
    sampled_data = data.sample(len(data))
    tx_types = np.random.randint(1, 4, size=len(data))
    data_list.append(data.assign(TX_TYPE=tx_types, user_id=idx))

for idx, (customer_id, data) in enumerate(fraud_transaction_data, start=normal_user_count+1):
    # 对每个组，取全部数据
    sampled_data = data.sample(len(data))
    tx_types = [random.choices([1, 2, 3], [transfer_weight, 1, 1])[0] for _ in range(len(data))]
    data_list.append(data.assign(TX_TYPE=tx_types, user_id=idx))


final_data = pd.concat(data_list, ignore_index=True)

# 保存交易数据，文件名中包含风险用户比例和转账权重
transaction_data_filename = os.path.join(folder_name, f"transaction_risk-ratio{risk_user_ratio}%_trans-weight{transfer_weight}.csv")
final_data.to_csv(transaction_data_filename)