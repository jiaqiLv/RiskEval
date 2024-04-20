import pandas as pd

user_info_data = pd.read_csv("./data/user_info.csv")
print(user_info_data.shape)
# 先随机找出700个正常用户和300个风险用户
normal_user_data = user_info_data[user_info_data['label'] == 0].sample(n=700)
fraud_user_data = user_info_data[user_info_data['label'] == 1].sample(n=300)
user_data = pd.concat([normal_user_data, fraud_user_data], ignore_index=True)
user_data['id'] = range(1, len(user_data) + 1)
user_data.to_csv("user_data.csv")
