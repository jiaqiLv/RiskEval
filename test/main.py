import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据加载和预处理
df = pd.read_csv('./data/processed/risk-ratio10.0%_trans-weight3/user_data_risk-ratio10.0%.csv')
X = df.drop(df.columns[[0, 8]], axis=1)
y = df.iloc[:, 9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义PyTorch数据集
class RiskDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_data = RiskDataset(torch.FloatTensor(X_train), 
                         torch.FloatTensor(y_train.values))
test_data = RiskDataset(torch.FloatTensor(X_test), 
                        torch.FloatTensor(y_test.values))


# 定义数据加载器
batch_size = 32
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# 定义神经网络模型
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer_1 = nn.Linear(X_train.shape[1], 32) 
        self.layer_2 = nn.Linear(32, 32)
        self.layer_out = nn.Linear(32, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

# 实例化模型，定义损失函数和优化器
model = DNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
def train(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    print(f"Training Loss: {epoch_loss:.3f}, Training Accuracy: {epoch_acc:.3f}")


# 测试模型
def test(model, criterion, test_loader):
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            print(f"Test Accuracy: {acc:.3f}, Test Loss: {loss:.3f}")

# 计算二元分类的准确率
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

# 开始训练和测试
epochs = 30
for e in range(1, epochs+1):
    train(model, criterion, optimizer, train_loader)
    if e%10 == 0:
        print(f"Epoch {e}")
        test(model, criterion, test_loader)
