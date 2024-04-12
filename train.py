import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


USER_DATA_PATH = 'preprocess/user_data_risk-ratio5.0%.csv'

def plot_accuracy(accuracies):
    plt.plot(accuracies)
    plt.title('Accuracy Plot')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    # step1: data partition
    data_df = pd.read_csv(USER_DATA_PATH)
    y = np.array(data_df['label'].values)
    X_df = data_df.drop(['id','label'],axis=1)
    X = np.array(X_df.values)
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

    # step2: model definition
    model = XGBClassifier()

    # step3: train
    model.fit(X_train,y_train)

    # step4: evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    print(predictions)
