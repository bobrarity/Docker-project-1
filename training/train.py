import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model():
    try:
        df = pd.read_csv('train.csv')
    except FileNotFoundError:
        print('Training data file not found. Please check the file path.')
        return
    except Exception as e:
        print(f'An error occurred while loading training data: {e}')
        return

    try:
        X = df.drop('target', axis=1)
        y = df['target']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    except Exception as e:
        print(f'An error occurred in data preprocessing: {e}')
        return

    try:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.long)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = IrisNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(100):
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        torch.save(model.state_dict(), '/app/shared_folder/trained_model.pth')
    except Exception as e:
        print(f'An error occurred during training or model saving: {e}')


if __name__ == '__main__':
    train_model()
