import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class MLP(nn.Module):
    def __init__(self, features_in, features_out, hidden_layers):
        super().__init__()
        layers = []
        in_size = features_in
        for h in hidden_layers:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            in_size = h
        layers.append(nn.Linear(in_size, features_out))
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        return self.network(input)

class MultiEmoVA(Dataset):
    def __init__(self, data_path):
        super().__init__()

        data = pd.read_csv(data_path)
        scaler = StandardScaler()
        self.inputs = torch.tensor(scaler.fit_transform(data.drop("emotion", axis=1).to_numpy(dtype=np.float32)))

        self.index2label = [label for label in data["emotion"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}

        self.labels = torch.tensor(data["emotion"].apply(lambda x: label2index[x]))

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    val_loss /= len(val_loader)
    return accuracy, val_loss

def main():
    dataset = MultiEmoVA("dataset.csv")
    generator = torch.Generator()#.manual_seed(3)
    train, val, test = random_split(dataset, [0.7, 0.2, 0.1], generator=generator)

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)

    features_in = train[0][0].shape[0]  # Number of input features
    features_out = len(dataset.index2label)  # Number of output classes

    # Hyperparameter tuning
    hidden_layer_configs = [
        [100, 50],
        [200, 100, 50],
        [300, 200, 100, 50]
    ]

    best_model = None
    best_accuracy = 0
    best_config = None

    for hidden_layers in hidden_layer_configs:
        model = MLP(features_in, features_out, hidden_layers)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        early_stopping = EarlyStopping(patience=5)

        for epoch in range(100):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_accuracy, val_loss = evaluate_model(model, val_loader, criterion, device)

            print(f"Epoch [{epoch+1}/100], Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            if early_stopping(val_loss):
                print("Early stopping")
                break

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                best_config = hidden_layers
                torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Validation Accuracy: {best_accuracy:.2f}% with hidden layers: {best_config}")

    # Test the best model
    best_model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    best_model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / len(test) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()