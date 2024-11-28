import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms

class MLP(nn.Module):
    def __init__(self, features_in=20, features_out=7):
        super().__init__()
        self.linear1 = nn.Linear(features_in, 1000)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(1000, 500)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(500, features_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.relu(self.linear1(input))
        x = self.dropout1(x)
        x = self.relu(self.linear2(x))
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

class MultiEmoVA(Dataset):
    def __init__(self, data_path):
        super().__init__()
        data = pd.read_csv(data_path)
        self.inputs = torch.tensor(data.drop("emotion", axis=1).to_numpy(dtype=np.float32))
        self.index2label = [label for label in data["emotion"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}
        self.labels = torch.tensor(data["emotion"].apply(lambda x: label2index[x]).to_numpy(dtype=np.int64))

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
            # print(f"counter: {self.counter}, best_loss: {self.best_loss:.2f}")
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
    features_in = dataset.inputs.shape[1]
    features_out = len(dataset.index2label)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = torch.Generator().manual_seed(2023)
    train, val, test = random_split(dataset, [0.7, 0.2, 0.1], generator=generator)

    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)

    model = MLP(features_in, features_out)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    best_accuracy = 0
    early_stopping = EarlyStopping(patience=10)

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_accuracy, val_loss = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if early_stopping(val_loss):
            print("Early stopping")
            break

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

    # Test the best model
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()