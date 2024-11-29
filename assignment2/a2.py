import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np

# ===================================Hyperparameter tuning===================================
# config while testing:
# layers_config = [
#     [100, 50],
#     [200, 100, 50],
#     [300, 200, 100, 50],
# ]
layers_config = [
    [100, 50]
]
# config while testing:
# dropouts = [
#     0.1, 0.3, 0.5, 0.7,
# ]
dropouts = [
    0.1
]
# config while testing:
# learning_rates = [
#     0.001, 0.01, 0.1,
# ]
learning_rates = [
    0.01
]
# config while testing:
# epochs = np.arange(50, 201, 50)
epochs = [50]

# ===========================================================================================

# multi-layer perceptron class
class MLP(nn.Module):
    def __init__(self, features_in, features_out, layer, dropout):
        super().__init__()
        layers = []
        in_size = features_in
        # iterate over layers list to create matching neural network
        for h in layer:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU()) # activation function to introduce non-linearity
            layers.append(nn.Dropout(dropout)) # randomly drop neurons to prevent overfitting
            in_size = h # update input size for next layer to match output size of current layer
        layers.append(nn.Linear(in_size, features_out)) # output layer
        self.network = nn.Sequential(*layers) # create neural network with layers list

    # define forward pass
    def forward(self, input):
        return self.network(input) 

# class to read and preprocess training dataset
class MultiEmoVA(Dataset):
    def __init__(self, data_path):
        super().__init__()

        data = pd.read_csv(data_path)
        self.inputs = torch.tensor(data.drop("emotion", axis=1).to_numpy(dtype=np.float32))
        self.index2label = [label for label in data["emotion"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}
        self.labels = torch.tensor(data["emotion"].apply(lambda x: label2index[x]))

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)

# class to read and preprocess test dataset
class TestDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        data = pd.read_csv(data_path)
        self.inputs = torch.tensor(data.to_numpy(dtype=np.float32))

    def __getitem__(self, index):
        return self.inputs[index]

    def __len__(self):
        return len(self.inputs)
    
# based on: https://github.com/Bjarten/early-stopping-pytorch/blob/main/early_stopping_pytorch/early_stopping.py
# purpose is to stop training when validation loss stops decreasing
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

def train_model(model, train_loader, loss_fun, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate_model(model, val_loader, loss_fun, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += loss_fun(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    val_loss /= len(val_loader)
    return accuracy, val_loss

def main():
    dataset = MultiEmoVA("dataset.csv")
    generator = torch.Generator().manual_seed(2024)
    train, val, test = random_split(dataset, [0.80, 0.15, 0.05], generator=generator)

    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    val_loader = DataLoader(val, batch_size=128, shuffle=False)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)

    features_in = train[0][0].shape[0] # number of features in input, i.e. 20
    features_out = len(dataset.index2label) # number of output features, i.e. 7

    # best_model = None
    best_accuracy = 0
    best_config = None
    best_drop_out = None
    best_lr = None
    best_epoch = None

    for layer in layers_config:
        print(f"Training with layers: {layer}")
        for dropout in dropouts:
            for lr in learning_rates:
                for epoch_config in epochs:
                    model = MLP(features_in, features_out, layer, dropout)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    device = "mps" if torch.backends.mps.is_available() else device
                    model = model.to(device)

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    loss_fun = nn.CrossEntropyLoss()

                    early_stopping = EarlyStopping(patience=5)

                    for epoch in range(epoch_config):
                        train_loss = train_model(model, train_loader, loss_fun, optimizer, device)
                        val_accuracy, val_loss = evaluate_model(model, val_loader, loss_fun, device)
                        # print(f"Epoch [{epoch+1}/100], Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

                        if early_stopping(val_loss):
                            break

                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            best_config = layer
                            best_drop_out = dropout
                            best_lr = lr
                            best_epoch = epoch_config

    print(f"Best Validation Accuracy: {best_accuracy:.2f}% with layers: {best_config}, dropout: {best_drop_out}, lr: {best_lr}, epoch: {best_epoch}")

    # Test the best model
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / len(test) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    test_dataset = TestDataset("test_to_submit.csv")
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    emotions = ['neutral', 'disgust', 'sad', 'happy', 'surprise', 'angry', 'fear']

    with open('outputs', 'a') as f:
        for i, prediction in enumerate(predictions):
            if i < len(predictions) - 1:
                f.write(emotions[prediction] + '\n')
            else:
                f.write(emotions[prediction])

if __name__ == "__main__":
    main()