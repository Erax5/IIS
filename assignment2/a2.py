import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch

# 1. Read and Preprocess the dataset in a format that is appropriate for training
# 2. Do a balanced split of the dataset for train/val/test.
# 3. Choose pytorch model?
# 4. Do some kind of hyperparameter tuning/model selection using the validation dataset
# 5. Analyse
# 6. Classify test_to_submit dataset
# 7. Report & submit

class MLP(nn.Module):
    def __init__(self, features_in=2, features_out=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(features_in, 100),
            nn.ReLU(),
            nn.Linear(100, features_out)
        )

    def forward(self, input):
        return self.net(input)


class MultiEmoVA(Dataset):
    def __init__(self, data_path):
        super().__init__()

        data = pd.read_csv(data_path)
        # everything in pytorch needs to be a tensor
        self.inputs = torch.tensor(data.drop("emotion", axis=1).to_numpy(dtype=np.float32))

        # we need to transform label (str) to a number. In sklearn, this is done internally
        self.index2label = [label for label in data["emotion"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}

        self.labels = torch.tensor(data["emotion"].apply(lambda x: torch.tensor(label2index[x])))

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)


# labels contains the emotion e.g. as tensor(0)
# index2label contains the emotions as strings e.g. as "neutral"

# 0 = neutral
# 1 = disgust
# 2 = sad
# 3 = happy
# 4 = surprise
# 5 = angry
# 6 = fear
def main():
    dataset = MultiEmoVA("dataset.csv")
    # __getitem__(self, index):
    for index, data in enumerate(dataset):
        print(f"Index: {index}, Data: {data[0]}, Label: {dataset.index2label[data[1]]}")
        if index == 5:
            break

    # passing a generator to random_split is similar to specifying the seed in sklearn
    generator = torch.Generator().manual_seed(2023)

    # this can also generate multiple sets at the same time with e.g. [0.7, 0.2, 0.1]
    train, test = random_split(dataset, [0.8, 0.2], generator=generator)

    train_loader = DataLoader(  # this loads the data that we need dynamically
        train,
        batch_size=4,  # instead of taking 1 data point at a time we can take more, making our training faster and more stable
        shuffle=True  # Shuffles the data between epochs (see below)
    )
    model = MLP(train[0][0].shape[0], len(dataset.index2label))

    optim = torch.optim.SGD(model.parameters(), lr=0.001)

if __name__ == "__main__":
    main()





# # 1. Read and Preprocess the dataset in a format that is appropriate for training
# class EmotionDataset(Dataset):
#     def __init__(self, data_path):
#         data = pd.read_csv(data_path)
#         self.inputs = torch.tensor(data.drop("emotion", axis=1).values, dtype=torch.float32)
#         self.labels = torch.tensor(data["emotion"].astype('category').cat.codes.values, dtype=torch.long)
    
#     def __len__(self):
#         return len(self.inputs)
    
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.labels[idx]

# # 2. Do a balanced split of the dataset for train/val/test.
# def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
#     train_size = int(train_ratio * len(dataset))
#     val_size = int(val_ratio * len(dataset))
#     test_size = len(dataset) - train_size - val_size
#     return random_split(dataset, [train_size, val_size, test_size])

# # 3. Define a PyTorch model
# class MyModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(MyModel, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_size)
#         )
    
#     def forward(self, x):
#         return self.network(x)

# # 4. Hyperparameter tuning/model selection using the validation dataset
# def hyperparameter_tuning(train_loader, val_loader, input_size, output_size):
#     model = MyModel(input_size, output_size)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
#     best_model = None
#     best_accuracy = 0
    
#     for epoch in range(10):  # Example: 10 epochs
#         model.train()
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
        
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
        
#         accuracy = 100 * correct / total
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model = model.state_dict()
    
#     return best_model, best_accuracy

# # 5. Analyze the results
# def analyze_results(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     accuracy = 100 * correct / total
#     print(f'Test Accuracy: {accuracy}%')

# # 6. Classify test_to_submit dataset
# def classify_test_to_submit(model, data_path):
#     test_data = pd.read_csv(data_path)
#     inputs = torch.tensor(test_data.values, dtype=torch.float32)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#     return predicted

# # 7. Report & submit
# def main():
#     dataset = EmotionDataset('dataset.csv')
#     train_data, val_data, test_data = split_dataset(dataset)
    
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
#     test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
#     input_size = dataset.inputs.shape[1]
#     output_size = len(dataset.labels.unique())
    
#     best_model_state, best_accuracy = hyperparameter_tuning(train_loader, val_loader, input_size, output_size)
#     print(f'Best Validation Accuracy: {best_accuracy}%')
    
#     model = MyModel(input_size, output_size)
#     model.load_state_dict(best_model_state)
    
#     analyze_results(model, test_loader)
    
#     predictions = classify_test_to_submit(model, 'test_to_submit.csv')
#     print(f'Predictions for test_to_submit.csv: {predictions}')

# if __name__ == "__main__":
#     main()