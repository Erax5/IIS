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
    def __init__(self, input_size, output_size, hidden_layers):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size

        # Add hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        # Add output layer
        layers.append(nn.Linear(in_features, output_size))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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
    # for index, data in enumerate(dataset):
    #     print(f"Index: {index}, Data: {data[0]}, Label: {dataset.index2label[data[1]]}")
    #     if index == 5:
    #         break

    # passing a generator to random_split is similar to specifying the seed in sklearn
    generator = torch.Generator().manual_seed(2023)

    # this can also generate multiple sets at the same time with e.g. [0.7, 0.2, 0.1]
    train, test = random_split(dataset, [0.8, 0.2], generator=generator)

    train_loader = DataLoader(  # this loads the data that we need dynamically
        train,
        batch_size=4,  # instead of taking 1 data point at a time we can take more, making our training faster and more stable
        shuffle=True  # Shuffles the data between epochs (see below)
    )

    hidden_layers = [50, 100, 50]
    model = MLP(train[0][0].shape[0], len(dataset.index2label), hidden_layers)

    optim = torch.optim.SGD(model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()

    # Check if we have GPU acceleration, if we do our code will run faster
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # we need to move our model to the correct device
    model = model.to(device)

    # Training loop
    num_epochs = 100
    for _ in range(num_epochs):
        # model.train()  # Set the model to training mode
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optim.step()
            optim.zero_grad()
   
    # tell pytorch we're not training anymore
    with torch.no_grad():
        test_loader = DataLoader(test, batch_size=4)
        correct = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            # Here we go from the models output to a single class and compare to ground truth
            correct += (predictions.softmax(dim=1).argmax(dim=1) == labels).sum()
        print(f"Accuracy is: {correct / len(test) * 100}%")

if __name__ == "__main__":
    main()
