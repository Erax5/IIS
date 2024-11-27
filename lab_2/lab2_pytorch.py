import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch


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
        self.inputs = torch.tensor(data.drop("Emotion", axis=1).to_numpy(dtype=np.float32))

        # we need to transform label (str) to a number. In sklearn, this is done internally
        self.index2label = [label for label in data["Emotion"].unique()]
        label2index = {label: i for i, label in enumerate(self.index2label)}

        self.labels = torch.tensor(data["Emotion"].apply(lambda x: torch.tensor(label2index[x])))

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)


def main():
    dataset = MultiEmoVA("Face-Feat.txt")

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

    loss_fn = nn.CrossEntropyLoss()

    # Check if we have GPU acceleration, if we do our code will run faster
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if you are on mac with the new M1, M2, ... chips you can try the following instead of cuda
    device = "mps" if torch.backends.mps.is_available() else device

    print(f"Using device: {device}")

    # we need to move our model to the correct device
    model = model.to(device)

    # it is common to do a training loop multiple times, we call these 'epochs'
    for epoch in range(6):
        for inputs, labels in train_loader:
            # both input, output and model need to be on the same device
            inputs = inputs.to(device)
            labels = labels.to(device)

            out = model(inputs)
            loss = loss_fn(out, labels)

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
