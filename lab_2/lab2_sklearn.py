# import useful libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def train_and_eval(model, train_in, train_out, val_in, val_out):
    model.fit(train_in, train_out)
    predicted_val = model.predict(val_in)
    print("\nPredicted classes: ", predicted_val, "\n")

    # Evaluate model
    return accuracy_score(val_out, predicted_val)


def main():
    # read and print data
    data = pd.read_csv("Face-Feat.txt")
    print(data)

    # See classes
    print("Unique classes", data["Emotion"].unique(), "\n")

    # see class balance
    for class0 in data["Emotion"].unique():
        print(f"Found {(data['Emotion'] == class0).value_counts().iloc[1]} samples for class {class0}")

    # Let split the dataset for training
    labels = data["Emotion"]
    inputs = data.drop("Emotion", axis=1)

    # split = 70/20/10
    data_in, test_in, data_out, test_out = train_test_split(
        inputs,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels  # balances labels across the sets
    )
    train_in, val_in, train_out, val_out = train_test_split(
        data_in,
        data_out,
        test_size=(0.2/0.9),  # 20% of the original data
        random_state=42,
        stratify=data_out
    )
    print("\nLenght of each split of the data: ", len(train_in), len(val_in), len(test_in), "\n")

    # Train model 1
    model_1 = DecisionTreeClassifier()
    print(
        "\nAccuracy of model_1: ",
        train_and_eval(model_1, train_in, train_out, val_in, val_out)
    )

    ### EXERCISE: train two additional models

    # Train model 2
    model_2 = SVC()
    print(
        "\nAccuracy of model_2: ",
        train_and_eval(model_2, train_in, train_out, val_in, val_out)
    )

    # Train model 3
    model_3 = KNeighborsClassifier()
    print(
        "\nAccuracy of model_3: ",
        train_and_eval(model_3, train_in, train_out, val_in, val_out)
    )

    # model_3 is the best. Let's evaluate on the test set
    print(
        "Best model accuracy on test set: ",
        accuracy_score(
            test_out,
            model_3.predict(test_in)
        )
    )

    # Hyperparameter search/tuning
    param_grid = [
        {"kernel": ["poly"], "degree": [3, 15, 25, 50]},
        {"kernel": ["rbf", "linear", "sigmoid"]}
    ]

    best_model = GridSearchCV(SVC(), param_grid)
    best_model.fit(train_in, train_out)  # Fits on all combinations and keeps best model

    print(
        "\n\nBest model with best parameters on test set: ",
        accuracy_score(
            test_out,
            best_model.predict(test_in)
        )
    )
    print(
        "Best parameters of best model: ",
        best_model.best_params_
    )

    ### EXERCISE: do a grid search on different hyperparameters of DecisionTree

    param_grid = [
        {"criterion": ["gini", "entropy"], "max_depth": [1, 2, 5, 500]}
    ]

    decision_tree_search = GridSearchCV(DecisionTreeClassifier(), param_grid)
    decision_tree_search.fit(train_in, train_out)
    print(
        "\n\nDecision tree with best parameters on test set: ",
        accuracy_score(
            test_out,
            decision_tree_search.predict(test_in)
        )
    )
    print(
        "Best parameters of Decision tree: ",
        decision_tree_search.best_params_
    )


if __name__ == "__main__":
    main()
