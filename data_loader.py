import numpy as np
from sklearn.model_selection import train_test_split

def preprocessing(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def normalize_data(X_train, X_valid, X_test):
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_valid, X_test

# If you are using your own dataset, you must:
#     1. Upload your four .npy files (X_train, X_test, y_train, y_test) inside a folder.
#        insert the new folder inside of datasets
#        change << def load_data(folder_name='datasets/YourDatasetHere}' >>

def load_data(folder_name='datasets/CharacterTrajectories', mode="train"):
    
    X_train = np.load(f'{folder_name}/x_train.npy')
    X_test = np.load(f'{folder_name}/x_test.npy')
    y_train = np.load(f'{folder_name}/y_train.npy')
    y_test = np.load(f'{folder_name}/y_test.npy')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    X_train, X_valid, X_test = normalize_data(X_train, X_valid, X_test)

    unique_labels = np.unique(y_train)
    num_classes = len(unique_labels)
    input_size = X_train.shape[1]

    dataset_name = folder_name.split("/")[-1].upper()
    if mode == "train":
        print(f"\n~~~ {dataset_name} DATASET LOADED < STARTING TRAINING:\n")
    elif mode == "eval":
        print(f"\n~~~ {dataset_name} DATASET LOADED < OBTAINING mAP EVALUATION:\n")

    return X_train, X_valid, X_test, y_train, y_valid, y_test, num_classes, input_size