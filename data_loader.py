import numpy as np
from pathlib import Path

def load_dataset(root_path):
    """
    Loads the dataset.
    Assumes that .npy files have the format (Rows=Samples).
    """
    root = Path(root_path)
    classes = sorted([classFolder.name for classFolder in root.iterdir() if classFolder.is_dir()])
    
    list_data = []
    list_tags = []

    for i, class_name in enumerate(classes):
        class_folder = root / class_name
        
        file_path = list(class_folder.glob('*.npy'))[0]
        data = np.load(file_path)
            
        n_samples = data.shape[0]
        labels = np.full(n_samples, i)
            
        list_data.append(data.T)
        list_tags.append(labels)

    X = np.hstack(list_data)
    Y = one_hot_encoding(np.concatenate(list_tags))

    return X, Y, classes

def one_hot_encoding(tags):
    """
    Converts a vector of integer labels into a One-Hot matrix.
    """
    n_classes = np.max(tags) + 1
    n_samples = len(tags)
    
    Y_onehot = np.zeros((n_samples, n_classes))
    Y_onehot[np.arange(n_samples), tags] = 1
    
    return Y_onehot.T