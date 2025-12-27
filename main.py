import numpy as np
from data_managing import load_dataset
from linear_model import fit

def train(path_train, output_file, method='HH'):
    X, Y, classes = load_dataset(path_train)
    W = fit(X, Y, method=method)

    np.savez(output_file, W=W, classes=classes)

def predict(model_file, unlabeled_data_path):
    data = np.load(model_file)
    W = data['W']
    classes = data['classes']

    X = np.load(unlabeled_data_path).T
    y_score = W @ X
    
    indexed_classes = np.argmax(y_score, axis=0)    
    percentages = [(class_name, np.mean(indexed_classes == i) * 100) for i, class_name in enumerate(classes)]

    return classes[indexed_classes], percentages
