import numpy as np
from data_loader import load_dataset
from linear_model import fit

def train(path_train, output_file, method='RH'):
    X, Y, _  = load_dataset(path_train)
    W = fit(X, Y, method=method)
    np.save(output_file, W)

def predict(W_file, X):
    W = np.load(W_file)

    y_score = W @ X
    return np.argmax(y_score, axis=0)


W = train('cats_and_dogs/train', './W.SVD_permisive', 'SVD')