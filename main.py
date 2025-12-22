import numpy as np
from data_loader import load_dataset
from linear_model import fit

def train(path_train, output_file, method='RH'):
    X, Y, _  = load_dataset(path_train)
    W = fit(X, Y, method=method)
    np.save(output_file, W)

def predict(W, X, load_from_file=True):
    if load_from_file:
        W = np.load(W)

    y_score = W @ X
    return np.argmax(y_score, axis=0)

#train('cats_and_dogs/train', './W_model.npy')

W = np.load('cats_and_dogs/train/cats/efficientnet_b3_embeddings.npy')