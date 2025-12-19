import numpy as np
from data_loader import load_dataset
from linear_model import fit

def main(path_train, output_file, method):
    X, Y = load_dataset(path_train)
    W = fit(X, Y, method=method)
    np.save(output_file, W)

