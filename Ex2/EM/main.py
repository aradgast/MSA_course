from data_class import Samples
from em_algo import EMAlgo
import numpy as np

if __name__ == "__main__":
    params = {0: {"weight": 0.4, "mean": np.array([0, 0]), "cov": np.array([[1, 0.9], [0.9, 1]])},
              1: {"weight": 0.4, "mean": np.array([0, 0]), "cov": np.array([[1, -0.9], [-0.9, 1]])},
              2: {"weight": 0.2, "mean": np.array([0, 0]), "cov": np.array([[0.05, 0], [0, 5]])}}
    weights = np.array([0.4, 0.4, 0.2])[:, None]
    mus = np.zeros((3, 2))
    cov1 = np.array([[1, 0.9], [0.9, 1]])
    cov2 = np.array([[1, -0.9], [-0.9, 1]])
    cov3 = np.array([[0.05, 0], [0, 5]])
    data_model = Samples(params)
    data_model.create(number_of_samples=1000)
    data_model.plot_samples()

