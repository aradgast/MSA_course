from data_class import Samples
from em_algo import EMAlgo
import numpy as np
import matplotlib.pyplot as plt
from functions import ellips_plot_2d

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
    data_model.create(number_of_samples=10000)
    em_algo = EMAlgo(tol=1e-6)
    params_est, loss = em_algo.apply(data_model.data)
    print(params_est)
    plt.plot(loss)
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()
    # plot the distribution of the new params
    data_model.plot_samples()
    for key, val in params.items():
        if key == 0:
            ellips_plot_2d(val.get("mean"), val.get("cov"), 100, 'b', label="True")
        else:
            ellips_plot_2d(val.get("mean"), val.get("cov"), 100, 'b')
    for key, val in params_est.items():
        if key == 0:
            ellips_plot_2d(val.get("mean"), val.get("cov"), 100, 'r', label="Estimated")
        else:
            ellips_plot_2d(val.get("mean"), val.get("cov"), 100, 'r')
    plt.legend()
    plt.show()
