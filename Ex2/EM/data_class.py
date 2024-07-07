import numpy as np
import matplotlib.pyplot as plt


class Samples:
    def __init__(self, params: dict):
        self.params = params
        self.mix_number = len(self.params.keys())
        self.dim = self.params[0].get("mean").shape[0]
        self.data = None
        self.labels = None

    def create(self, number_of_samples):
        data = np.zeros((number_of_samples, self.dim))
        probs = [param.get("weight") for param in self.params.values()]
        weights = np.random.choice([i for i in range(self.mix_number)], size=number_of_samples, p=probs)
        for k, w in enumerate(weights):
            mu = self.params.get(w).get("mean")
            sigma = self.params.get(w).get("cov")
            data[k] = np.random.multivariate_normal(mean=mu, cov=sigma)
        self.data = data
        self.labels = weights

    def plot_samples(self):
        plt.figure(figsize=(11, 8))
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels)
        plt.title(f'Data Generation - {self.data.shape[0]} points')
        plt.grid()
        plt.tight_layout()
        # plt.savefig(f'Data Generation - {N} points.jpeg')
        plt.show()
