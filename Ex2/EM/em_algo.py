import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal

class EMAlgo:
    def __init__(self, mix_number: int = 3, tol: float = 1e-3):
        self.mix_number = mix_number
        self.tol = tol

    def apply(self, data):
        loss_list = []
        params = self._init_params(data.shape[-1])
        loss_diff = np.inf
        prev_loss = 0
        while np.abs(loss_diff) > self.tol:
            e = self.e_step(data, params)
            new_params = self.m_step(data, e)
            curr_loss = self.calc_loss(data, new_params)
            loss_diff = curr_loss - prev_loss
            prev_loss = curr_loss
            loss_list.append(curr_loss)
            params = new_params

        return params, loss_list

    def calc_loss(self, data, params):
        loss = 0
        for key, val in params.items():
            loss += val.get("weights") * multivariate_normal.pdf(data, mean=val.get("mean"), cov=val.get("cov"))
        loss = np.sum(np.log(loss))
        return loss

    def e_step(self, data, params):
        denomerator = 0
        e = np.zeros((self.mix_number, data.shape[0]))
        for key, val in params.items():
            numerator = val.get("weights") * multivariate_normal.pdf(data, mean=val.get("mean"), cov=val.get("cov"))
            denomerator += numerator
            e[key] = numerator
        e /= denomerator
        return e

    def m_step(self, data, e):
        new_weights = np.mean(e, axis=1)[:, None]

        new_means = np.zeros((self.mix_number, data.shape[-1]))
        for m in range(self.mix_number):
            new_means[m] = np.sum(e[m][:, None] * data, axis=0) / np.sum(e[m])
        new_covs = np.zeros((self.mix_number, data.shape[-1], data.shape[-1]))
        for m in range(self.mix_number):
            diff = data - new_means[m]
            new_covs[m] = np.matmul((e[m][:, None] * diff).T, diff) / np.sum(e[m])

        new_params = {k: {"weights": w, "mean": mu, "cov": sig} for k, (w, mu, sig) in enumerate(zip(new_weights, new_means, new_covs))}
        return new_params



    def _init_params(self, dim: int):
        weights = np.random.uniform(0, 1, size=(self.mix_number, 1))
        weights /= np.sum(weights)

        means = np.random.uniform(0, 1, size=(self.mix_number, dim))
        covs = self.__init_covs(dim)

        params = {k: {"weights": w, "mean": mu, "cov": sig} for k, (w, mu, sig) in enumerate(zip(weights, means, covs))}
        return params

    def __init_covs(self, dim):
        covs = np.random.rand(self.mix_number, dim, dim)
        for m in range(self.mix_number):
            covs[m] = covs[m] @ covs[m].transpose() + np.eye(dim)
        return covs

    def plot_results(self):
        pass
