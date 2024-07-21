import numpy as np
import scipy as sc
import scipy.special as sp
import matplotlib.pyplot as plt

def inverse_q_function(val: float):
    res = sc.stats.norm.ppf(val, loc=0, scale=1)
    return res


def calculate_threshold(dim: int, gamma: float):
    res = np.sqrt(8 * dim * (dim + 2)) * inverse_q_function(1 - gamma / 2)
    return res


def create_samples(snr: float, number_of_samples: int = 1000):
    # Generate a random binary signal (+1 or -1)
    signal = sc.stats.bernoulli.rvs(0.5, size=number_of_samples)
    signal[signal == 0] = -1

    # Calculate the required noise power for the given SNR
    snr_linear = 10 ** (-snr / 10)

    # Generate Gaussian noise with the calculated noise power
    noise = np.random.normal(0, np.sqrt(snr_linear), number_of_samples)

    # Combine signal and noise
    samples = signal + noise

    return samples

def calc_kurtosis_test(samples: np.ndarray, p:int=1):
    N = len(samples)
    mu = np.mean(samples)
    std = np.std(samples)
    normed = ((samples - mu) / std) ** 4
    B = np.mean(normed)
    test = np.sqrt(N) * (B - p * (p + 2))
    return test


if __name__ == "__main__":
    pd_list = []
    snr_values = [-10, -5, 0, 5, 10]
    monte_carlo_iter = 1000
    threshold = calculate_threshold(dim=1, gamma=0.05)
    for snr in snr_values:
        pd = 0
        for i in range(monte_carlo_iter):
            x = create_samples(snr=snr)
            pd += (np.abs(calc_kurtosis_test(x)) > threshold) / monte_carlo_iter
        pd_list.append(pd)
    plt.title("Detection as a function of SNR")
    plt.semilogy(snr_values, pd_list)
    plt.xlabel("SNR [dB]")
    plt.ylabel("$P_{D}$")
    plt.grid()
    plt.show()
