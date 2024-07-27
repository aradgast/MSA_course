import numpy as np
from scipy.stats import multivariate_normal, chi2
from sklearn.neighbors import KernelDensity


# Function to generate data
def generate_data(N):
    X = np.random.randn(N, 1)
    Y = np.cos(X) + np.random.randn(N, 1)
    Z = np.hstack((X, Y))
    return Z

def get_det(matrix):
    matrix = np.atleast_2d(matrix)
    if matrix.shape[0] == 1:
        return np.abs(matrix[0])
    return np.linalg.det(matrix)


# Function to whiten data
def whiten_data(Z):
    Sigma_Z = np.cov(Z, rowvar=False)
    U, S, _ = np.linalg.svd(Sigma_Z)
    W = U @ np.diag(1 / np.sqrt(S)) @ U.T
    Z_whitened = Z @ W.T
    return Z_whitened


# Function to compute kernel density estimation
def kde(data, points, bandwidth):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data[:, None])
    log_density = kde.score_samples(points[:, None])
    density = np.exp(log_density)
    return density


# Function to estimate densities and mutual information
def estimate_mutual_information(Z, h, h_marg, N):
    f_XY = np.zeros(N)
    f_X = np.zeros(N)
    f_Y = np.zeros(N)

    for i in range(N):
        mu = Z[i, :]
        f_X[i] = kde(Z[:, 0], np.array([mu[0]]), h_marg)
        f_Y[i] = kde(Z[:, 1], np.array([mu[1]]), h_marg)
        f_XY[i] = np.mean(multivariate_normal.pdf(Z, mean=mu, cov=(h ** 2) * np.eye(2)))

    MI = np.abs(np.mean(np.log(f_XY / (f_X * f_Y))))
    return MI


# Function to perform permutation test
def permutation_test(Z, X, Y, h, h_marg, N, P):
    T_PERM = np.zeros(P)

    for p in range(P):
        Y_permuted = np.random.permutation(Y)
        Z_permuted = np.hstack((X, Y_permuted))
        MI_PERM = estimate_mutual_information(Z_permuted, h, h_marg, N)
        T_PERM[p] = np.sqrt(1 - np.exp(-2 * MI_PERM))

    return T_PERM


# GLRT function for independence testing
def glrt_independence(X, Y):
    N = len(X)
    p = X.shape[1]
    q = Y.shape[1]

    cov_X = np.cov(X, rowvar=False)
    cov_Y = np.cov(Y, rowvar=False)
    cov_XY = np.cov(np.hstack((X, Y)), rowvar=False)

    log_det_cov_X = np.log(get_det(cov_X))
    log_det_cov_Y = np.log(get_det(cov_Y))
    log_det_cov_XY = np.log(get_det(cov_XY))

    glrt_statistic = (N - (p + q + 3) / 2) * (log_det_cov_X + log_det_cov_Y - log_det_cov_XY)
    return glrt_statistic


# Function to calculate p-value for GLRT statistic
def calculate_glrt_p_value(glrt_statistic, p, q):
    df = p * q  # Degrees of freedom for the chi-squared distribution
    p_value = 1 - chi2.cdf(glrt_statistic, df)
    return p_value


# Main function to run the analysis
def main():
    N = 1000
    P = 1000
    Z = generate_data(N)
    X, Y = Z[:, 0].reshape(-1, 1), Z[:, 1].reshape(-1, 1)
    Z_whitened = whiten_data(Z)

    h = N ** (-1 / 6)
    h_marg = N ** (-1 / 5)

    MI = estimate_mutual_information(Z_whitened, h, h_marg, N)
    T = np.sqrt(1 - np.exp(-2 * MI))

    T_PERM = permutation_test(Z_whitened, X, Y, h, h_marg, N, P)
    P_VAL = np.sum(T_PERM > T) / P

    # GLRT for independence testing
    glrt_statistic = glrt_independence(X, Y)
    glrt_p_value = calculate_glrt_p_value(glrt_statistic, X.shape[1], Y.shape[1])

    print(f'Mutual Information: {MI}')
    print(f'T-statistic: {T}')
    print(f'Permutation Test P-value: {P_VAL}')
    print(f'GLRT Statistic: {glrt_statistic}')
    print(f'GLRT P-value: {glrt_p_value}')


if __name__ == "__main__":
    main()
