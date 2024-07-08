import numpy as np
import matplotlib.pyplot as plt


def calc_rmse(p, N, h):
    res1 = (N - 1) / (N * ((1 + h ** 2) ** p))
    res3 = 1 / (N * ((h ** 2)*(2 + h ** 2)) ** (p / 2))
    res4 = -2 * ((1 + h ** 2) ** (-p / 2))
    res = res1 + 1 + res3 + res4
    if (res < 0).any():
        print(1)
    return res


def find_opt_rmse(p, N):
    h = np.linspace(0.01, 10, 300)[:, None]
    rmse_list = []
    for n in N:
        rmse_tmp = calc_rmse(p, n, h)
        rmse = np.min(rmse_tmp, axis=0)
        rmse_list.append(float(rmse))
    return np.array(rmse_list)

def run():
    tol = 0
    N_list = np.pow(2, np.linspace(1, 50, 50))[:, None]
    p_list = np.linspace(1, 20, 20)
    res_N = []
    for p in p_list:
        rmse = find_opt_rmse(p, N_list)
        idx_cond = int(np.argwhere(rmse <= (0.1 + tol))[0].squeeze())
        res_N.append(N_list[idx_cond])
    res_N = np.array(res_N)
    plt.semilogy(p_list, res_N)
    plt.title("Minimal Relative MSE")
    plt.xlabel("Dimension")
    plt.ylabel("Number of samples")
    plt.xticks(p_list[::2])
    plt.grid()
    plt.show()




if __name__ == '__main__':
    run()
