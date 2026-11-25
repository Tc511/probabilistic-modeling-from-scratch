"""公共数据生成与绘图工具。概率模型和训练逻辑保留在各章 Notebook。"""

import numpy as np
import matplotlib.pyplot as plt


def generate_observations(rng, n=900):
    """生成第 01–05 章共享的二维观测数据；不返回真实类别标签。"""
    centers = np.array([[-2., 0.], [2., 0.], [0., 3.]])
    classes = rng.integers(0, 3, n)
    noise = rng.normal(size=(n, 2))
    y = np.zeros((n, 2))
    for i in range(n):
        y[i] = centers[classes[i]] + 0.45 * noise[i]
    return y


def plot_observations(y):
    """绘制原始二维观测。"""
    plt.scatter(y[:, 0], y[:, 1], s=5)
    plt.axis('equal')
    plt.show()


def compare(y, samples):
    """使用相同坐标范围比较观测数据与模型生成数据。"""
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax[0].scatter(y[:, 0], y[:, 1], s=5, alpha=.5)
    ax[0].set_title('Observed')
    ax[1].scatter(samples[:, 0], samples[:, 1], s=5, alpha=.5)
    ax[1].set_title('Generated')
    for a in ax:
        a.set_aspect('equal')
    plt.show()


def plot_history(history, ylabel='mean log likelihood'):
    """绘制训练目标随迭代次数的变化。"""
    plt.plot(history)
    plt.xlabel('iteration')
    plt.ylabel(ylabel)
    plt.show()


def plot_likelihood(candidates, likelihood, reference, reference_label):
    """绘制一维参数对应的序列对数似然。"""
    plt.plot(candidates, likelihood)
    plt.axvline(reference, ls='--', label=reference_label)
    plt.legend()
    plt.xlabel('a')
    plt.ylabel('sequence log likelihood')
    plt.show()
