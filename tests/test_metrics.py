import torch
import numpy as np

from src import metrics
from src.metrics import divergence
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def test_kl_divergence():
    p = torch.randn(1000000)
    q = 1.0 + 2.0 * torch.randn(1000000)

    d1 = divergence(p, q, type='kl', n_components=0)
    print(d1)
    assert isinstance(d1, float)
    d2 = divergence(p, q, type='kl', n_components=1)
    print(d2)
    assert isinstance(d2, float)
    assert np.isclose(d1, d2, atol=1e-5)
    d3 = divergence(p, q, type='kl', n_components=3)
    print(d3)
    assert isinstance(d3, float)


def evaluate_histogram(x, hist, bin_edges):
    idx = np.digitize(x, bin_edges)
    mask = np.logical_and(np.less(0, idx), np.less(idx, len(bin_edges)))
    r = np.zeros_like(x)
    r[mask] = hist[idx[mask] - 1]


if __name__ == '__main__':
    # p = torch.randn(10)
    # q = 1.0 + 2.0 * torch.randn(10)

    # p = p.unsqueeze(-1)
    # q = q.unsqueeze(-1)
    # p_hist, p_edges = np.histogram(p.unsqueeze(-1), bins='auto', density=True)
    # q_hist, q_edges = np.histogram(q.unsqueeze(-1), bins='auto', density=True)
    # # px = p_hist[np.digitize(p, p_edges) - 1]

    # # qx = q_hist[np.digitize(p, q_edges) - 1]
    # px = evaluate_histogram(p, p_hist, p_edges)
    # qx = evaluate_histogram(q, p_hist, p_edges)

    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-1, 1, 3)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(p)
    p_kde = grid.best_estimator_
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(q)
    q_kde = grid.best_estimator_
    px = p_kde.score_samples(p)
    qx = q_kde.score_samples(p)
    d = np.mean(px - qx).item()
    print(d)
    test_kl_divergence()