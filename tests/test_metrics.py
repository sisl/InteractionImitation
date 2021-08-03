import torch
import numpy as np

from src import metrics
from src.metrics import divergence, evaluate_histogram
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
    d4 = divergence(p, q, type='kl', n_components=-1)
    assert isinstance(d4, float)
    d5 = divergence(q, p, type='kl', n_components=-1)
    assert isinstance(d5, float)

    p = 1000 * torch.randn(1000)
    q = torch.randn(1000)
    d6 = divergence(p, q, type='kl', n_components=-1)
    assert ~np.isfinite(d6)

def test_evaluate_histogram():
    N = 10000
    p = torch.randn(N)
    q = 1.0 + 2.0 * torch.randn(2*N)
    
    p_hist, p_edges = np.histogram(p.unsqueeze(-1), bins='auto', density=True)
    q_hist, q_edges = np.histogram(q.unsqueeze(-1), bins='auto', density=True)
    px = evaluate_histogram(p, p_hist, p_edges)
    assert px.shape == p.shape
    qx = evaluate_histogram(q, p_hist, p_edges)
    assert qx.shape == q.shape
    px = evaluate_histogram(p, q_hist, q_edges)
    assert px.shape == p.shape
    qx = evaluate_histogram(q, q_hist, q_edges)
    assert qx.shape == q.shape


def test_js_divergence():
    N = 1000
    p = torch.randn(N)
    q = 1.0 + 2.0 * torch.randn(2*N)

    d1 = divergence(p, q, type='js')
    d2 = divergence(q, p, type='js')
    assert d1 == d2

    p = 1000 * torch.randn(1000)
    q = torch.randn(1000)
    d1 = divergence(p, q, type='js')
    d2 = divergence(q, p, type='js')
    assert d1 == d2
    assert np.isfinite(d1)


if __name__ == '__main__':
    p = torch.randn(10)
    q = 1.0 + 2.0 * torch.randn(10)

    p = p.unsqueeze(-1)
    q = q.unsqueeze(-1)
    p_hist, p_edges = np.histogram(p.unsqueeze(-1), bins='auto', density=True)
    q_hist, q_edges = np.histogram(q.unsqueeze(-1), bins='auto', density=True)
    # px = p_hist[np.digitize(p, p_edges) - 1]

    # qx = q_hist[np.digitize(p, q_edges) - 1]
    px = evaluate_histogram(p, p_hist, p_edges)
    qx = evaluate_histogram(q, p_hist, p_edges)
