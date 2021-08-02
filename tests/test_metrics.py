import torch
import numpy as np

from src import metrics
from src.metrics import divergence

def test_kl_divergence():
    p = torch.randn(1000000)
    q = 1.0 + 2.0 * torch.randn(1000000)

    d1 = divergence(p, q, type='kl', n_components=0)
    assert isinstance(d1, float)
    d2 = divergence(p, q, type='kl', n_components=1)
    assert isinstance(d2, float)
    assert np.isclose(d1, d2, atol=1e-5)
    d3 = divergence(p, q, type='kl', n_components=3)
    assert isinstance(d3, float)

