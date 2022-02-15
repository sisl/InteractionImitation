import torch
from optimization import conjugate_gradient

def test_cg_eye():
    A = torch.eye(2)
    b = torch.tensor([1., 2.])
    x1 = conjugate_gradient(lambda x: A @ x, b, 2)
    x2 = torch.inverse(A) @ b
    assert torch.allclose(x1, x2)

def test_cg_eyep1():
    A = torch.eye(2) + 1
    b = torch.tensor([1., 2.])
    x1 = conjugate_gradient(lambda x: A @ x, b, 2)
    x2 = torch.inverse(A) @ b
    assert torch.allclose(x1, x2, atol=1e-7)

def test_cg3():
    A = torch.tensor([[4., 2.], [2., 4.]])
    b = torch.tensor([2., 1.])
    x1 = conjugate_gradient(lambda x: A @ x, b, 100)
    x2 = torch.inverse(A) @ b
    assert torch.allclose(x1, x2)
