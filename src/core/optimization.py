import torch

def conjugate_gradient(A, b, max_iters, res_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - A(x)
    p = r

    rTr = r.T @ r

    for _ in range(max_iters):
        Ap = A(p)
        alpha = rTr / (p.T @ Ap)
        x = x + alpha * p

        r = r - alpha * Ap
        if torch.norm(r) < res_tol:
            break

        rTrnew = r.T @ r
        beta = rTrnew / rTr
        p = r + beta * p
        rTr = rTrnew

    return x

def line_search(f, x0, dx, g0, alpha, condition, max_steps=10, c1=0.1):
    assert 0 < alpha < 1
    
    f0 = f(x0)
    for _ in range(max_steps):
        x = x0 + dx

        if (f(x) > f0 + c1 * g0.T @ dx) and condition(x):
            return x
        
        dx *= alpha
    
    print('Line search failed, returning x0')
    return x0
