# %%

import numpy as np
import torch
from timeit import default_timer as timer

# %%

def powerseries(x, deg):
    return torch.stack([x**i for i in range(deg+1)],dim=-1)

def improved_powerseries(x, deg):
    r = torch.ones(*x.shape, deg+1, dtype=torch.float64)
    for i in range(1,deg+1):
        r[:, :, i] = r[:, :, i-1] * x
    return r

def horner_scheme(x, poly):
    deg = poly.shape[-1]
    nsteps = x.shape[-1]
    r = poly[:, -1:].repeat(1, nsteps)
    for i in range(2, deg+1):
        r *= x
        r += poly[:, -i:1-i]
    return r

# %%

nv = 151
delta = 10
n = 20

state_s = torch.rand((nv, 1))
nan_idx = np.random.choice([True, False], 151)
state_s[nan_idx] = np.nan

# %%

n_coef = 21
xpoly = torch.rand((nv, n_coef),dtype=torch.float64)
ypoly = torch.rand((nv, n_coef),dtype=torch.float64)

ds = delta * torch.arange(1,n+1).repeat(nv,1)

s = ds + state_s
s = s.type(torch.float64)

smax = s[:, 0]
smax = smax.unsqueeze(-1)


start = timer()
for _ in range(100):
    deg = xpoly.shape[-1] - 1
    expand_sims = powerseries(s, deg) # (nv, n, deg+1)
    # print(expand_sims.shape)
    y = (ypoly.unsqueeze(1) * expand_sims).sum(dim=-1)
    x = (xpoly.unsqueeze(1) * expand_sims).sum(dim=-1)
end = timer()
print("Powerseries: {}".format((end-start)*1))

start = timer()
for _ in range(100):
    deg = xpoly.shape[-1] - 1
    expand_sims = improved_powerseries(s, deg) # (nv, n, deg+1)
    # print(expand_sims.shape)
    yp = (ypoly.unsqueeze(1) * expand_sims).sum(dim=-1)
    xp = (xpoly.unsqueeze(1) * expand_sims).sum(dim=-1)
end = timer()
print("Improved Powerseries: {}".format((end-start)*1))

start = timer()
for _ in range(100):
    x_horner = horner_scheme(s, xpoly)
    y_horner = horner_scheme(s, ypoly)
end = timer()
print("Horner: {}".format((end-start)*1))

start = timer()
for _ in range(100):
    x_max = horner_scheme(smax, xpoly)
    y_max = horner_scheme(smax, ypoly)
end = timer()
# print("Horner smax: {}".format((end-start)*1))

assert np.all(np.isclose(xp,x)[~nan_idx])
assert np.all(np.isclose(yp,y)[~nan_idx])
assert np.all(np.isclose(x_horner,x)[~nan_idx])
assert np.all(np.isclose(y_horner,y)[~nan_idx])
# %%
