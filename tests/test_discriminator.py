from intersim.envs.intersimple import NRasterized
from src.discriminator import CnnDiscriminator
import torch

def test_image_concatenation():
    env = NRasterized()
    disc = CnnDiscriminator(env)
    s = torch.tensor(env.reset()).unsqueeze(0)
    a = torch.tensor([[0.5]])
    sa = disc._concatenate(s, a)

    assert s.shape == (1, 5, 200, 200)
    assert a.shape == (1, 1)
    assert sa.shape == (1, 6, 200, 200)
    assert torch.allclose(sa[:, :5], 1.0 * s)
    assert (sa[:, 5] == a.unsqueeze(-1)).all()

def test_image_concatenation3():
    env = NRasterized()
    disc = CnnDiscriminator(env)

    s1 = env.reset()
    a1 = 0.15
    s2, _, _, _ = env.step(0.9)
    a2 = 0.25
    s3, _, _, _ = env.step(-0.9)
    a3 = 0.35
    
    s = torch.stack([
        torch.tensor(s1),
        torch.tensor(s2),
        torch.tensor(s3)
    ], axis=0)
    a = torch.tensor([
        [a1],
        [a2],
        [a3],
    ])
    sa = disc._concatenate(s, a)

    assert s.shape == (3, 5, 200, 200)
    assert a.shape == (3, 1)
    assert sa.shape == (3, 6, 200, 200)
    assert torch.allclose(sa[:, :5], 1.0 * s)
    assert (sa[:, 5] == a.unsqueeze(-1)).all()
