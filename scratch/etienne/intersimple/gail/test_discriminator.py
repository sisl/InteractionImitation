from intersim.envs.intersimple import NRasterized
from discriminator import CnnDiscriminator
import torch

def test_image_concatenation():
    env = NRasterized()
    disc = CnnDiscriminator(env)
    s = torch.tensor(env.reset()).unsqueeze(0)
    a = torch.tensor([[0.5]])
    sa = disc._concatenate(s, a)

    assert sa.shape == (1, 6, 200, 200)
    assert torch.allclose(sa[:, :5], 1.0 * s)
    assert (sa[:, 5] == a).all()
