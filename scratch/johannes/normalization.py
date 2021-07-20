import torch
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

loader = DataLoader(train_set, batch_size=len(train_set), num_workers=1)
# load whole dataset
input_data, out_data = next(iter(loader))
out_data = out_data.float()
# compute mean and std only over batch dimension
m_in, s_in = input_data.mean(dim=0), input_data.std(dim=0)
m_out, s_out = out_data.mean(dim=0), out_data.std(dim=0)

input_tf = transforms.Normalize(m_in, s_in)
out_tf = transforms.Normalize(m_out, s_out)

transformed_input = input_tf(input_data)
transformed_output = torch.sigmoid(out_tf(out_data))

# scale sigmoid output [0, 1] to acceleration interval [a_min, a_max]
a_min, a_max = (-4, 2)
# compute m and s such that normalization with m and s results in desired scaling
s = 1 / (a_max - a_min)
m = - s * a_min
scaling = transforms.Normalize(m, s)

# DOES NOT WORK SINCE TORCHVISION NORMALIZE WORKS ONLY ON IMAGES
scaled_output = scaling(transformed_output)
