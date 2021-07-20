import torch
from torch import nn

from sklearn import preprocessing

class Normalization(nn.Module):
    def __init__(self, X):
        super(Normalization, self).__init__()
        self.fit(X)

    def fit(self, X):
        raise NotImplementedError('Please implement fit()')

    def transform(self, X):
        raise NotImplementedError('Please implement transform()')

    def inverse_transform(self, X):
        raise NotImplementedError('Please implement inverse_transform()')

    def forward(self, X):
        return self.transform(X)

class SciKitNormalization(Normalization):
    def __init__(self, tf, X):
        self.tf = tf
        super(SciKitNormalization, self).__init__(X)

    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        return torch.tensor(self.tf.transform(X), dtype=torch.float)

    def inverse_transform(self, X):
        return torch.tensor(self.tf.inverse_transform(X), dtype=torch.float)

class SciKitStandardization(SciKitNormalization):
    def __init__(self, X):
        super(SciKitStandardization, self).__init__(preprocessing.StandardScaler(), X)

class SciKitMinMaxScaler(SciKitNormalization):
    def __init__(self, X):
        super(SciKitMinMaxScaler, self).__init__(preprocessing.MinMaxScaler(), X)


ns = 5
na = 1
n_batch = 1000

state = torch.rand(n_batch, ns)
action = torch.rand(n_batch, na)

s_tf = SciKitStandardization(state)
a_tf = SciKitMinMaxScaler(action)

print(torch.linalg.norm(s_tf.inverse_transform(s_tf(state)) - state))
print(torch.linalg.norm(a_tf.inverse_transform(a_tf(action)) - action))




# class Foo:
#     def __init__(self):
#         return None
#     def baz(self):
#         print("Foo.baz()")

# class Bar(Foo):
#     def __init__(self):
#         return None

# bar = Bar()
# bar.baz()