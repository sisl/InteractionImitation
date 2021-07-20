import torch
from torch import nn

from sklearn import preprocessing

class Transform(nn.Module):
    """
    Base class to normalize observations and actions for network.
    """

    def __init__(self):
        super(Transform, self).__init__()
        # self.fit(X)

    def fit(self, X):
        """
        Fit transformer to X
        Args:
            X (torch.tensor): (B, N) tensor of B data points with N features
        """
        raise NotImplementedError('Please implement fit()')

    def transform(self, X):
        """
        Transform X. fit() has to be called first
        Args:
            X (torch.tensor): (B, N) tensor where N has to be the same as during fit()
        """
        raise NotImplementedError('Please implement transform()')

    def inverse_transform(self, X):
        """
        Inverse transformation
        Args:
            X (torch.tensor): (B, N) tensor
        """
        raise NotImplementedError('Please implement inverse_transform()')

    def forward(self, X):
        return self.transform(X)

class SciKitTransform(Transform):
    """
    Wrappers around scikit-learn transforms
    """
    def __init__(self, tf):
        self.tf = tf
        super(SciKitTransform, self).__init__()

    def fit(self, X):
        self.tf.fit(X)

    def transform(self, X):
        return torch.tensor(self.tf.transform(X), dtype=torch.float)

    def inverse_transform(self, X):
        return torch.tensor(self.tf.inverse_transform(X), dtype=torch.float)

class SciKitStandardScaler(SciKitNormalization):
    """
    Wrapper around scikit-learn's StandardScaler for standardizing each feature individually.
    """
    def __init__(self):
        super(SciKitStandardScaler, self).__init__(preprocessing.StandardScaler())

class SciKitMinMaxScaler(SciKitNormalization):
    """
    Wrapper around scikit-learn's MinMaxScaler for scaling features to [0, 1] individually.
    """
    def __init__(self):
        super(SciKitMinMaxScaler, self).__init__(preprocessing.MinMaxScaler())

