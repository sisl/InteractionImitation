import torch
from torch import nn
import numpy as np
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

class MinMaxScaler(Transform):
    """
    Scale tensor so each feature is in [0, 1]
    """
    def __init__(self, reduce_dim:int=None):
        """
        Initialize SciKitTransform
        Args:
            reduce_dim (int): dimension to start calculating featues from
                e.g. with reduce_dim=2, (A, B, C, D, E) will be reshaped to (A*B, C*D*E)
        """
        self.reduce_dim = reduce_dim
        super(MinMaxScaler, self).__init__()

    def fit(self, X):
        nd = X.ndim
        if self.reduce_dim:
            self.nfeatures = int(torch.tensor(X.shape[self.reduce_dim:]).prod())
        else:
            assert nd==2, 'Invalid ndim'
            self.nfeatures = X.shape[1]
        
        X = X.reshape((-1,self.nfeatures))
        nans = torch.isnan(X)
        X[nans] = float('inf')
        self.min = X.min(0,keepdims=True)[0]

        X[nans] = -float('inf')
        self.span = X.max(0,keepdims=True)[0] - self.min

        X[nans] = np.nan

    def transform(self, X):
        
        assert hasattr(self, 'min') and hasattr(self, 'span'), 'Model not yet fit'
        shape = X.shape
        X = X.reshape((-1,self.nfeatures))
        t = (X - self.min) / self.span
        return t.reshape(shape)

    def inverse_transform(self, X):

        assert hasattr(self, 'min') and hasattr(self, 'span'), 'Model not yet fit'
        shape = X.shape
        X = X.reshape((-1,self.nfeatures))
        it =  X * self.span + self.min
        return it.reshape(shape)


class SciKitTransform(Transform):
    """
    Wrappers around scikit-learn transforms
    """
    def __init__(self, tf, reduce_dim:int=None):
        """
        Initialize SciKitTransform
        Args:
            tf: transform
            reduce_dim (int): dimension to start calculating featues from
                e.g. with reduce_dim=2, (A, B, C, D, E) will be reshaped to (A*B, C*D*E)
        """
        self.tf = tf
        self.reduce_dim = reduce_dim
        super(SciKitTransform, self).__init__()

    def fit(self, X):
        nd = X.ndim
        if self.reduce_dim:
            self.nfeatures = int(torch.tensor(X.shape[self.reduce_dim:]).prod())
        else:
            assert nd==2, 'Invalid ndim'
            self.nfeatures = X.shape[1]

        self.tf.fit(X.reshape((-1,self.nfeatures)))

    def transform(self, X):
        shape = X.shape
        t = torch.tensor(self.tf.transform(X.reshape((-1,self.nfeatures))), dtype=torch.float)
        return t.reshape(shape)

    def inverse_transform(self, X):
        shape = X.shape
        it =  torch.tensor(self.tf.inverse_transform(X.reshape((-1,self.nfeatures))), dtype=torch.float)
        return it.reshape(shape)

class SciKitStandardScaler(SciKitTransform):
    """
    Wrapper around scikit-learn's StandardScaler for standardizing each feature individually.
    """
    def __init__(self, **kwargs):
        super(SciKitStandardScaler, self).__init__(preprocessing.StandardScaler(), **kwargs)

class SciKitMinMaxScaler(SciKitTransform):
    """
    Wrapper around scikit-learn's MinMaxScaler for scaling features to [0, 1] individually.
    """
    def __init__(self, **kwargs):
        super(SciKitMinMaxScaler, self).__init__(preprocessing.MinMaxScaler(), **kwargs)

