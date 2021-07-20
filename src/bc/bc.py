import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from src.policies import 

class BehaviorCloningPolicy():
    """
    Class for (continuous) behavior cloning policy
    """
    
    def __init__(self, transforms={}, **kwargs):
        self._transforms = transforms
        self._policy_model = PolicyModel(**kwargs)

    @property
    def transforms(self):
        return self._transforms
    
    @transforms.setter
    def transforms(self, transforms)
        self._transforms=transforms

    def __call__(self, ob):

        if 'actions' in ob.keys():
            # extract state from dataloader samples  
            pass
        else:
            # extract state from observation (using simulator)
            ob['path_x'] = ob['paths'][0]
            ob['path_y'] = ob['paths'][1]

        # run observation through transforms
        for key in ['state', 'relative_state', 'path_x', 'path_y']:             
            if key in self._transforms.keys():
                ob[key] = self._transforms[key](ob[key])

        # run transformed state through model

        # untransform action

        pass

    @classmethod
    def load_model(cls, filestr, **kwargs):
        """
        Load a model from a file prefix
        Args:
            filestr (str): string prefix to load model from
        Returns
            model (BehaviorCloningPolicy): loaded model
        """
        transforms = pickle.load(open(filestr+'_transforms.pkl', 'rb'))
        model = cls(transforms=transforms, **kwargs)
        model._policy_model.load_state_dict(torch.load(filestr+'_model.pt'))
        return model
    
    def eval(self):
        self._policy_model.eval()

    def save_model(self, filestr):
        """
        Save transforms and state_dict to a location specificed by filestr
        Args:
            filestr (str): string prefix to save model to
        """
        pickle.dump(self._transforms, open(filestr+'_transforms.pkl', 'wb'))
        torch.save(self._policy_model.state_dict(), filestr+'_model.pt')

def generate_transforms(dataset):
    """
    Generate transform dictionary from dataset
    """
    pass

def train(train_dataset, cv_dataset, policy_class, filestr=filestr, **kwargs):
    
    # hyperparams
    train_epochs = 10000
    cv_every = 100
    train_batch_size = 64
    cv_batch_size = 256

    # generate transform from train_dataset
    transforms = generate_transforms(train_dataset)

    # initialize policy
    policy = BehaviorCloningPolicy(transforms=transforms, **kwargs)

    # training and testing dataloaders
    training_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    cv_loader = DataLoader(cv_dataset, batch_size=cv_batch_size, shuffle=True)

    # generate loss function, optimizer

    for i in train_epochs:

        # sample mini-batch and run through policy
        pred_action = policy(batch) 

        # compute loss and step optimizer
        loss.backwards()
        optimizer.step()


        # measure L2 on every cv epoch
        if i % cv_every == 0:
            pass

    policy.save_model(filestr)
