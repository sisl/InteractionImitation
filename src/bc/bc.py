import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import pickle
#from torch.utils.tensorboard import SummaryWriter

from src.policies import DeepSetsPolicy
from src.util.transform import SciKitMinMaxScaler
import json5

class BehaviorCloningPolicy():
    """
    Class for (continuous) behavior cloning policy
    """
    
    def __init__(self, config: dict, transforms: dict={}):
        """
        Initialize BehaviorCloningPolicy
        Args:
            config (dict): configuration file to initialize DeepSetsPolicy with
            transforms (dict): dictionary of transforms to apply to different fields
        """
        self._config = config
        self._transforms = transforms
        self._policy_model = DeepSetsPolicy(config["ego_state"], config["deepsets"], config["path_encoder"], config["head"])
        
    @property
    def transforms(self):
        return self._transforms
    
    @transforms.setter
    def transforms(self, transforms):
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
                ob[key] = self._transforms[key].transform(ob[key])

        # run transformed state through model
        action = self._policy_model(ob)
        assert action.ndim == 2, 'action has incorrect shape'

        # untransform action
        if 'action' in self._transforms.keys():
            action = self._transforms['action'].inverse_transform(action)
        return action

    @classmethod
    def load_model(cls, config: dict, filestr: str):
        """
        Load a model from a file prefix
        Args:
            config (dict): configuration dict to set up model
            filestr (str): string prefix to load model from
        Returns
            model (BehaviorCloningPolicy): loaded model
        """
        transforms = pickle.load(open(filestr+'_transforms.pkl', 'rb'))
        model = cls(config, transforms=transforms)
        model._policy_model.load_state_dict(torch.load(filestr+'_model.pt'))
        return model
    
    def eval(self):
        self._policy_model.eval()

    def parameters(self):
        return self._policy_model.parameters()

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
    Args:
        dataset (Dataset): dataset of demo observations and actions
    """
    transforms = {
        'action': SciKitMinMaxScaler(),
        'state': SciKitMinMaxScaler(),
        'relative_state': SciKitMinMaxScaler(reduce_dim=2),
        'path_x': SciKitMinMaxScaler(reduce_dim=2),
        'path_y': SciKitMinMaxScaler(reduce_dim=2),
    }
    for key in transforms.keys():
        transforms[key].fit(dataset[:][key])

    return transforms

def train(train_dataset, cv_dataset, policy, filestr, **kwargs):
    
    # hyperparams
    train_epochs = 10000
    cv_every = 100
    train_batch_size = 64
    cv_batch_size = 256 # doesn't matter
    learning_rate = 1e-3
    weight_decay=0.1

    # generate transform from train_dataset
    transforms = generate_transforms(train_dataset)

    # initialize policy
    policy.transforms = transforms

    # training and testing dataloaders
    training_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    cv_loader = DataLoader(cv_dataset, batch_size=cv_batch_size, shuffle=True)

    # generate loss function, optimizer
    loss_fn = nn.HuberLoss(reduction='sum')
    import pdb
    pdb.set_trace()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for i in train_epochs:

        epoch_loss = 0
        for (batch_idx, batch) in enumerate(training_loader):
            # sample mini-batch and run through policy
            pred_action = policy(batch) 
            loss = loss_fn(pred_action, batch['action'])

            # compute loss and step optimizer
            optimizer.zero_grad()
            loss.backwards()
            optimizer.step()

            epoch_loss += loss.item() / len(train_dataset)
        
        # Write epoch loss
        if i % 10 == 0:
            print('Epoch: {}, Training Loss: {}'.format(i, epoch_loss))

        # measure cv loss
        if i % cv_every == 0:
            with torch.no_grad():
                cv_loss = 0.
                for (batch_idx, batch) in enumerate(cv_loader):
                    pred_action = policy(batch) 
                    loss = loss_fn(pred_action, batch['action'])
                    cv_loss += loss.item() / len(cv_dataset)
            print('Epoch: {}, CV Loss: {}'.format(i, cv_loss))

    policy.save_model(filestr)
