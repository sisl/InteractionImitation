import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from torch.utils.tensorboard import SummaryWriter

from src.policies import DeepSetsPolicy
from src.util.transform import MinMaxScaler
from tqdm import tqdm
import json5
from ray import tune

def bc_config(ray_config):
    config = {
        'ego_encoder': {'input_dim': 5, 'hidden_n': 0, 'hidden_dim':0, 'output_dim': 0},
        'deepsets': {
            'input_dim': 6,
            'phi': {
                'hidden_n': ray_config['deepsets_phi_hidden_n'],
                'hidden_dim': ray_config['deepsets_phi_hidden_dim']
                },
            'latent_dim': ray_config['deepsets_latent_dim'],
            'rho': {
                'hidden_n': ray_config['deepsets_rho_hidden_n'],
                'hidden_dim': ray_config['deepsets_rho_hidden_dim']
                },
            'output_dim': ray_config['deepsets_output_dim']
        },
        'path_encoder': {'input_dim': 40, 'hidden_n': 0, 'hidden_dim': 0, 'output_dim': 0},
        'head': {
            'input_dim': 0, # computed in constructor
            'hidden_n': ray_config['head_hidden_n'],
            'hidden_dim': ray_config['head_hidden_dim'],
            'output_dim': 1, # number of outputs e.g. number of actions, or just one
            'final_activation': ray_config['head_final_activation'],
        },
        'optim': {
            'optimizer':'adam',
            'lr':ray_config['lr'],
            'weight_decay':ray_config['weight_decay']
        },
        'train_epochs': 40,
        'train_batch_size': ray_config['train_batch_size'],
        'loss': ray_config['loss'],

    }
    return config

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
        self._policy = DeepSetsPolicy(config)
        
    @property
    def transforms(self):
        return self._transforms
    
    @transforms.setter
    def transforms(self, transforms):
        self._transforms=transforms

    @property 
    def policy(self):
        return self._policy
    
    @policy.setter
    def policy(self, policy):
        self._policy = policy

    def __call__(self, ob):

        if 'ego_state' in ob.keys():
            # extract state from dataloader samples  
            pass
        else:
            # extract state from observation (using simulator)
            ob['ego_state'] = ob['state']
            ob['path'] = torch.stack(ob['paths'],dim=-1)

        # run observation through transforms
        for key in ['ego_state', 'relative_state', 'path']:             
            if key in self._transforms.keys():
                ob[key] = self._transforms[key].transform(ob[key])

        # run transformed state through model
        action = self._policy(ob)
        assert action.ndim == 2, 'action has incorrect shape'

        # untransform action
        if 'action' in self._transforms.keys():
            action = self._transforms['action'].inverse_transform(action)
        return action

    @classmethod
    def load_model(cls, filestr: str, config: dict = None):
        """
        Load a model from a file prefix
        Args:
            config (dict): configuration dict to set up model
            filestr (str): string prefix to load model from
        Returns
            model (BehaviorCloningPolicy): loaded model
        """
        if not config:
            with open(filestr+'_config.json', 'r') as cfg:
                config = json5.load(cfg)
        transforms = pickle.load(open(filestr+'_transforms.pkl', 'rb'))
        model = cls(config, transforms=transforms)
        model._policy.load_state_dict(torch.load(filestr+'_model.pt'))
        return model
    
    def eval(self):
        self._policy.eval()

    def parameters(self):
        return self._policy.parameters()

    def save_model(self, filestr, save_config=True, save_transforms=True):
        """
        Save transforms and state_dict to a location specificed by filestr
        Args:
            filestr (str): string prefix to save model to
            save_config (bool): whether to save the config file (as a json)
            save_transforms (bool): whether to save transforms (as a pickle)
        """
        if save_config:
            with open(filestr+'_config.json', 'w') as cfg:
                json5.dump(self._config, cfg)
        if save_transforms:
            pickle.dump(self._transforms, open(filestr+'_transforms.pkl', 'wb'))
        torch.save(self._policy.state_dict(), filestr+'_model.pt')

def generate_transforms(dataset):
    """
    Generate transform dictionary from dataset
    Args:
        dataset (Dataset): dataset of demo observations and actions
    """
    transforms = {
        'action': MinMaxScaler(),
        'ego_state': MinMaxScaler(),
        'relative_state': MinMaxScaler(reduce_dim=2),
        'path': MinMaxScaler(reduce_dim=2),
    }
    for key in transforms.keys():
        if key == 'action':
            transforms[key].fit(dataset[:][key])
        else:
            transforms[key].fit(dataset[:]['state'][key])

    return transforms

def train(config, policy, train_dataset, cv_dataset, filestr, **kwargs):
    
    using_ray = kwargs.get('ray', False)
    if using_ray:
        print('using ray')

    # hyperparams
    loss_type = config['loss']
    train_epochs = config['train_epochs']
    train_batch_size = config['train_batch_size']
    optimizer_type = config['optim']['optimizer']
    learning_rate = config['optim']['lr']
    weight_decay = config['optim']['weight_decay']

    cv_every = 1
    print_epoch_every = 1000
    print_cv_every = 5
    checkpoint_every = 100
    cv_batch_size = 256 # doesn't matter

    # generate transform from train_dataset
    transforms = generate_transforms(train_dataset)

    # initialize policy
    policy.transforms = transforms

    # training and testing dataloaders
    training_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    cv_loader = DataLoader(cv_dataset, batch_size=cv_batch_size, shuffle=True)

    # change policy dtype
    policy.policy = policy.policy.type(train_dataset[0]['state']['ego_state'].dtype)

    # generate loss function, optimizer
    cv_loss_fn = nn.MSELoss(reduction='sum')
    if loss_type == 'huber':
        loss_fn = nn.HuberLoss(reduction='sum') 
    elif loss_type == 'mse':
        loss_fn = nn.MSELoss(reduction='sum')
    else:
        raise NotImplementedError
    if optimizer_type == 'adam':  
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError

    # generate tensorboard writer
    if not using_ray:
        writer = SummaryWriter(filestr)
    
    for i in tqdm(range(train_epochs)):
        
        # save model checkpoints
        if i % checkpoint_every == 0:
            policy.save_model(filestr + '_epoch%04i'%(i) )

        # train
        epoch_loss = 0
        for (batch_idx, batch) in enumerate(training_loader):
            
            # sample mini-batch and run through policy
            pred_action = policy(batch['state']) 
            loss = loss_fn(pred_action, batch['action'])

            # compute loss and step optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / len(train_dataset)
        
        # if i % print_epoch_every == 0:
        #     print('Epoch: {}, Training Loss: {}'.format(i, epoch_loss))

                # measure cv loss

        if i % cv_every == 0:
            with torch.no_grad():
                cv_loss = 0.
                for (batch_idx, batch) in enumerate(cv_loader):
                    pred_action = policy(batch['state']) 
                    loss = cv_loss_fn(pred_action, batch['action'])
                    cv_loss += loss.item() / len(cv_dataset)
            
            
        # Write epoch loss
        if using_ray:
            if i % cv_every == 0:
                tune.report(training_loss=epoch_loss, cv_loss=cv_loss, training_iteration=i+1)
            else:
                tune.report(training_loss=epoch_loss, training_iteration=i+1)
        else:
            writer.add_scalar('training loss',epoch_loss, i)
            if i % cv_every == 0:
                writer.add_scalar('cv loss', cv_loss, i)

        # if i % print_cv_every == 0:    
        #     print('Epoch: {}, CV Loss: {}'.format(i, cv_loss))


    policy.save_model(filestr)
