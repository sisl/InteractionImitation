import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import pickle
import itertools
from torch.utils.tensorboard import SummaryWriter

from src.policies import IntersimStateNet, IntersimStateActionNet, IntersimPolicy, generate_transforms
from src.util.transform import MinMaxScaler
from src.util.nn_training import optimizer_factory
from tqdm import tqdm
import json5
from ray import tune

def vd_config(ray_config):
    config = {
        'policy_net': {
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
        },
        'value_net': {
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
            'action_dim': 1,
            'head': {
                'input_dim': 0, # computed in constructor
                'hidden_n': ray_config['head_hidden_n'],
                'hidden_dim': ray_config['head_hidden_dim'],
                'output_dim': 1, # number of outputs e.g. number of actions, or just one
                'final_activation': ray_config['head_final_activation'],
            },
        },
        'policy_optim': {
            'optimizer':'adam',
            'lr':ray_config['policy_lr'],
            'weight_decay':ray_config['policy_weight_decay']
        },
        'value_optim': {
            'optimizer':'adam',
            'lr':ray_config['value_lr'],
            'weight_decay':ray_config['value_weight_decay']
        },
        'train_epochs': 40,
        'train_batch_size': ray_config['train_batch_size'],
        'discount': ray_config['discount'],
        'clip_grad_norm': ray_config['clip_grad_norm'],
    }
    return config

class ValueDicePolicy(IntersimPolicy):
    """
    Class for value dice policy
    """
    
    def __init__(self, config: dict, transforms: dict):
        """
        Initialize ValueDicePolicy
        Args:
            config (dict): configuration file to initialize IntersimDeepSetsNet with
            transforms (dict): dictionary of transforms to apply to different fields
        """
        super(ValueDicePolicy, self).__init__(config, transforms)
        self._policy = IntersimStateNet(config["policy_net"])
        self._value = IntersimStateActionNet(config["value_net"])
        
    @property 
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        self._value = value

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
        model._policy.load_state_dict(torch.load(filestr+'_policy.pt'))
        model._value.load_state_dict(torch.load(filestr+'_value.pt'))
        return model
    
    def parameters(self):
        return itertools.chain(self._policy.parameters(), self._value.parameters())

    def policy_parameters(self):
        return self.policy.parameters()

    def value_parameters(self):
        return self.value.parameters()

    def eval(self):
        self.policy.eval()
        self.value.eval()

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
        torch.save(self._policy.state_dict(), filestr+'_policy.pt')
        torch.save(self._value.state_dict(), filestr+'_value.pt')



def train(config, policy, train_dataset, cv_dataset, filestr, **kwargs):
    
    using_ray = kwargs.get('ray', False)
    if using_ray:
        print('using ray')

    # hyperparams
    train_epochs = config['train_epochs']
    train_batch_size = config['train_batch_size']
    discount = config['discount']
    clip_grad_norm = config['clip_grad_norm']

    cv_every = 1
    print_epoch_every = 1000
    print_cv_every = 5
    checkpoint_every = 100
    cv_batch_size = 256 # doesn't matter

    # training and testing dataloaders
    training_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    cv_loader = DataLoader(cv_dataset, batch_size=cv_batch_size, shuffle=True)

    # change policy dtype
    dtype = train_dataset[0]['state']['ego_state'].dtype
    policy.policy = policy.policy.type(dtype)
    policy.value = policy.value.type(dtype)

    # define loss function
    def f_value_dice_loss(batch):
        # get s, a, s', s_0 from batch
        state = batch['state']
        action = batch['action']
        next_state = batch['next_state']
        initial_state = state

        # append action to state batches
        #   use expert action for s
        state['action'] = action
        #   run s' and s_0 through policy
        initial_state['action'] = policy(initial_state)
        next_state['action'] = policy(next_state)

        # transform state and action before inputting to value network
        # (for the policy network this is done in policy.__call__() )
        state = policy.transform_observation(state)
        initial_state = policy.transform_observation(initial_state)
        next_state = policy.transform_observation(next_state)

        # evaluate value network
        value = (policy.value(state))
        value_init = (policy.value(initial_state))
        value_next = (policy.value(next_state))
        
        # linear loss
        linear_loss = (1 - discount) * torch.mean(value_init)

        # nonlinear loss
        value_diff = value - discount * value_next
        nonlinear_loss = torch.logsumexp(value_diff, dim=0)

        loss = nonlinear_loss - linear_loss
        return loss


    policy_optimizer = optimizer_factory(config['policy_optim'], policy.policy_parameters())
    value_optimizer = optimizer_factory(config['value_optim'], policy.value_parameters())

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

            loss = f_value_dice_loss(batch)

            policy_loss = -loss #+ ORTHOGONAL_REGULARIZER
            value_loss = loss #+ GRADIENT_REGULARIZER

            # compute loss and step optimizer
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            value_loss.backward()

            clip_grad_norm_(policy.policy.parameters(), clip_grad_norm)
            clip_grad_norm_(policy.value.parameters(), clip_grad_norm)

            policy_optimizer.step()
            value_optimizer.step()

            epoch_loss += loss.item() / len(train_dataset)
        
        # if i % print_epoch_every == 0:
        #     print('Epoch: {}, Training Loss: {}'.format(i, epoch_loss))

        # measure cv loss
        if i % cv_every == 0:
            with torch.no_grad():
                cv_loss = 0.
                for (batch_idx, batch) in enumerate(cv_loader):
                    loss = f_value_dice_loss(batch)
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
