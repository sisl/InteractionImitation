import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
#from torchvision import transforms, utils
from interimit.expert_data import load_expert_data
import os
opj = os.path.join

class InteractionDatasetMultiAgent(Dataset):
    """
    Class to handle getting full multi-agent observations and actions
    """
    pass

class InteractionDatasetSingleAgent(Dataset):
    """Class to load states and actions for individual agents."""

    def __init__(self, output_dir='expert_data', loc:int = 0, tracks:list = [0], transforms={}):
        """
        Args:
            output_dir (string): Directory with all the images.
            loc (int): location index
            tracks (list[int]): track indices
            transforms (dict): dictionary of transforms to apply to different variables
        """
        self.output_dir = output_dir
        self.loc = loc
        self.tracks = tracks
        self.transforms = transforms
        #self.action_transform = transforms.get('action', None)
        #self.state_transform = transforms.get('state', None)
        #self.relative_state_transform = transforms.get('relative_state', None)
        #self.paths_x_transform = transforms.get('paths_x', None)
        #self.paths_y_transform = transform.get('paths_y',None)

        self._load_dataset()

    def _load_dataset(self):
        """
        Load the full datasets ahead of time
        """
        self.raw_data = {'state':[], 'relative_state':[], 'action':[], 'path_x':[], 'path_y':[]}
        max_nv = 0
        for track in self.tracks:
            try:
                observations, actions = load_expert_data(path=self.output_dir, loc=self.loc, track=track)
                print('Loaded location {} track {}'.format(self.loc,track))
            except:
                print('Failed to load location {} track {}'.format(self.loc,track))
                continue
            T = len(actions)
            for t in range(T):
                nni = ~torch.isnan(observations[t]['state'][:,0])
                max_nv = max(max_nv,nni.count_nonzero())
                self.raw_data['state'].append(observations[t]['state'][nni])
                self.raw_data['relative_state'].append(observations[t]['relative_state'][nni.nonzero(),nni.nonzero()])
                self.raw_data['action'].append(actions[t][nni])
                self.raw_data['path_x'].append(observations[t]['paths'][0][nni])
                self.raw_data['path_y'].append(observations[t]['paths'][1][nni])

        # cat lists
        self.raw_data['state'] = torch.cat(self.raw_data['state'])
        self.raw_data['action'] = torch.cat(self.raw_data['action'])
        self.raw_data['path_x'] = torch.cat(self.raw_data['path_x'])
        self.raw_data['path_y'] = torch.cat(self.raw_data['path_y'])

        # pad second dimension of relative state
        for i in range(len(self.raw_data['relative_state'])):
            nv1, nv2, d = self.raw_data['relative_state'][i].shape
            pad = torch.zeros(nv1, max_nv-nv2, d) * np.nan
            self.raw_data['relative_state'][i] = torch.cat((self.raw_data['relative_state'][i], pad), dim=1)
        self.raw_data['relative_state'] = torch.cat(self.raw_data['relative_state'])

        # mandate equal length
        assert len(self.raw_data['state']) == len(self.raw_data['relative_state']) \
            == len(self.raw_data['action']) \
            == len(self.raw_data['path_x']) \
            == len(self.raw_data['path_y']), 'dataset lengths unequal'
        
    def __len__(self):
        return len(self.raw_data['state'])

    def __getitem__(self, idx):
        """ 
        Sample from the dataset
        Args:
            idx: index or indices of B samples
        Returns:
            sample (dict): sample dictionary with the following entries:
                state (torch.tensor): (B, 5) raw state
                relative_state (torch.tensor): (B, max_nv, d) relative state (padded with nans)
                path_x (torch.tensor): (B, P) tensor of P future path x positions
                path_y (torch.tensor): (B, P) tensor of P future path y positions
                action (torch.tensor): (B, 1) actions taken from each state
        """
        keys = ['state', 'relative_state', 'path_x', 'path_y', 'action']
        sample = {key:self.raw_data[key][idx] for key in keys}

        for key in keys: 
            if key in self.transforms.keys():
                sample[key] = self.transforms[key](sample[key])

        return sample