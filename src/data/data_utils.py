import torch
from torch.utils.data import Dataset
import numpy as np
from src.expert_data import load_expert_data
import os
opj = os.path.join

class InteractionDatasetMultiAgent(Dataset):
    """
    Class to handle getting full multi-agent observations and actions
    """
    pass

class InteractionDatasetSingleAgent(Dataset):
    """Class to load states and actions for individual agents."""

    def __init__(self, output_dir='expert_data', loc:int = 0, tracks:list = [0], dtype=torch.float32):
        """
        Args:
            output_dir (string): Directory with all the images.
            loc (int): location index
            tracks (list[int]): track indices
        """
        self.output_dir = output_dir
        self.loc = loc
        self.tracks = tracks
        self.dtype = dtype
        self.keys = ['ego_state', 'relative_state', 'path', 'action', 'next_ego_state', 'next_relative_state', 'next_path']
        self._load_dataset()

    def _load_dataset(self):
        """
        Load the full datasets ahead of time
        """
        self.raw_data = {key:[] for key in self.keys}
        max_nv = 0
        for track in self.tracks:
            try:
                data = load_expert_data(path=self.output_dir, loc=self.loc, track=track)
                print('Loaded location {} track {}'.format(self.loc,track))
            except:
                print('Failed to load location {} track {}'.format(self.loc,track))
                continue
            max_nv = max(max_nv, data['relative_state'].shape[1])
            for key in self.keys:
                self.raw_data[key].append(data[key])

        # pad second dimension of relative state
        for i in range(len(self.raw_data['relative_state'])):
            nv1, nv2, d = self.raw_data['relative_state'][i].shape
            pad = torch.zeros(nv1, max_nv-nv2, d, dtype=self.dtype) * np.nan
            self.raw_data['relative_state'][i] = torch.cat((self.raw_data['relative_state'][i], pad), dim=1)
            self.raw_data['next_relative_state'][i] = torch.cat((self.raw_data['next_relative_state'][i], pad), dim=1)

        # cat lists
        for key in self.keys:
            self.raw_data[key] = torch.cat(self.raw_data[key]).type(self.dtype)

        # mandate equal length
        lengths = [len(self.raw_data[key]) for key in self.keys]
        assert min(lengths) == max(lengths), 'dataset lengths unequal'
        
    def __len__(self):
        return len(self.raw_data['ego_state'])

    def __getitem__(self, idx):
        """ 
        Sample from the dataset
        Args:
            idx: index or indices of B samples
        Returns:
            sample (dict): sample dictionary with the following entries:
                state (dict): state dictionary with the following entries:
                    ego_state (torch.tensor): (B, 5) raw state
                    relative_state (torch.tensor): (B, max_nv, d) relative state (padded with nans)
                    path (torch.tensor): (B, P, 2) tensor of P future path x and y positions
                action (torch.tensor): (B, 1) actions taken from each state
                next_stat (dict): next state dictionary with the following entries:
                    ego_state (torch.tensor): (B, 5) raw next state
                    relative_state (torch.tensor): (B, max_nv, d) next relative state (padded with nans)
                    path (torch.tensor): (B, P, 2) tensor of P future next path x and y positions
        """
        #sample = {key:self.raw_data[key][idx] for key in self.keys}
        sample = {
            'state':{
                'ego_state':self.raw_data['ego_state'][idx],
                'relative_state':self.raw_data['relative_state'][idx],
                'path':self.raw_data['path'][idx]
            },
            'action':self.raw_data['action'][idx],
            'next_state':{
                'ego_state':self.raw_data['next_ego_state'][idx],
                'relative_state':self.raw_data['next_relative_state'][idx],
                'path':self.raw_data['next_path'][idx]},
        }
        return sample