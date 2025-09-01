# Copyright (c) 2025 German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Jonas Gunkel jonas.gunkel@dlr.de
# Date of creation: 2025-06-01

"""
This file contains the methods that provide pytorch dataloaders for training, validating, and testing
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial
from itertools import groupby
import os
import random

def data_provider_hyper(data_path, matrix_path=None, batch_size=32, shuffle=True, max_traj_len=1000, test_traj_len = None, train_cities=None, test_cities=None, include_time=True, use_start_loc=False, start_loc=0, start_time=0, crop_by_day=False, workers = 8, eval=False, pin_memory=False):
    """
    Function that provides pytorch DataLoaders. Each sample of the Dataloader consists of a tuple (x, y, lengths, map, cities),
    being the input x, the expected output y, the lenghts of x and y, respectively, the matrix map, representing the map of the
    corresponding city, and the id of the city itself.
    """

    train_path = os.path.join(data_path, 'train_seqs.npy')
    val_path = os.path.join(data_path, 'val_seqs.npy')
    test_path = os.path.join(data_path, 'test_seqs.npy')

    if eval:
        test_dataset = HyperDataset(test_path, matrix_path, cities=test_cities, max_len=test_traj_len, include_time=include_time, crop_by_day=crop_by_day)
        test_loader = DataLoader(test_dataset, batch_size, num_workers=workers, shuffle=shuffle, collate_fn=partial(hyper_collate_padded_time, use_start_loc=use_start_loc, start_loc=start_loc, start_time=start_time), pin_memory=pin_memory)
        return test_dataset, test_loader

    if test_cities is None:
        test_cities = train_cities

    if test_traj_len is None:
        test_traj_len = max_traj_len

    workers_tr = min([(os.cpu_count() // 4)*3, workers])
    workers_vl = min([(os.cpu_count() // 8), workers//2])
    workers_tst = min([(os.cpu_count() // 8), 1])
    
    train_dataset = HyperDataset(train_path, matrix_path, cities=train_cities, max_len=max_traj_len, include_time=include_time, crop_by_day=crop_by_day)
    val_dataset = HyperDataset(val_path, matrix_path, cities=train_cities, max_len=max_traj_len, include_time=include_time, crop_by_day=crop_by_day)
    test_dataset = HyperDataset(test_path, matrix_path, cities=test_cities, max_len=test_traj_len, include_time=include_time, crop_by_day=crop_by_day)


    if include_time:
        train_loader = DataLoader(train_dataset, batch_size, num_workers=workers_tr, shuffle=shuffle, collate_fn=partial(hyper_collate_padded_time, use_start_loc=use_start_loc, start_loc=start_loc, start_time=start_time), pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size, num_workers=workers_vl, shuffle=False, collate_fn=partial(hyper_collate_padded_time, use_start_loc=use_start_loc, start_loc=start_loc, start_time=start_time), pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size, num_workers=workers_tst, shuffle=False, collate_fn=partial(hyper_collate_padded_time, use_start_loc=use_start_loc, start_loc=start_loc, start_time=start_time), pin_memory=pin_memory)
        return train_loader, val_loader, test_loader

    train_loader = DataLoader(train_dataset, batch_size, num_workers=workers_tr, shuffle=shuffle, collate_fn=partial(hyper_collate_padded, use_start_loc=use_start_loc, start_loc=start_loc), pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=workers_vl, shuffle=False, collate_fn=partial(hyper_collate_padded, use_start_loc=use_start_loc, start_loc=start_loc), pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=workers_tst, shuffle=False, collate_fn=partial(hyper_collate_padded, use_start_loc=use_start_loc, start_loc=start_loc), pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


class HyperDataset(Dataset):
    def __init__(self, data_path, matrix_path=None, cities=None, max_len=None, include_time=False, crop_by_day=False):
        super().__init__()
        self.include_time = include_time
        self.crop_by_day = crop_by_day

        if isinstance(cities, str):
            cities = eval(cities)

        if isinstance(cities, int):
            cities = [cities]

        self.cities = cities
        self.maps = np.load(matrix_path, allow_pickle=True)
        self.maps = self.norm_matrices([torch.tensor(arr['map']) for arr in self.maps])
        self.data = self.filter(np.load(data_path, allow_pickle=True), cities=cities)

        assert set([int(dt[0,-1]) for dt in self.data]) >= set(cities), f'filtering failed, data slippery! got trajs from {list(set([dt[0,-1] for dt in self.data])-set(cities))}, which is not part of {cities}'
        assert set([int(dt[0,-1]) for dt in self.data]) <= set(cities), f'wrong list of cities provided! Provided cities are {cities}, but data only contains cities from {list(set([dt[0,-1] for dt in self.data]))}'

        if not max_len is None or self.crop_by_day:
            self.data = self.crop_length(self.data, max_len)

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        if self.include_time:
            return [torch.stack((torch.tensor(((sequence[:,0]-1)*200) + (sequence[:,1]-1) + 1).to(torch.long), torch.tensor(sequence[:,3] + 1).to(torch.long), torch.tensor(sequence[:,6] + 1).to(torch.long)), dim=-1), self.maps[int(sequence[0,-1])-1], int(sequence[0,-1])-1] # locs + day of week + time of day
        return [torch.tensor(((sequence[:,0]-1)*200) + (sequence[:,1]-1)+1).to(torch.long), self.maps[int(sequence[0,-1])-1], int(sequence[0,-1])-1] # +1 at the end is needed to padd with zeros
    
    def crop_length(self, data, max_len):
        """
        Crop sequences, either by day or into subsequences of no longer than the specified max_len.
        """
        if self.crop_by_day:
            lst = [s for seq in data for s in self.split_by_day(seq, col=5)]
            return [s for s in lst if len(s) == 48]
        return [subarr for seq in data for subarr in np.array_split(seq, seq.shape[0]//max_len+1)]
    
    def split_by_day(self, seq, col=5):
        """
        Split sequences into unique days (specified by column 5).
        """
        return [np.array(list(group)) for _, group in groupby(seq, key=lambda x: x[col])]
    
    def norm_matrices(self, matrices):
        """
        Norm matrices in last two dimensions by dividing through the maximum to create relative amount of POIs
        """
        return [torch.div(mat, torch.amax(mat, dim=(0,1)).view(1,1,-1)+1e-5) for mat in matrices]
    
    def filter(self, data, cities=None):
        """
        Filters the given trajectories based on chosen cities.
        """

        if cities is None:
            return [arr['trajectory'] for arr in data]
        
        if not isinstance(cities, list):
            cities = [cities]
        
        return [itm['trajectory'] for itm in data if itm['city'] in cities]
            

def hyper_collate_padded(batch, use_start_loc=False, start_loc=0):

    pad_value = 0

    seqs = [item[0] for item in batch]
    maps = [item[1] for item in batch]
    cities = [item[2] for item in batch]

    if use_start_loc:
        seqs = [torch.cat((torch.tensor([start_loc]), t)) for t in seqs]

    lengths = torch.tensor([(t.size()[0]-1) for t in seqs]) # -1 is needed as we crop & shift for causal training
    batch_in = torch.nn.utils.rnn.pad_sequence([seq[:-1] for seq in seqs], batch_first = True, padding_value=pad_value)
    batch_out = torch.nn.utils.rnn.pad_sequence([seq[1:] for seq in seqs], batch_first = True, padding_value=pad_value)
    maps = torch.tensor(np.array(maps)).permute(0,3,1,2).float()
    cities = torch.tensor(cities)

    return batch_in, batch_out, lengths, maps, cities


def hyper_collate_padded_time(batch, use_start_loc=False, start_loc=0, start_time=0):

    pad_value = 0

    seqs = [item[0] for item in batch]
    maps = [item[1] for item in batch]
    cities = [item[2] for item in batch]

    if use_start_loc:
        seqs = [torch.cat((torch.cat((torch.tensor(start_loc).unsqueeze(0), t[0,1].unsqueeze(0), torch.tensor(start_time).unsqueeze(0))).unsqueeze(0), t)) for t in seqs]

    lengths = torch.tensor([(t.size()[0]-1) for t in seqs]) # -1 is needed as we crop & shift for causal training
    batch_in = torch.nn.utils.rnn.pad_sequence([seq[:-1,:] for seq in seqs], batch_first = True, padding_value=pad_value)
    batch_out = torch.nn.utils.rnn.pad_sequence([seq[1:,:] for seq in seqs], batch_first = True, padding_value=pad_value)
    maps = torch.tensor(np.array(maps)).permute(0,3,1,2).float()
    cities = torch.tensor(cities)

    return batch_in, batch_out, lengths, maps, cities