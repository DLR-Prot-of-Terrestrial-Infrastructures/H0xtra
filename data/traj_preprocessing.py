# Copyright (c) 2025 German Aerospace Center - Deutsches Zentrum fuer
# Luft- und Raumfahrt e. V. (DLR)
#
# Author(s):        Jonas Gunkel jonas.gunkel@dlr.de
# Date of creation: 2025-06-01

"""
This file contains the scripts for making trajectories from raw check-in data
"""

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os

def get_wdwe(dow):
    return 1 if ((dow == 0) or (dow == 6)) else 0

def create_user_sequence(data, city_id = None):
    """
    create sequences of location visitations for all individuals from raw data

    output array:
    dim 0: x value
    dim 1: y value
    dim 2: time delta
    dim 3: day of week
    dim 4: working day/weekend day
    dim 5: day
    dim 6: time
    dim 7: user id
    dim 8: city id
    """

    df = data.sort_values(['uid','d', 't'])
    df['t_continuous'] = (df['d']*48)+df['t']
    df['t_delta'] = df['t_continuous'] - df['t_continuous'].shift(1)
    df['dow'] = df['d'] % 7
    df['wdwe'] = df['dow'].apply(get_wdwe)

    ls = []

    for user, group in tqdm(df.sort_values(['d', 't']).groupby(by = 'uid')):

        group.iloc[0, group.columns.get_loc('t_delta')] = 0
        group_array = group[['x', 'y', 't_delta', 'dow', 'wdwe', 'd', 't']].dropna().to_numpy()
        group_array = np.concatenate((group_array, np.expand_dims(np.ones(group_array.shape[0])*user, 1), np.expand_dims(np.ones(group_array.shape[0])*city_id, 1)), axis = 1)
        ls.append(group_array)

    return ls

def create_data(cities, city_id = None):
    """
    Takes either dataframes or a list of dataframes (for multiple cities) and stores the trajectories as a numpy array.
    """

    if isinstance(cities, pd.core.frame.DataFrame):
        lst= create_user_sequence(data = cities, city_id = city_id)

    else:
        lst = []
        for city, city_id in zip(cities, city_id):
            ls = create_user_sequence(data = city, city_id = city_id)
            lst += ls

    arr = np.array([{'trajectory': seq, 'city': int(seq[0,-1]), 'user': int(seq[0,-2])} for seq in lst])
    os.makedirs('trajs', exist_ok=True)
    np.save('trajs/seq.npy', arr)



def forward_fill_time_series(arr):
    """
    Forward fill a trajectory
    """

    # get start and end times
    start_day, start_time = arr[0, 5], arr[0, 6]
    end_day, end_time = arr[-1, 5], arr[-1, 6]
    
    # generate range for (day, time)
    start_idx = start_day * 48 + start_time
    end_idx = end_day * 48 + end_time
    full_times = np.arange(start_idx, end_idx + 1)
    full_grid = np.column_stack((full_times // 48, full_times % 48))
    
    full_series = np.zeros((len(full_grid), 9))
    full_series[:, 5:7] = full_grid
    
    # Searchsorted for forward-filling
    original_times = arr[:, 5] * 48 + arr[:, 6]
    idx = np.searchsorted(original_times, full_times, side='right') - 1
    idx[idx < 0] = 0
    
    # Assign values
    full_series[:, :2] = arr[idx, :2]
    full_series[:, -2:] = arr[idx, -2:]
    full_series[:, 3] = full_series[:, 5] % 7
    full_series[:, 4] = np.where((full_series[:, 3] == 0) | (full_series[:, 3] == 6), 1, 0)
    
    return full_series

def forward_fill_time_series_skipmissing(arr):
    """
    Forward fill an trajectory, omitting days for which no data is reported
    """
    # Extract unique days and their first and last recorded timestamps
    unique_days, first_indices = np.unique(arr[:, 5], return_index=True)
    first_times = arr[first_indices, 6].astype(int)
    last_times = {day: arr[arr[:, 5] == day, 6].max().astype(int) for day in unique_days}
    
    # Determine the start of each day's range
    day_diffs = np.diff(unique_days, prepend=unique_days[0] - 1)
    start_times = np.where(day_diffs == 1, 0, first_times)
    
    # Determine the end of each day's range
    next_day_missing = np.diff(unique_days, append=unique_days[-1] + 1) > 1
    end_times = np.where(next_day_missing, [last_times[day] for day in unique_days], 47)
    
    # Generate full (day, time) grid
    full_days = np.concatenate([np.full(end - start + 1, day) for day, start, end in zip(unique_days, start_times, end_times)])
    full_times = np.concatenate([np.arange(start, end + 1) for start, end in zip(start_times, end_times)])
    full_grid = np.column_stack((full_days, full_times))
    
    # Filter the grid to only include times within the given range
    start_idx = arr[0, 5] * 48 + arr[0, 6]
    end_idx = arr[-1, 5] * 48 + arr[-1, 6]
    full_times_flat = full_grid[:, 0] * 48 + full_grid[:, 1]
    valid_mask = (full_times_flat >= start_idx) & (full_times_flat <= end_idx)
    full_grid = full_grid[valid_mask]
    
    full_series = np.zeros((len(full_grid), 9))
    full_series[:, 5:7] = full_grid  # Fill in (day, time) columns
    
    # Searchsorted for forward-filling
    original_times = arr[:, 5] * 48 + arr[:, 6]
    idx = np.searchsorted(original_times, full_times_flat[valid_mask], side='right') - 1
    idx[idx < 0] = 0  # Ensure no negative indices
    
    # Assign values
    full_series[:, :2] = arr[idx, :2]
    full_series[:, -2:] = arr[idx, -2:]
    full_series[:, 3] = full_series[:, 5] % 7
    full_series[:, 4] = np.where((full_series[:, 3] == 0) | (full_series[:, 3] == 6), 1, 0)
    
    return full_series


def filter_sequences_by_reports_per_day(sequences, min_reports_per_day=6, allow_gaps=True, fill=True):
    """
    Filters full sequences, keeping only those where each day has at least `min_reports_per_day` values. If allow_gaps is True, then include trajectories with missing days over the full period. If fill is True, then forward fills all gaps with the last reported location.
    """
    valid_sequences = []
    
    for seq in tqdm(sequences):
        
        # Count reports per day
        unique_days, counts_per_day = np.unique(seq[:, 5], return_counts=True)
                
        # If all days have at least `min_reports_per_day`, keep the traj
        if allow_gaps:
            if np.all(counts_per_day >= min_reports_per_day):
                if fill:
                    valid_sequences.append(forward_fill_time_series_skipmissing(seq))
                else:
                    valid_sequences.append(seq)
        else:
            if np.all(counts_per_day >= min_reports_per_day) and len(unique_days) == unique_days[-1]-unique_days[0]+1:
                if fill:
                    valid_sequences.append(forward_fill_time_series(seq))
                else:
                    valid_sequences.append(seq)
    
    return valid_sequences


if __name__ == "__main__":

    #paths to raw data
    city_a = 'raw/cityA-dataset.csv'
    city_b = 'raw/cityB-dataset.csv'
    city_c = 'raw/cityC-dataset.csv'
    city_d = 'raw/cityD-dataset.csv'

    df_a = pd.read_csv(city_a)
    df_b = pd.read_csv(city_b)
    df_c = pd.read_csv(city_c)
    df_d = pd.read_csv(city_d)

    # create all trajectories
    create_data(cities=(df_a, df_b, df_c, df_d), city_id = (1,2,3,4))

    ######### filter trajectories as described in our paper #########

    random.seed(1337)
    data_path = 'trajs/seq.npy'
    sequences = np.load(data_path, allow_pickle=True)
    cities = [1,2,3,4]
    sequences = [itm['trajectory'] for itm in sequences if itm['city'] in cities]

    # trajectory generation data
    seqs_trajgen = filter_sequences_by_reports_per_day(sequences, min_reports_per_day=5, allow_gaps=True, fill=True)

    # split into four cities
    seqs_trajgen_1 = [seq for seq in seqs_trajgen if seq[0,-1]==1]
    seqs_trajgen_2 = [seq for seq in seqs_trajgen if seq[0,-1]==2]
    seqs_trajgen_3 = [seq for seq in seqs_trajgen if seq[0,-1]==3]
    seqs_trajgen_4 = [seq for seq in seqs_trajgen if seq[0,-1]==4]

    # shuffle for train-val-test split
    random.shuffle(seqs_trajgen_1)
    random.shuffle(seqs_trajgen_2)
    random.shuffle(seqs_trajgen_3)
    random.shuffle(seqs_trajgen_4)

    train_1 = seqs_trajgen_1[:int(.7*len(seqs_trajgen_1))]
    val_1 = seqs_trajgen_1[int(.7*len(seqs_trajgen_1)):int(.8*len(seqs_trajgen_1))]
    test_1 = seqs_trajgen_1[int(.8*len(seqs_trajgen_1)):]

    train_2 = seqs_trajgen_2[:int(.7*len(seqs_trajgen_2))]
    val_2 = seqs_trajgen_2[int(.7*len(seqs_trajgen_2)):int(.8*len(seqs_trajgen_2))]
    test_2 = seqs_trajgen_2[int(.8*len(seqs_trajgen_2)):]

    train_3 = seqs_trajgen_3[:int(.7*len(seqs_trajgen_3))]
    val_3 = seqs_trajgen_3[int(.7*len(seqs_trajgen_3)):int(.8*len(seqs_trajgen_3))]
    test_3 = seqs_trajgen_3[int(.8*len(seqs_trajgen_3)):]

    train_4 = seqs_trajgen_4[:int(.7*len(seqs_trajgen_4))]
    val_4 = seqs_trajgen_4[int(.7*len(seqs_trajgen_4)):int(.8*len(seqs_trajgen_4))]
    test_4 = seqs_trajgen_4[int(.8*len(seqs_trajgen_4)):]

    train_set = train_1 + train_2 + train_3 + train_4
    val_set = val_1 + val_2 + val_3 + val_4
    test_set = test_1 + test_2 + test_3 + test_4

    sets_names = ['train', 'val', 'test']
    sets = [train_set, val_set, test_set]

    os.makedirs('trajs/trajgen', exist_ok=True)

    for nm, st in zip(sets_names, sets):

        lst = np.array([{'trajectory': seq, 'city': int(seq[0,-1]), 'user': int(seq[0,-2])} for seq in st])
        np.save(f'trajs/trajgen/{nm}_seqs.npy', lst)


    # next-location prediction data
    random.seed(1337)
    sequences = np.load(data_path, allow_pickle=True)
    cities = [1,2,3,4]
    sequences = [itm['trajectory'] for itm in sequences if itm['city'] in cities]
    seqs_nextloc = filter_sequences_by_reports_per_day(sequences, min_reports_per_day=5, allow_gaps=False, fill=False)

    # split into four cities
    seqs_nextloc_1 = [seq for seq in seqs_nextloc if seq[0,-1]==1]
    seqs_nextloc_2 = [seq for seq in seqs_nextloc if seq[0,-1]==2]
    seqs_nextloc_3 = [seq for seq in seqs_nextloc if seq[0,-1]==3]
    seqs_nextloc_4 = [seq for seq in seqs_nextloc if seq[0,-1]==4]

    # shuffle for train-val-test split
    random.shuffle(seqs_nextloc_1)
    random.shuffle(seqs_nextloc_2)
    random.shuffle(seqs_nextloc_3)
    random.shuffle(seqs_nextloc_4)

    train_1 = seqs_nextloc_1[:int(.7*len(seqs_nextloc_1))]
    val_1 = seqs_nextloc_1[int(.7*len(seqs_nextloc_1)):int(.8*len(seqs_nextloc_1))]
    test_1 = seqs_nextloc_1[int(.8*len(seqs_nextloc_1)):]

    train_2 = seqs_nextloc_2[:int(.7*len(seqs_nextloc_2))]
    val_2 = seqs_nextloc_2[int(.7*len(seqs_nextloc_2)):int(.8*len(seqs_nextloc_2))]
    test_2 = seqs_nextloc_2[int(.8*len(seqs_nextloc_2)):]

    train_3 = seqs_nextloc_3[:int(.7*len(seqs_nextloc_3))]
    val_3 = seqs_nextloc_3[int(.7*len(seqs_nextloc_3)):int(.8*len(seqs_nextloc_3))]
    test_3 = seqs_nextloc_3[int(.8*len(seqs_nextloc_3)):]

    train_4 = seqs_nextloc_4[:int(.7*len(seqs_nextloc_4))]
    val_4 = seqs_nextloc_4[int(.7*len(seqs_nextloc_4)):int(.8*len(seqs_nextloc_4))]
    test_4 = seqs_nextloc_4[int(.8*len(seqs_nextloc_4)):]

    train_set = train_1 + train_2 + train_3 + train_4
    val_set = val_1 + val_2 + val_3 + val_4
    test_set = test_1 + test_2 + test_3 + test_4

    sets_names = ['train', 'val', 'test']
    sets = [train_set, val_set, test_set]

    os.makedirs('trajs/nextloc', exist_ok=True)

    for nm, st in zip(sets_names, sets):

        lst = np.array([{'trajectory': seq, 'city': int(seq[0,-1]), 'user': int(seq[0,-2])} for seq in st])
        np.save(f'trajs/nextloc/{nm}_seqs.npy', lst)