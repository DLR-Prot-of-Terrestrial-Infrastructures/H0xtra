# partially adapted from https://github.com/Star607/Cross-city-Mobility-Transformer

"""
This file contains several utility functions required for evaluating the model's output.
"""

import numpy as np
import random
import warnings
from tqdm import tqdm
from collections import Counter
import torch
from torch.nn import functional as F
from metrics import IndividualEval
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_top_k(output, target, padding_value):
    """
    Compute the top-k accuracies for multiple values of k, mean reciprocal rank and average distance error.
    """

    acc_K = [1, 5, 10, 20] # ks chosen for top-k accuracy
    result = {}

    # target location ids to coordinates
    target_x = torch.div(target - 1, 200, rounding_mode="floor")
    target_y = torch.remainder(target - 1, 200)

    # pad mask
    pad_mask = (target != padding_value)

    # order location ids by probability
    vocab_size = output.shape[-1]
    topk_vals, topk_indices = torch.topk(output, vocab_size, dim=-1)
    target = target.unsqueeze(-1).expand_as(topk_indices)

    correct = (topk_indices == target)
    correct = correct & pad_mask.unsqueeze(-1)

    # get top-k accuracies
    for k in acc_K:
        topk_correct = correct[:, :, :k].any(dim=-1)
        accuracy = topk_correct.float().sum()
        result[k] = accuracy.item()
    
    result['num_of_test'] = pad_mask.float().sum().item()

    # get Mean Reciprocal Rank (MRR)
    ranks = (correct.float() * torch.arange(1, vocab_size + 1, device=output.device)).sum(dim=-1)
    reciprocal_ranks = torch.where(ranks > 0, 1.0 / ranks, torch.zeros_like(ranks))
    mrr = (reciprocal_ranks * pad_mask).sum()# / pad_mask.float().sum()
    result['mrr'] = mrr.item()

    # predicted location ids to coordinates
    top1_indices = topk_indices[:, :, 0]
    pred_x = torch.div(top1_indices - 1, 200, rounding_mode="floor")#(top1_indices - 1 % 200).float()
    pred_y = torch.remainder(top1_indices - 1, 200)#(top1_indices - 1 // 200).float()

    distances = geodistance(target_x, target_y, pred_x, pred_y)
    distances = distances * pad_mask.float()
    distances = distances.sum().item()
    result['distances'] = distances
    
    return result

def partial_metrics_sum(results):
    """
    Aggregate metrics across batches.
    """
    acc_K = [1, 5, 10, 20]
    result = {}

    if isinstance(results[acc_K[0]], list):
        num_of_test = np.sum(results['num_of_test'])
        for K in acc_K:
            result[K] = np.sum(results[K])
            result[K] /= num_of_test
            #print(f'top{K} accuracy: {result[K]}')
        result['mrr'] = np.sum(results['mrr'])
        result['mrr'] /= num_of_test
        result['distances'] = np.sum(results['distances'])
        result['distances'] /= num_of_test
    else:
        num_of_test = results['num_of_test']
        for K in acc_K:
            result[K] = results[K]
            result[K] /= num_of_test
            print(f'top{K} accuracy: {result[K]}')
        result['mrr'] = results['mrr']
        result['mrr'] /= num_of_test
        result['distances'] = results['distances']
        result['distances'] /= num_of_test

    return result

def geodistance(x1, y1, x2, y2):
    """
    Compute the Euclidean distance between two points (x1, y1) and (x2, y2) in meters.
    """
    x1 = 500. * x1  # transform to meters
    y1 = 500. * y1
    x2 = 500. * x2
    y2 = 500. * y2
    distance = torch.sqrt(torch.square(x1 - x2) + torch.square(y1 - y2))
    return distance

def append_dict(d_o, d_n):
    for key, value in d_n.items():
        if key in d_o:
            d_o[key].append(value)
        else:
            d_o[key] = [value]

def extend_dict(d_o, d_n):
    for key, value in d_n.items():
        if key in d_o:
            d_o[key].extend(value)
        else:
            d_o[key] = value

def print_last_k_mean(d, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        print(*[f'{key}:  {np.nanmean(value[-k:])}' for key, value in d.items()], sep=', ')

def distance_versus_probability(prediction, target, bin_edges):
    """
    Plot cumulative probability mass as a function of distance.
    """

    B, K = prediction.shape
    device = prediction.device

    assert K == 40001

    prediction = prediction[:,1:]
    prediction = torch.softmax(prediction, dim=-1)
    probablities, topk_indices = torch.topk(prediction, K-1, dim=-1)

    # location ids to coordinates
    target_x = torch.div(target - 1, 200, rounding_mode="floor")
    target_y = torch.remainder(target - 1, 200)
    pred_x = torch.div(topk_indices, 200, rounding_mode="floor")
    pred_y = torch.remainder(topk_indices, 200)
    
    distances = geodistance(target_x, target_y, pred_x, pred_y)
    bin_indices = torch.bucketize(distances, bin_edges, right=False)

    bin_probs = torch.zeros((B, len(bin_edges) + 1), dtype=torch.float, device=device)
    for b in range(B):
        bin_probs[b].scatter_add_(0, bin_indices[b], probablities[b].to(torch.float))

    # cumulative probability
    cumulative_probs = torch.cumsum(bin_probs, dim=-1)
    mean_cumulative = cumulative_probs.mean(dim=0)

    return mean_cumulative.detach().cpu().numpy()


@torch.no_grad()
def generate_and_test_fixed_emb(model, data, device, path, start_token_loc=0, start_token_time=0, length=None, samples=None, batch_size=1, day=None, top_k=None, temperature=1):
    """
    Autoregressively generate trajectories with fixed location embeddings.
    
    Arguments:
        model: trained model.
        data: dataset.data containing the true trajectory data.
        device: device to perform model passes on
        path: folder to save the generated trajectories
        length: length of samples
        samples: number of samples. If None, the routine generates the same of amount of trajectories as in data
        batch_size: batch size for generation
        day: if specified, then only for this day data is generated. Otherwise day is sampled according to the occurances in data
        top_k: restrict to the k most propable locations as predicted by the model
        temperature: temperature for scaling the logits
    """

    city = data.dataset.cities
    if isinstance(city, list):
        assert len(city) == 1, f'Generation is only implemented for single cities! The current data contains {len(city)} cities.'
        city = city[0]

    city_map = data.dataset.maps[int(city)-1].to(device)
    data, days = get_data(data.dataset.data)

    if not day:
        value_counts = Counter(days)
        values = np.array(list(value_counts.keys()))
        counts = np.array(list(value_counts.values()))
        probabilities = counts / counts.sum()

    if not samples:
        samples = len(data)
    if not length:
        length = len(data[0])

    generated_data = []

    model.eval()

    matrix = city_map.permute(2,0,1).float()
    model.fix_embeddings(matrix)

    # define start 'tokens' for sampling
    start_day = torch.ones((batch_size, 1, 1)).to(device)
    start_loc = torch.ones((batch_size, 1, 1)).to(device)
    start_time = torch.ones((batch_size, 1, 1)).to(device)
    time = torch.ones((batch_size, 1, 1)).to(device)

    for k in tqdm(range(0, samples, batch_size)):
    
        current_time = 0
        current_day = day if day else np.random.choice(values, p=probabilities) # either use given day or sample from distribution defined above
        start_day.fill_(current_day)
        start_loc.fill_(start_token_loc)
        start_time.fill_(start_token_time)
        time.fill_(current_time)
        start_idx = torch.cat((start_loc, start_day, start_time), dim=-1).to(torch.long)


        for j in range(length):
            with torch.no_grad():
                out = model.generate_fixed_emb(start_idx)
            logits = out[:,-1,:] / temperature
            logits[:,0] = -float('Inf') # set the logit value of the pad token to -inf
            logits[:,start_token_loc] = -float('Inf') # set the logit value of the artificial start location to -inf
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)   # apply softmax to convert logits to (normalized) probabilities
            out_idx = torch.multinomial(probs, num_samples=1)   # sample from the distribution
            time += 1 # progress to next time step
            new_idx = torch.cat((out_idx.unsqueeze(-1), start_day, time), dim=-1)
            start_idx = torch.cat((start_idx, new_idx), dim=1).to(torch.long)

        result_batch = np.array(start_idx[:,1:,0].cpu())
        generated_data.append(result_batch)

    # evaluate
    evaluator = IndividualEval(days=length//48)

    gt_data = np.array(data)
    generated_data = np.concatenate(generated_data)

    # compute metrics
    d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd = evaluator.get_individual_jsds(generated_data, gt_data, plot=False, pth=path)
    result = {'distance': d_jsd, 'radius': g_jsd, 'duration': du_jsd, 'periodicity': p_jsd, 'G-rank': l_jsd, 'I-rank': f_jsd}

    # save generated trajectories
    used_data = np.array({'gt':gt_data, 'gen': generated_data})
    np.save(os.path.join(path, 'data.npy'), used_data)

    model.train()

    return used_data, result

def get_data(data):
    return [np.array(((sequence[:,0]-1)*200) + (sequence[:,1]-1) + 1) for sequence in data], [sequence[0,3] + 1 for sequence in data]


def create_heatmaps(sequences, k=25):
    """
    Create heatmaps from trajectories
    """

    lngths = len(sequences) if isinstance(sequences, list) else 1
    if lngths == 1:
        sequences = [sequences]

    height = 200
    width = 200
    time_interval = 48

    heatmaps = np.zeros((lngths, time_interval, height, width))

    for l in range(lngths):
        for traj in sequences[l]:
            traj = traj.astype(int)
            x = (traj - 1) // 200
            y = (traj - 1) % 200
            time_idx = np.arange(len(x))
            np.add.at(heatmaps, (l, time_idx, x, y), 1)

    summed_heatmaps = np.sum(heatmaps, axis=1) # aggregated over time steps

    hm = np.array({'hm': heatmaps,'hm_summed': summed_heatmaps})

    return hm