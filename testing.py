# partially adapted from https://github.com/karpathy/nanoGPT, (copyright (c) 2022 Andrej Karpathy)

"""
This file contains the testing script for the provided pretrained models
"""

import os
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch._dynamo

from model.transformer import GPTConfig
from model.hypernetwork import HyperNetworkConfig
from model.h0xtra import H0xtra
from data.provider_hyper import data_provider_hyper
from utils import partial_metrics_sum, get_top_k, append_dict, set_seed, create_heatmaps, generate_and_test_fixed_emb

os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
torch._dynamo.config.suppress_errors = True
torch.autograd.set_detect_anomaly(True)


def test_H0xtra(cfg):

    out_dir = os.path.join('results', cfg.task, cfg.model, str(cfg.test_city))
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg.seed)
    device = cfg.device

    # model init
    gpt_model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size,
                    bias=cfg.bias, vocab_size=cfg.vocab_size, dropout=cfg.dropout, time_size=cfg.time_size) # start with model_args from command line
    hyper_model_args = dict(n_layers=cfg.h_n_layers, in_channels=cfg.h_in_channels, d_embed=cfg.h_d_embed, kernel_size=cfg.h_kernel_size, hypernetwork=cfg.h_net, crop_by_day=cfg.crop_by_day, coords=cfg.h_coords)

    gptconf = GPTConfig(**gpt_model_args)
    hyperconf = HyperNetworkConfig(**hyper_model_args)

    model = H0xtra(hyperconf, gptconf, task=cfg.task)
    model.to(device)

    # Test model on test cities
    checkpoint = torch.load(os.path.join('pretrained', cfg.task ,cfg.model + '.pt'), weights_only=False)
    state_dict = checkpoint if cfg.model == 'mc' else checkpoint[cfg.test_city]
    model.load_state_dict(state_dict)
    model.eval()

    # compile the model
    if cfg.compile:
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    test_data, test_loader = data_provider_hyper(cfg.data_path, cfg.matrix_path, shuffle=True, batch_size=128, test_cities=[cfg.test_city], max_traj_len=cfg.max_traj_len, test_traj_len=cfg.test_traj_len, use_start_loc=cfg.use_start_loc, crop_by_day=cfg.crop_by_day, eval=True, workers=cfg.workers)
    
    if cfg.task == 'nextloc':

        losses = torch.zeros(len(test_loader))
        results = {}
        for k, data in enumerate(tqdm(test_loader)):
            X, Y, lngths, maps, cities = data
            X = X.to(device)
            Y = Y.to(device)
            maps = maps.to(device)
            lngths = lngths.to(device)
            with torch.no_grad():
                logits, targets, loss = model.generate(X, maps, Y, test_last=True, lengths=lngths)
            losses[k] = loss.item()
            append_dict(results, get_top_k(logits, targets, padding_value=0)) # compute metrics
        result = partial_metrics_sum(results)

        overall_results = {
        'test_loss': losses,
        'result': result,
    }

    if cfg.task == 'trajgen':

        gen_data, result = generate_and_test_fixed_emb(model, test_loader, device=cfg.device, start_token_loc=cfg.start_loc, start_token_time=cfg.start_time, batch_size=128, path=out_dir)

        if cfg.make_heatmaps:
                hm = create_heatmaps([gen_data.item()['gt'], gen_data.item()['gen']]) # create heatmaps from trajectories
        else:
            hm = None

        overall_results = {
            'result': result,
            'heatmaps': hm,
        }

    np.save(os.path.join(out_dir, 'test_results.npy'), np.array(overall_results))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # system
    parser.add_argument('--seed',  default=1337, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--compile', default=True, type=bool)
    parser.add_argument('--out_dir',  default='results', type=str)

    # data
    parser.add_argument('--matrix_path', default='data/maps/maps.npy', type=str)
    parser.add_argument('--data_path', default='data/trajs', type=str)

    # test setup
    parser.add_argument('--model', default='zs', type=str, choices=['zs', 'sc', 'mc'], help='Zero-shot, single-city, or multi-city version of H0xtra')
    parser.add_argument('--task', default='nextloc', type=str, choices=['nextloc', 'trajgen'], help='Choose a task: next-location prediction or trajectory generation')
    parser.add_argument('--test_city', default=4, type=int, help='City to test on')
    parser.add_argument('--test_only_last', default=True, type=bool, help='Whether next location prediction is evaluated only on the last location of a trajectory')

    # transformer configuration
    parser.add_argument('--n_layer', default=4, type=int, help='Number of layers in transformer backbone')
    parser.add_argument('--n_head', default=16, type=int, help='Number of attention heads in transformer backbone')
    parser.add_argument('--n_embd', default=32, type=int, help='Dimension of embedding space (per input variable)')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--bias', default=False, type=bool)

    # hypernetwork configuration
    parser.add_argument('--h_n_layers', default=8, type=int, help='Number of layers in hypernetwork')
    parser.add_argument('--h_in_channels', default=85, type=int, help='Number of input channels for hypernetwork. Default choice equals number of POI categories')
    parser.add_argument('--h_d_embed', default=32, type=int, help='Output dimension of hypernetwork (i.e., the dimension of the location embeddings)')
    parser.add_argument('--h_kernel_size', default=8, type=int, help='Kernel size in CNN hypernetwork')
    parser.add_argument('--h_net', default='unet', type=str, choices=['unet', 'cnn', 'mlp'], help='Hypernetwork architecture')
    parser.add_argument('--h_coords', default='s', type=str, choices=['s', 'c', 'n'], help='Positional information for the hypernetwork. The "s" stands for sinusoidal positional encodings, "c" for integer coords, and "n" for no coords')

    cfg = parser.parse_args()

    # adjust data to architecture
    if cfg.h_coords == 'c':
        cfg.matrix_path = 'data/maps/maps_coords.npy'

    # adjust config to chosen task
    if cfg.task == 'nextloc':
        cfg.use_start_loc = False
        cfg.vocab_size = 40001
        cfg.time_size = 49
        cfg.max_traj_len = 250
        cfg.test_traj_len = 40
        cfg.crop_by_day = False
        cfg.data_path = 'data/trajs/nextloc'
        cfg.start_loc = 0
        cfg.start_time = 0
        cfg.block_size = cfg.max_traj_len
    elif cfg.task == 'trajgen':
        cfg.use_start_loc = True
        cfg.vocab_size = 40002
        cfg.start_loc = 40001
        cfg.start_time = 49
        cfg.time_size = 50
        cfg.max_traj_len = 48
        cfg.test_traj_len = 40
        cfg.block_size = 48
        cfg.crop_by_day = True
        cfg.make_heatmaps = True
        cfg.data_path = 'data/trajs/trajgen'
    else:
        raise Exception('invalid task passed!')

    test_H0xtra(cfg)