# partially adapted from https://github.com/karpathy/nanoGPT, (copyright (c) 2022 Andrej Karpathy)

"""
This file contains the multi-city training routine, as well as validation and testing
"""

import os
import math
from contextlib import nullcontext
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch._dynamo

from model.transformer import GPTConfig
from model.hypernetwork import HyperNetworkConfig
from model.h0xtra import H0xtra
from data.provider_hyper import data_provider_hyper
from utils import partial_metrics_sum, get_top_k, append_dict, set_seed, create_heatmaps, distance_versus_probability, generate_and_test_fixed_emb

os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
torch._dynamo.config.suppress_errors = True
torch.autograd.set_detect_anomaly(True)


def train_H0xtra(cfg):

    config_dict = vars(cfg)

    out_dir = os.path.join(cfg.out_dir, cfg.task, 'train_cities' + str(cfg.train_cities))
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    best_val_loss = (1e9, 0)
    device = cfg.device

    # model init
    gpt_model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size,
                    bias=cfg.bias, vocab_size=cfg.vocab_size, dropout=cfg.dropout, time_size=cfg.time_size) # start with model_args from command line
    hyper_model_args = dict(n_layers=cfg.h_n_layers, in_channels=cfg.h_in_channels, d_embed=cfg.h_d_embed, kernel_size=cfg.h_kernel_size, hypernetwork=cfg.h_net, crop_by_day=cfg.crop_by_day, coords=cfg.h_coords)


    gptconf = GPTConfig(**gpt_model_args)
    hyperconf = HyperNetworkConfig(**hyper_model_args)

    model = H0xtra(hyperconf, gptconf, task=cfg.task)
    model.to(device)

    # optimizer
    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
    checkpoint = None # free up memory

    # compile the model
    if cfg.compile:
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    @torch.no_grad()
    def compute_val_loss(loader):
        model.eval()
        losses = torch.zeros(len(loader))
        results = {}
        for k, data in enumerate(tqdm(loader)):
            X, Y, lngths, maps, cities = data
            X = X.to(device)
            Y = Y.to(device)
            maps = maps.to(device)
            lngths = lngths.to(device)
            with ctx:
                logits, targets, loss = model.generate(X, maps, Y, test_last=cfg.test_only_last, lengths=lngths)
            losses[k] = loss.item()
            append_dict(results, get_top_k(logits, targets, padding_value=0))
        result = partial_metrics_sum(results)
        model.train()
        return losses.mean(), result

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < cfg.warmup_iters:
            return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.lr_decay_iters:
            return cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


    train_loader, val_loader, test_loader = data_provider_hyper(cfg.data_path, cfg.matrix_path, shuffle=True, batch_size=cfg.batch_size, train_cities=cfg.train_cities, test_cities=cfg.test_cities, max_traj_len=cfg.max_traj_len, test_traj_len=cfg.test_traj_len, use_start_loc=cfg.use_start_loc, start_loc=cfg.start_loc, start_time=cfg.start_time, crop_by_day=cfg.crop_by_day, workers=cfg.workers)
    train_losses = []
    val_losses = []
    val_mrr = []

    # update optimizer config
    cfg.lr_decay_iters = cfg.epochs * len(train_loader)
    cfg.warmup_iters = 0.01 * cfg.lr_decay_iters
    cfg.min_lr = cfg.learning_rate/10
    iter_num = 0

    # start of actual training
    for epoch in range(cfg.epochs):

        train_loss = []
        optimizer.zero_grad()

        for i, data in enumerate(tqdm(train_loader)):

            lr = get_lr(iter_num) if cfg.decay_lr else cfg.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            iter_num += 1

            X, Y, lngths, maps, cities = data
            X = X.to(device)
            Y = Y.to(device)
            maps = maps.to(device)

            logits, loss = model(X, maps, Y)

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_losses += train_loss

        val_loss, result = compute_val_loss(val_loader)
        print(f'For epoch {epoch}, train loss is {sum(train_loss)/len(train_loss)} and val loss is {val_loss}')
        val_losses.append(val_loss)
        val_mrr.append(result['mrr'])
        if val_loss < best_val_loss[0]:
            best_val_loss = (val_loss, epoch)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': (gpt_model_args, hyper_model_args),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'train_loss': train_losses,
                'val_loss': val_losses,
                'full_dict': config_dict,
                'results_val': result
            }
            
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # Test model on test cities
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), weights_only=False)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()
    cfg.test_cities = eval(cfg.test_cities) if type(cfg.test_cities) == str else cfg.test_cities
    
    for city in cfg.test_cities:

        print(f'Testing on city {city} with model from epoch {checkpoint['epoch']}')

        test_data, test_loader = data_provider_hyper(cfg.data_path, cfg.matrix_path, shuffle=True, batch_size=128, train_cities=cfg.train_cities, test_cities=[city], max_traj_len=cfg.max_traj_len, test_traj_len=cfg.test_traj_len, use_start_loc=cfg.use_start_loc, crop_by_day=cfg.crop_by_day, eval=True, workers=cfg.workers)
        
        if cfg.task == 'nextloc':

            losses = torch.zeros(len(test_loader))
            results = {}
            probvdist = []
            bin_edges = torch.linspace(0, 142000, steps=100).to(device) # bins for probability mass vs distance (max distance is ~142000m)
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
                probvdist.append(distance_versus_probability(logits.reshape(-1, logits.shape[-1]), targets, bin_edges))
            result = partial_metrics_sum(results)
            probvdist = np.mean(np.array(probvdist), axis=0)

            overall_results = {
            'test_loss': losses,
            'result': result,
            'probvdist': probvdist
        }

        if cfg.task == 'trajgen':

            gen_data, result = generate_and_test_fixed_emb(model, test_loader, device=cfg.device, start_token_loc=cfg.start_loc, start_token_time=cfg.start_time, batch_size=cfg.batch_size, path=out_dir)

            if cfg.make_heatmaps:
                    hm = create_heatmaps([gen_data.item()['gt'], gen_data.item()['gen']]) # create heatmaps from trajectories
            else:
                hm = None

            overall_results = {
                'full_dict': vars(cfg),
                'result': result,
                'heatmaps': hm,
            }

        np.save(os.path.join(out_dir, f'test_results{city}.npy'), np.array(overall_results))

    if len(cfg.test_cities) > 1:
        # load all test results and average them across test cities
        overall_results = {}
        for city in cfg.test_cities:
            test_results = np.load(os.path.join(out_dir, f'test_results{city}.npy'), allow_pickle=True).item()
            test_results['city'] = city
            append_dict(overall_results, test_results)
        
        if cfg.task == 'nextloc':
            overall_results['test_loss'] = np.mean([np.mean(overall_results['test_loss'][i].numpy()) for i in range(len(cfg.test_cities))])
            result_o = {}
            for k in [1, 5, 10, 20]:
                result_o[k] = np.mean(np.array([overall_results['result'][i][k] for i in range(len(cfg.test_cities))]), axis=0)
            result_o['mrr'] = np.mean(np.array([overall_results['result'][i]['mrr'] for i in range(len(cfg.test_cities))]), axis=0)
            result_o['distances'] = np.mean(np.array([overall_results['result'][i]['distances'] for i in range(len(cfg.test_cities))]), axis=0)
            overall_results['result'] = result_o
            overall_results['probvdist'] = np.mean(np.array([overall_results['probvdist'][i] for i in range(len(cfg.test_cities))]), axis=0)
        elif cfg.task == 'trajgen':
            result_o = {}
            result_o['distance'] = np.mean(np.array([overall_results['result'][i]['distance'] for i in range(len(cfg.test_cities))]), axis=0)
            result_o['radius'] = np.mean(np.array([overall_results['result'][i]['radius'] for i in range(len(cfg.test_cities))]), axis=0)
            result_o['duration'] = np.mean(np.array([overall_results['result'][i]['duration'] for i in range(len(cfg.test_cities))]), axis=0)
            result_o['periodicity'] = np.mean(np.array([overall_results['result'][i]['periodicity'] for i in range(len(cfg.test_cities))]), axis=0)
            result_o['G-rank'] = np.mean(np.array([overall_results['result'][i]['G-rank'] for i in range(len(cfg.test_cities))]), axis=0)
            result_o['I-rank'] = np.mean(np.array([overall_results['result'][i]['I-rank'] for i in range(len(cfg.test_cities))]), axis=0)
            overall_results['result'] = result_o
            overall_results['heatmaps'] = np.array([overall_results['heatmaps'][i] for i in range(len(cfg.test_cities))])
        np.save(os.path.join(out_dir, 'overall_results.npy'), np.array(overall_results))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # system
    parser.add_argument('--seed',  default=1337, type=int)
    parser.add_argument('--backend', default='nccl', type=str)
    parser.add_argument('--device', default='cuda', type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--workers', default=8, type=str)
    parser.add_argument('--dtype', default='float32', type=str, choices=['float32', 'float16'])
    parser.add_argument('--compile', default=True, type=bool)
    parser.add_argument('--out_dir',  default='results', type=str)

    # data
    parser.add_argument('--matrix_path', default='data/maps/maps.npy', type=str)
    parser.add_argument('--data_path', default='data/trajs', type=str)
    parser.add_argument('--train_cities', default='[1,2,3]', type=str, help='Cities to train on')
    parser.add_argument('--test_cities', default='[4]', type=str, help='City to test on. For zero-shot testing, choose a city not included in train_cities.')

    # training setup
    parser.add_argument('--task', default='trajgen', type=str, choices=['nextloc', 'trajgen'], help='Choose a task: next-location prediction or trajectory generation')

    # optimizer configuration
    parser.add_argument('--learning_rate', default=5e-3, type=float) # 6e-4
    parser.add_argument('--weight_decay', default=1e-1, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--decay_lr', default=True, type=bool)
    parser.add_argument('--warmup_iters', default=1000, type=int)
    parser.add_argument('--lr_decay_iters', default=600000, type=int)
    parser.add_argument('--min_lr', default=6e-5, type=float)

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

    # test settings
    parser.add_argument('--test_only_last', default=True, type=bool, help='Whether next location prediction is evaluated only on the last location of a trajectory')

    cfg = parser.parse_args()
    cfg.train_cities = eval(cfg.train_cities) if type(cfg.train_cities) == str else cfg.train_cities
    cfg.test_cities = eval(cfg.test_cities) if type(cfg.test_cities) == str else cfg.test_cities

    # adjust data to architecture
    if cfg.h_coords == 'c':
        cfg.matrix_path = 'data/maps/maps_coords.npy'

    # adjust config to chosen task
    if cfg.task == 'nextloc':
        cfg.epochs = 20
        cfg.batch_size = 16
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
        cfg.epochs = 10
        cfg.batch_size = 128
        cfg.use_start_loc = True
        cfg.vocab_size = 40002
        cfg.start_loc = 40001
        cfg.start_time = 49
        cfg.time_size = 50
        cfg.max_traj_len = 250
        cfg.test_traj_len = 40
        cfg.block_size = 48
        cfg.crop_by_day = True
        cfg.make_heatmaps = True
        cfg.data_path = 'data/trajs/trajgen'
    else:
        raise Exception('invalid task passed!')

    train_H0xtra(cfg)