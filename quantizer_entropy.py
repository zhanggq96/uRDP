import os
import json
import math
import csv
import argparse
from argparse import Namespace
from copy import deepcopy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from models import *
from utils import *


def entropy(distribution):
    H = 0
    for p in distribution:
        if p > 0:
            H += -p * math.log2(p)

    return H

def entropy_data(data):
    counts = Counter(data)
    N = sum(count for item, count in counts.items())
    distribution = sorted([count / N for item, count in counts.items()]) # Doesn't need to be sorted
    return entropy(distribution) # Maybe use scipy instead

def base_estimate_dim_entropy(eval_args, device):
    # Use model settings
    with open(os.path.join(eval_args.load_base_model_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        train_args = Namespace(**base_model_args)
        latent_dim = train_args.latent_dim_1
        L_1 = train_args.L_1

        assert_args_match(vars(train_args), vars(eval_args), ('latent_dim_1', 'L_1', 'limits', 'Lambda_base'))

    encoder = Encoder1(train_args).to(device)
    decoder = Decoder1(train_args).to(device)
    encoder.eval() # Disable batch norm, or else results inconsistent
    decoder.eval()

    # Load base model to continue from
    assert os.path.isfile(os.path.join(eval_args.load_base_model_path, 'encoder1.ckpt')), \
        f'No file {os.path.join(eval_args.load_base_model_path, "encoder1.ckpt")} found!'
    encoder.load_state_dict(torch.load(os.path.join(eval_args.load_base_model_path, 'encoder1.ckpt')))
    quantization_centers = encoder.q.centers

    if eval_args.train_or_test == 'train':
        dataloader, _, _ = load_dataset('mnist', eval_args.batch_size, eval_args.batch_size, shuffle_train=False)
    else:
        _, dataloader, _ = load_dataset('mnist', eval_args.batch_size, eval_args.batch_size, shuffle_test=False)

    q_all = []

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            centers = x.data.new(quantization_centers)

            # Batch size != latent dim size, so will be broadcast along new dim
            # on the line: dist = torch.abs(z-centers)
            assert x.size(0) != centers.size(0)

            z = encoder.encode(x) # no stochasticity added
            z = encoder.quantize(z)

            zsize = list(z.size())
            z = z.view(*(zsize + [1]))

            dist = torch.abs(z-centers)
            _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
            symbols = symbols.squeeze() # batch_size x z_dim, each entry is the symbol the dim was quantized to

            q_all.append(symbols)

    q_all = torch.cat(q_all, dim=0)
    h_by_dim = []
    for l in range(eval_args.latent_dim_1):
        q = [t.item() for t in q_all[:,l]]
        h = entropy_data(q)
        h_by_dim.append(h)

    if os.path.isfile(os.path.join(eval_args.experiment_path, '_entropy.json')):
        with open(os.path.join(eval_args.experiment_path, '_entropy.json'), 'r') as f:
            data = json.load(f)
            data['model_entropy_by_dim'] = h_by_dim
    else:
        data = {'model_entropy_by_dim': h_by_dim}

    with open(os.path.join(eval_args.experiment_path, '_entropy.json'), 'w') as f:
        json.dump(data, f)

    return h_by_dim

def base_estimate_entropy(eval_args, device):
    # Helper for unravelling argparse and loading model
    # Use model settings
    with open(os.path.join(eval_args.load_base_model_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        train_args = Namespace(**base_model_args)
        latent_dim_1 = train_args.latent_dim_1
        L_1 = train_args.L_1
        Lambda_base = train_args.Lambda_base

        assert_args_match(vars(train_args), vars(eval_args), ('latent_dim_1', 'L_1', 'limits', 'Lambda_base', 'enc_layer_scale'))

    encoder1 = Encoder1(train_args).to(device)
    decoder1 = Decoder1(train_args).to(device) # Ensure decoder loads correctly
    encoder1.eval() # Disable batch norm, or else results inconsistent
    decoder1.eval()

    assert os.path.isfile(os.path.join(eval_args.load_base_model_path, 'encoder1.ckpt')), \
        f'No file {os.path.join(eval_args.load_base_model_path, "encoder1.ckpt")} found!'
    encoder1.load_state_dict(torch.load(os.path.join(eval_args.load_base_model_path, 'encoder1.ckpt')))

    data = base_estimate_entropy_(encoder1, latent_dim_1, L_1, Lambda_base, eval_args.train_or_test,
                                  eval_args.batch_size, eval_args.experiment_path, eval_args.dataset, device)

    return data

def base_estimate_entropy_(encoder1, latent_dim_1, L_1, Lambda_base, train_or_test,
                           batch_size, experiment_path, dataset, device):
    # Estimate entropy of base model. Since it takes the model as an arg,
    # this can be called during training time periodically.
    # Load base model to continue from
    quantization_centers = encoder1.q.centers

    if train_or_test == 'train':
        dataloader, _, _ = load_dataset(dataset, batch_size, batch_size, shuffle_train=False)
    elif train_or_test == 'test': # Avoid using test set
        _, dataloader, _ = load_dataset(dataset, batch_size, batch_size, shuffle_test=False)

    q_all = []

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            centers = x.data.new(quantization_centers)

            # Batch size != latent dim size, so will be broadcast along new dim
            # on the line: dist = torch.abs(z-centers)
            assert x.size(0) != centers.size(0)

            z = encoder1.encode(x) # no stochasticity added
            z = encoder1.quantize(z)

            zsize = list(z.size())
            z = z.view(*(zsize + [1]))

            dist = torch.abs(z-centers)
            _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
            symbols = symbols.squeeze() # batch_size x z_dim, each entry is the symbol the dim was quantized to

            bs = x.size(0)
            q = torch.zeros(bs, dtype=torch.long).to(device)
            for d in range(latent_dim_1):
                q += symbols[:,d] * pow(L_1, d)
            q_all.append(q)

    q_all = list(torch.cat(q_all, dim=0))
    q_all = [t.item() for t in q_all]
    h = entropy_data(q_all)
    print(f'{latent_dim_1}-{L_1} Lambda={Lambda_base} Entropy: {h}')

    if os.path.isfile(os.path.join(experiment_path, '_entropy.json')):
        with open(os.path.join(experiment_path, '_entropy.json'), 'r') as f:
            data = json.load(f)
            data['model_entropy'] = h
    else:
        data = {'model_entropy': h}

    with open(os.path.join(experiment_path, '_entropy.json'), 'w') as f:
        json.dump(data, f)

    return data

def refined_estimate_entropy(eval_args, device):
    # Use model settings
    with open(os.path.join(eval_args.load_base_model_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        train_args_1 = Namespace(**base_model_args)
        latent_dim_1 = train_args_1.latent_dim_1
        L_1 = train_args_1.L_1
        Lambda_base = train_args.Lambda_base

        assert_args_match(vars(train_args_1), vars(eval_args), ('latent_dim_1', 'L_1', 'limits', 'Lambda_base', 'enc_layer_scale'))

    with open(os.path.join(eval_args.load_refined_model_path, '_settings.json'), 'r') as f:
        refined_model_args = json.load(f)
        train_args_2 = Namespace(**refined_model_args)
        latent_dim_2 = train_args_2.latent_dim_2
        L_2 = train_args_2.L_2
        Lambda_refined = train_args.Lambda_refined

        assert_args_match(vars(train_args_2), vars(eval_args), ('latent_dim_1', 'L_1', 'limits', 'Lambda_base',
                                                                'latent_dim_2', 'L_2', 'limits', 'Lambda_refined',
                                                                'enc_2_layer_scale'))

    encoder1 = Encoder1(train_args_1).to(device)
    decoder1 = Decoder1(train_args_1).to(device)
    encoder1.eval() # Disable batch norm, or else results inconsistent
    decoder1.eval()
    encoder2 = Encoder2(train_args_2).to(device)
    decoder2 = Decoder2(train_args_2).to(device)
    encoder2.eval()
    decoder2.eval()

    # Load base model to continue from
    assert os.path.isfile(os.path.join(eval_args.load_base_model_path, 'encoder1.ckpt')), \
        f'No file {os.path.join(eval_args.load_base_model_path, "encoder1.ckpt")} found!'
    encoder1.load_state_dict(torch.load(os.path.join(eval_args.load_base_model_path, 'encoder1.ckpt')))
    # Load refined model
    assert os.path.isfile(os.path.join(eval_args.load_refined_model_path, 'encoder2.ckpt')), \
        f'No file {os.path.join(eval_args.load_refined_model_path, "encoder2.ckpt")} found!'
    encoder2.load_state_dict(torch.load(os.path.join(eval_args.load_refined_model_path, 'encoder2.ckpt')))

    data = refined_estimate_entropy_(encoder1, latent_dim_1, L_1, Lambda_base,
                                     encoder2, latent_dim_2, L_2, Lambda_refined,
                                     eval_args.train_or_test, eval_args.batch_size,
                                     eval_args.experiment_path, eval_args.dataset, device)

    return data

def refined_estimate_entropy_(encoder1, latent_dim_1, L_1, Lambda_base,
                              encoder2, latent_dim_2, L_2, Lambda_refined,
                              train_or_test, batch_size, experiment_path, dataset, device):

    quantization_centers_1 = encoder1.q.centers
    quantization_centers_2 = encoder2.q.centers

    if train_or_test == 'train':
        dataloader, _, _ = load_dataset(dataset, batch_size, batch_size, shuffle_train=False)
    elif train_or_test == 'test': # Avoid using test set
        _, dataloader, _ = load_dataset(dataset, batch_size, batch_size, shuffle_test=False)

    q_all = []

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            centers_1 = x.data.new(quantization_centers_1)
            centers_2 = x.data.new(quantization_centers_2)
            centers = torch.cat((centers_1, centers_2), dim=0)

            # Batch size != latent dim size, so will be broadcast along new dim
            # on the line: dist = torch.abs(z-centers)
            assert x.size(0) != centers.size(0)

            z1 = encoder1.encode(x) # no stochasticity added
            z1 = encoder1.quantize(z1)
            z2 = encoder2.encode(x) # no stochasticity added
            z2 = encoder2.quantize(z2)

            z1size = list(z1.size())
            z1 = z1.view(*(z1size + [1]))
            z2size = list(z2.size())
            z2 = z2.view(*(z2size + [1]))
            z = torch.cat((z1, z2), dim=1)

            dist = torch.abs(z-centers)
            _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
            symbols = symbols.squeeze() # batch_size x z_dim, each entry is the symbol the dim was quantized to

            # Use max(base_1,base_2) for the base - things with zero count do not contribute to entropy
            base = max(L_1, L_2)
            bs = x.size(0)
            q = torch.zeros(bs, dtype=torch.long).to(device)
            for d in range(latent_dim_1): # Final value of d = (latent_dim_1-1)
                q += symbols[:,d] * pow(base, d)
            for d2 in range(d+1, d+1 + latent_dim_2):
                q += symbols[:,d2] * pow(base, d2)
            q_all.append(q)

    q_all = list(torch.cat(q_all, dim=0))
    q_all = [t.item() for t in q_all]
    h = entropy_data(q_all)
    # print(f'{train_args_1.latent_dim_1}-{train_args_1.L_1}-{train_args_2.latent_dim_2}-{train_args_2.L_2} Entropy: {h}')
    print(f'{latent_dim_1}-{L_1} Lambda={Lambda_base} {latent_dim_2}-{L_2} Lambda={Lambda_refined} Entropy: {h}')

    if os.path.isfile(os.path.join(experiment_path, '_entropy.json')):
        with open(os.path.join(experiment_path, '_entropy.json'), 'r') as f:
            data = json.load(f)
            data['model_entropy'] = h
    else:
        data = {'model_entropy': h}

    with open(os.path.join(experiment_path, '_entropy.json'), 'w') as f:
        json.dump(data, f)

    return data

# # entropy for reduced models
# def reduced_estimate_entropy(eval_args, device):
#     # Shares same decoder as base model. Only works for same dimension reduction right now
#     assert eval_args.latent_dim_0 == eval_args.latent_dim_1 and eval_args.L_0 == eval_args.L_1
#     return base_estimate_entropy(eval_args, device)

def reduced_estimate_entropy_(encoder1, latent_dim_1, L_1, Lambda_base,
                              latent_dim_0, L_0, Lambda_reduced,
                              train_or_test, batch_size, experiment_path, dataset, device):
    # Shares same decoder as base model. Only works for same dimension reduction right now
    assert latent_dim_0 == latent_dim_1 and L_0 == L_1
    return base_estimate_entropy_(encoder1, latent_dim_1, L_1, Lambda_base, train_or_test,
                                  batch_size, experiment_path, dataset, device)

def reduced_estimate_entropy(eval_args, device):
    # Use model settings
    with open(os.path.join(eval_args.load_base_model_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        train_args_1 = Namespace(**base_model_args)
        latent_dim_1 = train_args_1.latent_dim_1
        L_1 = train_args_1.L_1
        Lambda_base = train_args_1.Lambda_base

        assert_args_match(vars(train_args_1), vars(eval_args), ('latent_dim_1', 'L_1', 'limits', 'Lambda_base', 'enc_layer_scale'))

    with open(os.path.join(eval_args.load_reduced_model_path, '_settings.json'), 'r') as f:
        reduced_model_args = json.load(f)
        train_args_0 = Namespace(**reduced_model_args)
        latent_dim_0 = train_args_0.latent_dim_0
        L_0 = train_args_0.L_0
        Lambda_reduced = train_args_0.Lambda_reduced

        assert_args_match(vars(train_args_0), vars(eval_args), ('latent_dim_1', 'L_1', 'limits', 'Lambda_base',
                                                                'latent_dim_0', 'L_0', 'limits', 'Lambda_reduced'))

    encoder1 = Encoder1(train_args_1).to(device)
    encoder1.eval() # Disable batch norm, or else results inconsistent

    # Load base model to continue from
    assert os.path.isfile(os.path.join(eval_args.load_base_model_path, 'encoder1.ckpt')), \
        f'No file {os.path.join(eval_args.load_base_model_path, "encoder1.ckpt")} found!'
    encoder1.load_state_dict(torch.load(os.path.join(eval_args.load_base_model_path, 'encoder1.ckpt')))

    data = reduced_estimate_entropy_(encoder1, latent_dim_1, L_1, Lambda_base,
                                     latent_dim_0, L_0, Lambda_reduced,
                                     eval_args.train_or_test, eval_args.batch_size,
                                     eval_args.experiment_path, eval_args.dataset, device)

    return data


def extract_entropy_all(eval_args, device):
    all_model_dirs = get_model_dirs(eval_args.experiment_path, filtering=None)
    entropies = []

    for i, model_dir in enumerate(all_model_dirs):
        print(model_dir)
        mode = None
        with open(os.path.join(model_dir, '_settings.json'), 'r') as f1, \
             open(os.path.join(model_dir, '_entropy.json'), 'r') as f2:
            model_args = json.load(f1)
            data = json.load(f2)
            model_args = Namespace(**model_args)
            if model_args.mode == 'refined':
                mode = 'refined'
                print_string = (f'Refined: {model_args.latent_dim_1}-{model_args.L_1} Lambda={model_args.Lambda_base} '
                                f'{model_args.latent_dim_2}-{model_args.L_2} Lambda={model_args.Lambda_refined} '
                                f'Entropy: {data["model_entropy"]}\n')
                q_rate = calculate_rate(mode, model_args.latent_dim_1, model_args.L_1,
                                        latent_dim_2=model_args.latent_dim_2, L_2=model_args.L_2)
            elif model_args.mode == 'reduced':
                mode = 'reduced'
                print_string = (f'Reduced: {model_args.latent_dim_1}-{model_args.L_1} Lambda={model_args.Lambda_base} '
                                f'{model_args.latent_dim_0}-{model_args.L_0} Lambda={model_args.Lambda_reduced} '
                                f'Entropy: {data["model_entropy"]}\n')
                q_rate = calculate_rate(mode, model_args.latent_dim_1, model_args.L_1,
                                        latent_dim_0=model_args.latent_dim_0, L_0=model_args.L_0)
            elif model_args.mode == 'base':
                mode = 'base'
                print_string = (f'Base: {model_args.latent_dim_1}-{model_args.L_1} Lambda={model_args.Lambda_base} '
                                f'Entropy: {data["model_entropy"]}\n')
                q_rate = calculate_rate(mode, model_args.latent_dim_1, model_args.L_1)

        print(print_string)

        entropy_data = {
            'quantizer_rate': q_rate,
            'model_rate': data["model_entropy"],
            'discrepency': data["model_entropy"]/q_rate,

            'latent_dim_1': model_args.latent_dim_1,
            'L_1': model_args.L_1,
            'Lambda_base': model_args.Lambda_base,

            'latent_dim_2': model_args.latent_dim_2,
            'L_2': model_args.L_2,
            'Lambda_refined': model_args.Lambda_refined,

            'latent_dim_0': model_args.latent_dim_0,
            'L_0': model_args.L_0,
            'Lambda_reduced': model_args.Lambda_reduced,
        }
        entropies.append(entropy_data)

    with open(os.path.join(eval_args.experiment_path, '_model_entropies.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([
            'quantizer_rate', 'model_rate', 'discrepency',
            'latent_dim_1', 'L_1', 'Lambda_base',
            'latent_dim_2', 'L_2', 'Lambda_refined',
            'latent_dim_0', 'L_0', 'Lambda_reduced',
        ])
        for row in entropies:
            writer.writerow([
                row['quantizer_rate'], row['model_rate'], row['discrepency'],
                row['latent_dim_1'], row['L_1'], row['Lambda_base'],
                row['latent_dim_2'], row['L_2'], row['Lambda_refined'],
                row['latent_dim_0'], row['L_0'], row['Lambda_reduced']
            ])

def estimate_entropy_all(eval_args, device):
    all_model_dirs = get_model_dirs(eval_args.experiment_path, filtering=None)
    entropies = []

    for i, model_dir in enumerate(all_model_dirs):
        print(model_dir)
        mode = None
        with open(os.path.join(model_dir, '_settings.json'), 'r') as f:
            model_args = json.load(f)
            if model_args['mode'] == 'refined':
                mode = 'refined'
                model_args['load_refined_model_path'] = model_args['experiment_path']
            elif model_args['mode'] == 'reduced':
                mode = 'reduced'
                model_args['load_reduced_model_path'] = model_args['experiment_path']
            elif model_args['mode'] == 'base':
                mode = 'base'
                model_args['load_base_model_path'] = model_args['experiment_path']

            # These are not saved or essential, so it is fine to overwrite them
            model_args['batch_size'] = eval_args.batch_size
            model_args['train_or_test'] = eval_args.train_or_test
            model_args = Namespace(**model_args)

        if mode == 'base':
            # continue
            print('Calculating entropy for base model...')
            # print(model_args.load_base_model_path)
            data = base_estimate_entropy(model_args, device)
            print_string = (f'Base: {model_args.latent_dim_1}-{model_args.L_1} Lambda={model_args.Lambda_base} '
                            f'Entropy: {data["model_entropy"]}\n')
            q_rate = calculate_rate(mode, model_args.latent_dim_1, model_args.L_1)
        elif mode == 'refined':
            # continue
            print('Calculating entropy for refinement model...')
            data = refined_estimate_entropy(model_args, device)
            print_string = (f'Refined: {model_args.latent_dim_1}-{model_args.L_1} Lambda={model_args.Lambda_base} '
                            f'{model_args.latent_dim_2}-{model_args.L_2} Lambda={model_args.Lambda_refined} '
                            f'Entropy: {data["model_entropy"]}\n')
            q_rate = calculate_rate(mode, model_args.latent_dim_1, model_args.L_1,
                                    latent_dim_2=model_args.latent_dim_2, L_2=model_args.L_2)
        elif mode == 'reduced':
            print('Calculating entropy for reduced model...')
            data = reduced_estimate_entropy(model_args, device)
            print_string = (f'Reduced: {model_args.latent_dim_1}-{model_args.L_1} Lambda={model_args.Lambda_base} '
                            f'{model_args.latent_dim_0}-{model_args.L_0} Lambda={model_args.Lambda_reduced} '
                            f'Entropy: {data["model_entropy"]}\n')
            q_rate = calculate_rate(mode, model_args.latent_dim_1, model_args.L_1,
                                    latent_dim_0=model_args.latent_dim_0, L_0=model_args.L_0)

        entropy_data = {
            'quantizer_rate': q_rate,
            'model_rate': data["model_entropy"],
            'discrepency': data["model_entropy"]/q_rate,

            'latent_dim_1': model_args.latent_dim_1,
            'L_1': model_args.L_1,
            'Lambda_base': model_args.Lambda_base,

            'latent_dim_2': model_args.latent_dim_2,
            'L_2': model_args.L_2,
            'Lambda_refined': model_args.Lambda_refined,

            'latent_dim_0': model_args.latent_dim_0,
            'L_0': model_args.L_0,
            'Lambda_reduced': model_args.Lambda_reduced,
        }
        entropies.append(entropy_data)
        print(print_string)
        # if i > 0: break

    with open(os.path.join(eval_args.experiment_path, '_model_entropies.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([
            'quantizer_rate', 'model_rate', 'discrepency',
            'latent_dim_1', 'L_1', 'Lambda_base',
            'latent_dim_2', 'L_2', 'Lambda_refined',
            'latent_dim_0', 'L_0', 'Lambda_reduced',
        ])
        for row in entropies:
            writer.writerow([
                row['quantizer_rate'], row['model_rate'], row['discrepency'],
                row['latent_dim_1'], row['L_1'], row['Lambda_base'],
                row['latent_dim_2'], row['L_2'], row['Lambda_refined'],
                row['latent_dim_0'], row['L_0'], row['Lambda_reduced']
            ])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=5000, help="base, refined or reduced training mode")
    parser.add_argument("--train_or_test", type=str, default='train', help="base, refined or reduced training mode")
    parser.add_argument("--latent_dim_1", default=-1, type=int, help="Load pretrained base model for selection")
    parser.add_argument("--latent_dim_2", type=int, default=-1, help="dimensionality of the latent space for refinement model")
    parser.add_argument("--latent_dim_0", type=int, default=-1, help="dimensionality of the latent space for reduced model")
    parser.add_argument("--L_1", default=-1, type=int, help="Load pretrained base model for selection")
    parser.add_argument('--L_2', type=int, default=-1, help='number of quantization levels for refined model (default: -1)')
    parser.add_argument('--L_0', type=int, default=-1, help='number of quantization levels for reduced model (default: -1)')
    parser.add_argument("--Lambda_base", default=0, type=float, help="Lambda for base model")
    parser.add_argument("--Lambda_refined", type=float, default=-1, help="coefficient for perception loss for training refined model (default: -1)")
    parser.add_argument("--Lambda_reduced", type=float, default=-1, help="coefficient for perception loss for training reduced model (default: -1)")
    parser.add_argument('--limits', nargs=2, type=float, default=[-1,1], help='quanitzation limits (default: (-1,1))')
    parser.add_argument("--load_base_model_path", default=None, type=str, help="Load pretrained base model for selection")
    parser.add_argument("--load_refined_model_path", default=None, type=str, help="Load pretrained refined model for selection")
    parser.add_argument("--load_reduced_model_path", default=None, type=str, help="Load pretrained reduced model for selection")
    parser.add_argument("--experiment_path", default=None, type=str, help="Where to save")
    parser.add_argument("-mode", type=str, default='base', help="base, refined or reduced training mode")
    parser.add_argument("-method", type=str, default='joint', help="Method (joint entropy or dimensionwise)")
    args = parser.parse_args()

    if args.experiment_path is None:
        if args.mode == 'base':
            args.experiment_path = args.load_base_model_path
        elif args.mode == 'refined':
            args.experiment_path = args.load_refined_model_path
        elif args.mode == 'reduced':
            args.experiment_path = args.load_reduced_model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.method == 'joint':
        if args.mode == 'base':
            print('Calculating entropy for base model...')
            base_estimate_entropy(args, device)
        elif args.mode == 'refined':
            print('Calculating entropy for refinement model...')
            refined_estimate_entropy(args, device)
        elif args.mode == 'reduced':
            print('Calculating entropy for reduced model...')
            reduced_estimate_entropy(args, device)
    elif args.method == 'dim':
        base_estimate_dim_entropy(args, device)
    elif args.method == 'all_joint':
        estimate_entropy_all(args, device)
    elif args.method == 'all_joint_extract':
        extract_entropy_all(args, device)
