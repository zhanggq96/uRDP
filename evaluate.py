import random
import argparse
import os
import sys
import math
import json
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_rdp(xs, ys, colors, savepath, texts=None, colorbar='rate', line_segments='off',
             lines=None, line_colors=None, plot_settings=None, climits=None, rdp_name_suffix=''):
    """
    plot_settings: a list of dicts, corresponding to settings for each point
    """
    fig, ax = plt.subplots()
    xmin, xmax = ax.get_xlim()
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    plt.xlabel('MSE Distortion Loss')
    plt.ylabel('Wasserstein-1 Perception Loss')
    # plt.title('Rate-Distortion-Perception Tradeoff')
    if isinstance(texts, list):
        for xi, yi, text in zip(xs, ys, texts):
            plt.annotate(f'{text}', (xi, yi))

    plot_settings_dict = defaultdict(list)
    if plot_settings is not None:
        for setting in plot_settings:
            for key, value in setting.items():
                plot_settings_dict[key].append(value)

    print(list((key, len(value)) for key, value in plot_settings_dict.items()))
    # assert all values specified by checking whether all lengths equal
    assert len(set([len(value) for key, value in plot_settings_dict.items()])) == 1
    assert all(item not in plot_settings_dict.items() for item in ('s', 'c', 'edgecolors'))

    if line_segments == 'refined':
        # the lines are formatted ((xa_1,ya_1),(xb_1,yb_1),...) right now
        # but they need to be reformatted as (xa_1,xb_1,xa_2,xb_2,...), (ya_1,yb_1,ya_2,yb_2,...)
        xl, yl = reformat_lines(lines)
        c = 0
        for idx in range(0, len(xl), 2):
            if line_colors is not None:
                color = line_colors[c]
                c += 1
            else:
                color = 'silver'
            plt.plot(xl[idx:idx+2], yl[idx:idx+2], color=color, lw=1.2, zorder=0)

    plt.scatter(xs, ys, s=32, c=colors, cmap='plasma', **plot_settings_dict)
    plt.colorbar().set_label(colorbar)
    if climits[0] is not None and climits[1] is not None:
        assert climits[0] <= min(colors) and climits[1] >= max(colors)
        plt.clim(vmin=climits[0], vmax=climits[1])

    name = f'RDP_Tradeoff_{rdp_name_suffix}.pdf' if rdp_name_suffix else 'RDP_Tradeoff.pdf'
    plt.savefig(os.path.join(savepath, name), bbox_inches='tight')

def define_line_segments():
    pass

def plot_training_losses_(experiment_dir, d_avg, p_avg):
    with open(f'{experiment_dir}/_losses.csv', 'r') as f, \
         open(f'{experiment_dir}/_settings.json', 'r') as fs:
        settings = json.load(fs)
        reader = list(csv.reader(f))
        distortion_losses_training, perception_losses_training = [], []
        for i, line in enumerate(reader[1:]):
            line = list(line)
            if len(line) == 2:
                distortion_losses_training.append(float(line[0]))
                perception_losses_training.append(float(line[1]))
            else:
                distortion_losses_training.append(float(line[1]))
                perception_losses_training.append(float(line[2]))
        plot_training_losses(range(len(reader[1:])), distortion_losses_training,
                                perception_losses_training, experiment_dir, settings,
                                d_avg, p_avg)
        with open(f'{experiment_dir}/_losses_backup.csv', 'w') as f2:
            f.seek(0)
            for line in f.readlines():
                f2.write(line)

def plot_training_losses(epoches, distortion, perception, experiment_dir, settings,
                         d_avg, p_avg):
    """
    Source is RDP paper authors' implementation
    """
    fig3 = plt.figure(figsize=(10,4))

    plt.subplot(1, 2, 1)
    plt.plot(epoches, distortion, label='Distortion')
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epoches, perception, label='Perception')
    plt.legend()
    plt.xlabel('Epoch')

    if settings['mode'] == 'base':
        rate = -1
        if settings['L_1'] > 0:
            rate = settings['latent_dim_1'] * np.log2(settings['L_1'])

        fig3.suptitle('Distortion = %.3f, Rate = %.3f, $d_W$ = %.3f' % (d_avg, rate, p_avg))
        fig3.savefig(f'{experiment_dir}/losses.png')
    elif settings['mode'] == 'refined':
        rate_1, rate_2 = -1, -1
        if settings['L_1'] > 0 and settings['L_2'] > 0:
            rate_1 = settings['latent_dim_1'] * np.log2(settings['L_1'])
            rate_2 = settings['latent_dim_2'] * np.log2(settings['L_2'])

        fig3.suptitle('Distortion = %.3f, Rate 1 = %.3f, Rate 2 = %.3f, $d_W$ = %.3f' % (d_avg, rate_1, rate_2, p_avg))
        fig3.savefig(f'{experiment_dir}/losses.png')
    else: # mode == reduced
        rate_1, rate_0 = -1, -1
        if settings['L_1'] > 0 and settings['L_0'] > 0:
            rate_1 = settings['latent_dim_1'] * np.log2(settings['L_1'])
            rate_0 = settings['latent_dim_0'] * np.log2(settings['L_0'])

        fig3.suptitle('Distortion = %.3f, Rate 1 = %.3f, Rate 0 = %.3f, $d_W$ = %.3f' % (d_avg, rate_1, rate_0, p_avg))
        fig3.savefig(f'{experiment_dir}/losses.png')

    plt.close('all')

def eval_mode_two(eval_args):
    """
    Average training losses from last 5 iterations
    """
    experiment_dirs = get_model_dirs(eval_args.experiment_path_pre)

    n_losses = eval_args.n_losses
    ds, ps, bits, latent_dims, Ls, Lambdas, Lambda_bases = [], [], [], [], [], [], []
    plot_settings = []
    base_points, multistage_points = {}, {}

    print(experiment_dirs)
    for experiment_dir in experiment_dirs:
        with open(f'{experiment_dir}/_settings.json') as f:
            train_args = json.load(f)
            train_args = dict_to_namedtuple(train_args)
            mode = train_args.mode

            if mode == 'base':
                n_bits = -1
                if train_args.L_1 > 0:
                    n_bits = train_args.latent_dim_1 * math.log2(train_args.L_1)
                latent_dim = train_args.latent_dim_1
                Lambda_text = f'$\lambda_1$={train_args.Lambda_base}'
                L_text = f'$L_1$={train_args.L_1}'
            elif mode == 'refined':
                n_bits = -1
                if train_args.L_1 > 0 and train_args.L_2 > 0:
                    n_bits = train_args.latent_dim_1 * math.log2(train_args.L_1) \
                        + train_args.latent_dim_2 * math.log2(train_args.L_2)
                latent_dim = train_args.latent_dim_1 + train_args.latent_dim_2
                Lambda_text = f'$\lambda_1$={train_args.Lambda_base} \n $\lambda_2$={train_args.Lambda_refined}'
                L_text = f'$L_1$={train_args.L_1} \n $L_2$={train_args.L_2}'
            elif mode == 'reduced':
                n_bits = -1
                if train_args.L_1 > 0 and train_args.L_0 > 0:
                    n_bits = train_args.latent_dim_0 * math.log2(train_args.L_0)
                latent_dim = train_args.latent_dim_0
                Lambda_text = f'$\lambda_1$={train_args.Lambda_base} \n $\lambda_0$={train_args.Lambda_reduced}'
                L_text = f'$L_1$={train_args.L_1} \n $L_0$={train_args.L_0}'
            elif mode == 'joint_reduced':
                n_bits = -1
                if train_args.capacity == 'higher':
                    if train_args.L_1 > 0:
                        n_bits = train_args.latent_dim_1 * math.log2(train_args.L_1)
                    latent_dim = train_args.latent_dim_1
                    Lambda_text = f'$\lambda_1$={train_args.Lambda_base}'
                    L_text = f'$L_1$={train_args.L_1}'
                else: # train_args.capacity == 'lower'
                    if train_args.L_1 > 0 and train_args.L_0 > 0:
                        n_bits = train_args.latent_dim_0 * math.log2(train_args.L_0)
                    latent_dim = train_args.latent_dim_0
                    Lambda_text = f'$\lambda_1$={train_args.Lambda_base} \n $\lambda_0$={train_args.Lambda_reduced}'
                    L_text = f'$L_1$={train_args.L_1} \n $L_0$={train_args.L_0}'
            elif mode == 'joint_refined':
                n_bits = -1
                if train_args.capacity == 'lower':
                    if train_args.L_1 > 0:
                        n_bits = train_args.latent_dim_1 * math.log2(train_args.L_1)
                    latent_dim = train_args.latent_dim_1
                    Lambda_text = f'$\lambda_1$={train_args.Lambda_base}'
                    L_text = f'$L_1$={train_args.L_1}'
                else: # train_args.capacity == 'higher'
                    if train_args.L_1 > 0 and train_args.L_2 > 0:
                        n_bits = train_args.latent_dim_1 * math.log2(train_args.L_1) \
                               + train_args.latent_dim_2 * math.log2(train_args.L_2)
                    latent_dim = train_args.latent_dim_2
                    Lambda_text = f'$\lambda_1$={train_args.Lambda_base} \n $\lambda_2$={train_args.Lambda_refined}'
                    L_text = f'$L_1$={train_args.L_1} \n $L_2$={train_args.L_2}'
            elif mode == 'joint_reduced_reduced':
                if train_args.submode == 'joint_reduced_reduced_1':
                    if train_args.L_M1 > 0:
                        n_bits = train_args.latent_dim_M1 * math.log2(train_args.L_M1)
                    latent_dim = train_args.latent_dim_M1
                    Lambda_text = f'$\lambda_1$={train_args.Lambda_base} \n ' + \
                                  f'$\lambda_0$={train_args.Lambda_reduced} \n ' + \
                                  f'$\lambda_{{-1}}$={train_args.Lambda_reduced_reduced}'
                    L_text = f'$L_{{-1}}$={train_args.L_M1}'
                else:
                    pass

            bits.append(n_bits)
            latent_dims.append(latent_dim)
            Lambdas.append(Lambda_text)
            Lambda_bases.append(train_args.Lambda_base)
            Ls.append(L_text)

            # Use default settings if they exist
            if 'plot_settings' in train_args._fields:
                plot_settings.append(train_args.plot_settings)
            else:
                if mode == 'base':
                    plot_settings.append({'edgecolor': 'black'})
                elif mode == 'refined' or mode == 'reduced':
                    plot_settings.append({'edgecolor': 'white'})
                elif mode == 'joint_reduced':
                    plot_settings.append({'edgecolor': 'gray'})
                elif mode == 'joint_refined':
                    plot_settings.append({'edgecolor': 'blue'})
                elif mode == 'joint_reduced_reduced':
                    plot_settings.append({'edgecolor': 'green'})

        with open(f'{experiment_dir}/_losses.csv') as f:
            losses = f.readlines()
            losses = [line.split(',') for line in losses]

            distortion_loss_avg, perception_loss_avg = 0, 0
            for epoch, distortion_loss, perception_loss in losses[-n_losses:]:
                try:
                    distortion_loss = float(distortion_loss)
                    perception_loss = float(perception_loss)
                except ValueError as ve:
                    print('Directory:', experiment_dir)
                    print('Distortion loss:', distortion_loss)
                    print('Perception loss:', perception_loss)
                    print('Most like scenario is either an experiment was overwritten then cancelled, deleting all _losses.csv data')
                    print(f'Or the number of losses used to evaluate average loss ({n_losses}) is less than the number of recorded losses ({len(losses)})')
                    raise ValueError(ve)

                distortion_loss_avg += distortion_loss / n_losses
                perception_loss_avg += perception_loss / n_losses

            ds.append(distortion_loss_avg)
            ps.append(perception_loss_avg)

            if mode == 'base':
                # compute base point locations in case we want to draw lines
                # connecting refinement models to their base models
                base_points[(train_args.latent_dim_1,train_args.L_1,train_args.Lambda_base)] \
                    = {
                        'location': (distortion_loss_avg, perception_loss_avg),
                    }
            elif mode == 'refined':
                multistage_points[(train_args.latent_dim_2,train_args.L_2,train_args.Lambda_refined,
                                   train_args.latent_dim_1,train_args.L_1,train_args.Lambda_base)] \
                    = {
                        'location': (distortion_loss_avg, perception_loss_avg),
                        'base_point_setting': (train_args.latent_dim_1,train_args.L_1,train_args.Lambda_base)
                    }

        with open(f'{experiment_dir}/_evaluation2.csv', 'w') as f:
            if train_args.mode == 'base':
                latent_dim_2, L_2, Lambda_refined = None, None, None
                f.write('n_bits,distortion_loss,perception_loss,latent_dim_1,latent_dim_2,L_1,L_2,Lambda_base,Lambda_refined\n')
                f.write(f'''
                    {n_bits},{distortion_loss_avg},{perception_loss_avg},{train_args.latent_dim_1},
                    {latent_dim_2},{train_args.L_1},{L_2},{train_args.Lambda_base},
                    {Lambda_refined}
                ''')
            elif train_args.mode == 'fine':
                latent_dim_2 = train_args.latent_dim_2
                L_2 = train_args.L_2
                Lambda_refined = train_args.Lambda_refined

                f.write('n_bits,distortion_loss,perception_loss,latent_dim_1,latent_dim_2,L_1,L_2,Lambda_base,Lambda_refined\n')
                f.write(f'''
                    {n_bits},{distortion_loss_avg},{perception_loss_avg},{train_args.latent_dim_1},
                    {latent_dim_2},{train_args.L_1},{L_2},{train_args.Lambda_base},
                    {Lambda_refined}
                ''')
            else: # mode == reduced
                latent_dim_0 = train_args.latent_dim_0
                L_0 = train_args.L_0
                Lambda_reduced = train_args.Lambda_reduced
                f.write('n_bits,distortion_loss,perception_loss,latent_dim_1,latent_dim_0,L_1,L_0,Lambda_base,Lambda_reduced\n')
                f.write(f'''
                    {n_bits},{distortion_loss_avg},{perception_loss_avg},{train_args.latent_dim_1},
                    {latent_dim_0},{train_args.L_1},{L_0},{train_args.Lambda_base},
                    {Lambda_reduced}
                ''')

        plot_training_losses_(experiment_dir, distortion_loss_avg, perception_loss_avg)
        print(f'Done experiment {experiment_dir}')

    lines, line_settings = [], []

    for setting, details in multistage_points.items():
        base_point = base_points[details['base_point_setting']]
        lines.append((base_point['location'], details['location']))
        line_settings.append(setting) # Records latent_dim_2, L_2, Lambda_2, latent_dim_1, L_1, Lambda_1,

    info = {
        'distortions': ds,
        'perceptions': ps,
        'bits': bits,
        'latent_dims': latent_dims,
        'quantization_levels': Ls,
        'Lambdas': Lambdas,
        'Lambda_bases': Lambda_bases,
        'plot_settings': plot_settings,
        'lines': lines,
        'line_settings': line_settings
    }

    # return ds, ps, bits, latent_dims, Ls, Lambdas
    return info

def plot_entropies():
    # TODO
    pass

def preview_reduced_decoders(eval_args):
    """
    Example:
    python evaluate.py -experiment_path_pre experiments/8 -mode preview_reduced_decoders
        --Lambda_base 0.01 --latent_dim_1 2 --L_1 2 --test_batch_size 8
    """
    assert eval_args.test_batch_size >= 4

    base_subpath = os.path.join(f'{eval_args.latent_dim_1}-{eval_args.L_1}',
                                'Lambda={:.5f}'.format(eval_args.Lambda_base))
    base_path = os.path.join(eval_args.experiment_path_pre, base_subpath)
    print(f'Base path: {base_path}')
    reduced_paths = get_secondary_model_dirs([base_path,], filtering='reduced_only')

    os.makedirs(os.path.join(base_path, '_reduced'), exist_ok=True)

    with open(os.path.join(base_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        base_model_args = dict_to_namedtuple(base_model_args)
        dataset_name = base_model_args.dataset

    _, test_dataloader, unnormalizer = load_dataset(dataset_name, eval_args.test_batch_size,
                                       eval_args.test_batch_size, shuffle_test=eval_args.shuffle_test)

    encoder1 = Encoder1(base_model_args).to(device)
    encoder1.load_state_dict(torch.load(os.path.join(base_path, 'encoder1.ckpt'), map_location=device))
    encoder1.eval()

    x_test, _ = next(iter(test_dataloader))
    x_test = x_test.to(device)
    reconstructions = []
    for reduced_path in reduced_paths:
        print(f'Reduced model: {reduced_path}')
        with open(os.path.join(reduced_path, '_settings.json'), 'r') as f:
            reduced_model_args = json.load(f)
            reduced_model_args = dict_to_namedtuple(reduced_model_args)

        reduced_dims = str_values_to_tensor(reduced_model_args.reduced_dims, delimiter=',').to(device)

        decoder0 = Decoder0(reduced_model_args).to(device)
        decoder0.load_state_dict(torch.load(os.path.join(reduced_path, 'decoder0.ckpt'), map_location=device))
        decoder0.eval()

        with torch.no_grad():
            # Important: Permute over z entries as done in selection phase
            # for each element of the batch
            z = encoder1(x_test).index_select(1, reduced_dims)
            x_test_recon = decoder0(z)
        reconstructions.append(x_test_recon)

    all_images = [x_test,] + reconstructions
    all_images = torch.cat(all_images, dim=0)
    if not os.path.isfile(os.path.join(base_path, '_reduced', 'change_decoder.png')):
        save_image(unnormalizer(all_images), os.path.join(base_path, '_reduced', 'change_decoder.png'),
                   nrow=eval_args.test_batch_size, normalize=True)
    else:
        i = 0
        while os.path.isfile(os.path.join(base_path, '_reduced', f'change_decoder_{i}.png')):
            i += 1
        save_image(unnormalizer(all_images), os.path.join(base_path, '_reduced', f'change_decoder_{i}.png'),
                   nrow=eval_args.test_batch_size, normalize=True)
    # save_image(all_images, os.path.join(base_path, '_reduced'), normalize=True)

def custom_selection(selection, test_dataloader):
    selection = str_values_to_tensor(eval_args.custom_selection)
    assert len(selection) <= eval_args.test_batch_size
    assert eval_args.shuffle_test == 0
    x_test, _ = next(iter(test_dataloader))
    x_test = x_test.index_select(0, selection)

    return x_test

def compare_e2e_vs_universal(eval_args):
    n = eval_args.num_pics
    assert eval_args.test_batch_size >= n*n

    savepath = os.path.join(eval_args.experiment_path_pre,
                            f'{eval_args.latent_dim_1}-{eval_args.L_1}', '_compare_universal')
    os.makedirs(savepath, exist_ok=True)
    print(f'savepath: {savepath}')

    # --------- Evaluate universal models ----------

    main_base_subpath = os.path.join(f'{eval_args.latent_dim_1}-{eval_args.L_1}',
                                     'Lambda={:.5f}'.format(eval_args.Lambda_base))

    main_base_path = os.path.join(eval_args.experiment_path_pre, main_base_subpath)
    print(f'Base path: {main_base_path}')
    reduced_paths = get_secondary_model_dirs([main_base_path,], filtering='reduced_only_same_dim')
    print(f'Reduced paths: {reduced_paths}')

    with open(os.path.join(main_base_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        base_model_args = dict_to_namedtuple(base_model_args)
        dataset_name = base_model_args.dataset

    _, test_dataloader, unnormalizer = load_dataset(dataset_name, eval_args.test_batch_size,
                                       eval_args.test_batch_size, shuffle_test=eval_args.shuffle_test)

    encoder1 = Encoder1(base_model_args).to(device)
    encoder1.load_state_dict(torch.load(os.path.join(main_base_path, 'encoder1.ckpt'), map_location=device))
    encoder1.eval()


    if eval_args.custom_selection is None:
        x_test, _ = next(iter(test_dataloader))
    else:
        x_test = custom_selection(eval_args, test_dataloader)
    x_test = x_test.to(device)

    save_image(unnormalizer(x_test.data[:n*n]), os.path.join(savepath, '_originals.png'), nrow=n)
    for reduced_path in reduced_paths:
        print(f'Reduced path: {reduced_path}')
        with open(os.path.join(reduced_path, '_settings.json'), 'r') as f:
            reduced_model_args = json.load(f)
            reduced_model_args = dict_to_namedtuple(reduced_model_args)

        reduced_dims = str_values_to_tensor(reduced_model_args.reduced_dims, delimiter=',').to(device)

        decoder0 = Decoder0(reduced_model_args).to(device)
        decoder0.load_state_dict(torch.load(os.path.join(reduced_path, 'decoder0.ckpt'), map_location=device))
        decoder0.eval()

        with torch.no_grad():
            # Important: Permute over z entries as done in selection phase
            # for each element of the batch
            u1_test = uniform_noise([x_test.size(0), eval_args.latent_dim_1], encoder1.alpha).to(device)
            z = encoder1(x_test, u1_test).index_select(1, reduced_dims)
            x_test_recon = decoder0(z, u1_test.index_select(1, reduced_dims)) # permute noise

            # Save square grid of images
            save_image(unnormalizer(x_test_recon.data[:n*n]),
                       os.path.join(savepath, 'Lambda={:.5f}-reduced.png' \
                                              .format(reduced_model_args.Lambda_reduced)), nrow=n)

    # --------- Evaluate one stage models ----------

    other_base_Lambda_list = str_values_to_list(eval_args.other_base_Lambda, delimiter=',', dtype=float)
    other_base_paths = [
        os.path.join(eval_args.experiment_path_pre,
                     f'{eval_args.latent_dim_1}-{eval_args.L_1}',
                     'Lambda={:.5f}'.format(other_base_Lambda))
        for other_base_Lambda in other_base_Lambda_list
    ]

    for other_base_path in other_base_paths:
        print(f'Other base path: {other_base_path}')

        with open(os.path.join(other_base_path, '_settings.json'), 'r') as f:
            other_base_model_args = json.load(f)
            other_base_model_args = dict_to_namedtuple(other_base_model_args)

        encoder1 = Encoder1(other_base_model_args).to(device)
        encoder1.load_state_dict(torch.load(os.path.join(other_base_path, 'encoder1.ckpt'), map_location=device))
        encoder1.eval()

        decoder1 = Decoder1(other_base_model_args).to(device)
        decoder1.load_state_dict(torch.load(os.path.join(other_base_path, 'decoder1.ckpt'), map_location=device))
        decoder1.eval()

        with torch.no_grad():
            u1_test = uniform_noise([x_test.size(0), eval_args.latent_dim_1], encoder1.alpha).to(device)
            z = encoder1(x_test, u1_test)
            x_test_recon = decoder1(z, u1_test)

            # Save square grid of images
            save_image(unnormalizer(x_test_recon.data[:n*n]),
                       os.path.join(savepath, 'Lambda={:.5f}-e2e.png' \
                                              .format(other_base_model_args.Lambda_base)), nrow=n)

def compare_e2e_vs_refined(eval_args):
    n = eval_args.num_pics
    assert eval_args.test_batch_size >= n*n

    latent_dim_1_lower, latent_dim_1_higher = eval_args.latent_dim_1s
    L_1_lower, L_1_higher = eval_args.L_1s

    savepath = os.path.join(eval_args.experiment_path_pre,
                            f'{latent_dim_1_lower}-{L_1_lower}', '_compare_refined')
    os.makedirs(savepath, exist_ok=True)
    print(f'savepath: {savepath}')

    # --------- Evaluate refinement models ----------

    main_base_subpath = os.path.join(f'{latent_dim_1_lower}-{L_1_lower}',
                                     'Lambda={:.5f}'.format(eval_args.Lambda_base))

    main_base_path = os.path.join(eval_args.experiment_path_pre, main_base_subpath)
    print(f'Base path: {main_base_path}')
    refined_paths = get_secondary_model_dirs([main_base_path,], filtering='refined_only')
    print(f'Refined paths: {refined_paths}')

    with open(os.path.join(main_base_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        base_model_args = dict_to_namedtuple(base_model_args)
        dataset_name = base_model_args.dataset

    _, test_dataloader, unnormalizer = load_dataset(dataset_name, eval_args.test_batch_size,
                                       eval_args.test_batch_size, shuffle_test=eval_args.shuffle_test)

    encoder1 = Encoder1(base_model_args).to(device)
    encoder1.load_state_dict(torch.load(os.path.join(main_base_path, 'encoder1.ckpt'), map_location=device))
    encoder1.eval()

    if eval_args.custom_selection is None:
        x_test, _ = next(iter(test_dataloader))
    else:
        x_test = custom_selection(eval_args, test_dataloader)
    x_test = x_test.to(device)

    save_image(unnormalizer(x_test.data[:n*n]), os.path.join(savepath, '_originals.png'), nrow=n)
    for refined_path in refined_paths:
        print(f'Refined path: {refined_path}')
        with open(os.path.join(refined_path, '_settings.json'), 'r') as f:
            refined_model_args = json.load(f)
            refined_model_args = dict_to_namedtuple(refined_model_args)

        encoder2 = Encoder2(refined_model_args).to(device)
        encoder2.load_state_dict(torch.load(os.path.join(refined_path, 'encoder2.ckpt'), map_location=device))
        encoder2.eval()

        decoder2 = Decoder2(refined_model_args).to(device)
        decoder2.load_state_dict(torch.load(os.path.join(refined_path, 'decoder2.ckpt'), map_location=device))
        decoder2.eval()

        with torch.no_grad():
            # Important: Permute over z entries as done in selection phase
            # for each element of the batch
            u1_test = uniform_noise([x_test.size(0), latent_dim_1_lower], encoder1.alpha).to(device)
            u2_test = uniform_noise([x_test.size(0), refined_model_args.latent_dim_2], encoder2.alpha).to(device)
            u_test = torch.cat((u1_test, u2_test), dim=1)
            z1 = encoder1(x_test, u1_test)
            z2 = encoder2(x_test, u2_test)
            z = torch.cat((z1, z2), dim=1)
            x_test_recon = decoder2(z, u_test)

            # Save square grid of images
            save_image(unnormalizer(x_test_recon.data[:n*n]),
                       os.path.join(savepath, f'refined-{refined_model_args.latent_dim_2}-{refined_model_args.L_2}-' + \
                                              'Lambda={:.5f}.png' \
                                              .format(refined_model_args.Lambda_refined)), nrow=n)

    # --------- Evaluate end-to-end models of both rates ----------

    other_base_paths = []
    other_base_Lambda_list = str_values_to_list(eval_args.other_base_Lambda, delimiter=',', dtype=float)
    for latent_dim_1, L_1 in zip([latent_dim_1_lower, latent_dim_1_higher], [L_1_lower, L_1_higher]):
        other_base_paths += [
            os.path.join(eval_args.experiment_path_pre,
                        f'{latent_dim_1}-{L_1}',
                        'Lambda={:.5f}'.format(other_base_Lambda))
            for other_base_Lambda in other_base_Lambda_list
        ]

    for other_base_path in other_base_paths:
        print(f'Other base path: {other_base_path}')

        with open(os.path.join(other_base_path, '_settings.json'), 'r') as f:
            other_base_model_args = json.load(f)
            other_base_model_args = dict_to_namedtuple(other_base_model_args)

        encoder1 = Encoder1(other_base_model_args).to(device)
        encoder1.load_state_dict(torch.load(os.path.join(other_base_path, 'encoder1.ckpt'), map_location=device))
        encoder1.eval()

        decoder1 = Decoder1(other_base_model_args).to(device)
        decoder1.load_state_dict(torch.load(os.path.join(other_base_path, 'decoder1.ckpt'), map_location=device))
        decoder1.eval()

        with torch.no_grad():
            u1_test = uniform_noise([x_test.size(0), other_base_model_args.latent_dim_1], encoder1.alpha).to(device)
            z = encoder1(x_test, u1_test)
            x_test_recon = decoder1(z, u1_test)

            # Save square grid of images
            save_image(unnormalizer(x_test_recon.data[:n*n]),
                    os.path.join(savepath, f'e2e-{other_base_model_args.latent_dim_1}-{other_base_model_args.L_1}-' + \
                                            'Lambda={:.5f}.png' \
                                            .format(other_base_model_args.Lambda_base)), nrow=n)

        # with open(savepath)

def testfunc(eval_args):
    vars(eval_args)['test_batch_size'] = 5
    latent_dim_1 = eval_args.latent_dim_1
    L_1 = eval_args.L_1

    main_base_subpath = os.path.join(f'{latent_dim_1}-{L_1}',
                                     'Lambda={:.5f}'.format(eval_args.Lambda_base))

    main_base_path = os.path.join(eval_args.experiment_path_pre, main_base_subpath)
    print(f'Base path: {main_base_path}')

    with open(os.path.join(main_base_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        base_model_args = dict_to_namedtuple(base_model_args)
        dataset_name = base_model_args.dataset

    _, test_dataloader, unnormalizer = load_dataset(dataset_name, eval_args.test_batch_size,
                                       eval_args.test_batch_size, shuffle_test=eval_args.shuffle_test)

    encoder1 = Encoder1(base_model_args).to(device)
    encoder1.load_state_dict(torch.load(os.path.join(main_base_path, 'encoder1.ckpt'), map_location=device))
    encoder1.eval()

    decoder1 = Decoder1(base_model_args).to(device)
    decoder1.load_state_dict(torch.load(os.path.join(main_base_path, 'decoder1.ckpt'), map_location=device))
    decoder1.eval()

    x_test, _ = next(iter(test_dataloader))
    x_test = x_test.to(device)
    u1_test = uniform_noise([x_test.size(0), eval_args.latent_dim_1], encoder1.alpha).to(device)
    z = encoder1(x_test, u1_test)
    x_test_recon = decoder1(z, u1_test)
    print(torch.max(u1_test))
    print(torch.mean(x_test))
    print(torch.mean(x_test_recon))
    print(torch.max(x_test))
    print(torch.max(x_test_recon))
    print(torch.mean(torch.abs(x_test - x_test_recon)))
    print(torch.mean(torch.mul(x_test - x_test_recon, x_test - x_test_recon)))

    # python evaluate.py -experiment_path_pre experiments/S1 -mode testfunc --latent_dim_1 10 --L_1 8 --Lambda_base 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-experiment_path_pre", type=str, help="folder (including experiments/)")
    parser.add_argument("-mode", type=str, default='2', help="folder under experiments/")
    # for eval mode two
    parser.add_argument("--rdp_name_suffix", type=str, default='', help="suffix for plot title")
    parser.add_argument("--n_eval", type=int, default=3, help="[CALCULATE LOSSES MODE] number of batch estimates over entire test set to use for evaluating perception loss")
    parser.add_argument("--test_batch_size", type=int, default=5000, help="test set batch size (default: 1000)")
    parser.add_argument("--n_losses", type=int, default=3, help="[SAVED LOSSES MODE] number of losses to average over")
    parser.add_argument("--colorbar", type=str, default='bits', help="Colorbar to display - bits or dims")
    parser.add_argument("--text", type=str, default='Lambdas', help="Text to display - bits or dims")
    parser.add_argument("--line_segments", type=str, default='none', help="Line segments on plot between base/multistage models")
    parser.add_argument("--climits", nargs=2, type=str, default=[None, None], help="Line segments on plot between base/multistage models")
    # for preview_reduced_decoders
    # parser.add_argument("--base_subpath", type=str, default='', help="Path to load the base model under experiments/...subpaths.../")
    parser.add_argument("--Lambda_base", type=float, default=0.0, help="Lambda value for base model")
    parser.add_argument("--latent_dim_1", type=int, default=4, help="Latent dimension size of base model")
    parser.add_argument("--L_1", type=int, default=4, help="Number of quantization levels for base model")
    parser.add_argument("--shuffle_test", type=int, default=1, help="Whether to use random test samples or not")
    parser.add_argument("--random_seed", type=int, default=1, help="Random seed")
    parser.add_argument("--custom_selection", type=str, default=None, help="custom selection")
    # for compare_1stage_vs_2stage
    parser.add_argument("--enc_2_layer_scale", type=float, default=1.0, help="Scale layer size of encoder by factor")
    parser.add_argument("--other_base_Lambda", type=str, default='', help="Other base Lambdas for compare_1stage_vs_2stage")
    parser.add_argument("--num_pics", type=int, default=5, help="n, for n x n picture grids")
    parser.add_argument("--latent_dim_1s", nargs=2, type=int, default=[3,6], help="Latent dimension size of base model at higher and lower rate")
    parser.add_argument("--L_1s", nargs=2, type=int, default=[3,3], help="Number of quantization levels for base model at higher and lower rate")
    # for comparing refinement to base models
    # parser.add_argument("--other_base_Lambda_high_rate", type=str, default='', help="Other base Lambdas for base model of refinement models")
    # parser.add_argument("--other_base_Lambda_low_rate", type=str, default='', help="Other base Lambdas for base model of higher rate")

    eval_args = parser.parse_args()

    if eval_args.random_seed:
        random.seed(eval_args.random_seed)
        np.random.seed(eval_args.random_seed)
        torch.manual_seed(eval_args.random_seed)

    if eval_args.mode == '2':
        info = eval_mode_two(eval_args)
        ds, ps, bits, latent_dims, Ls, Lambdas, Lambda_bases, plot_settings \
            = info['distortions'], info['perceptions'], info['bits'], \
              info['latent_dims'], info['quantization_levels'], info['Lambdas'], \
              info['Lambda_bases'], info['plot_settings']

        lines, line_settings = info['lines'], info['line_settings']
        line_colors = ['silver' if line_setting[5] == 0 else 'grey'
                       for line_setting in line_settings]
        # for line in lines:
        #     print(line)

        if eval_args.text == 'Lambdas':
            texts = Lambdas
        elif eval_args.text == 'Ls':
            texts = Ls
        else:
            texts = ['' for _ in Lambdas]

        if eval_args.colorbar == 'bits':
            colorbar = bits
            coloarbar_label = 'rate'
        elif eval_args.colorbar == 'latent_dims':
            colorbar = latent_dims
            coloarbar_label = 'z_dims'
        else:
            colorbar = Lambda_bases
            coloarbar_label = '$\lambda_1$'

        climits = [float(eval_args.climits[0]) if isfloat(eval_args.climits[0]) else None,
                   float(eval_args.climits[1]) if isfloat(eval_args.climits[1]) else None]
        plot_rdp(ds, ps, colorbar, eval_args.experiment_path_pre, texts=texts, colorbar=coloarbar_label,
                 plot_settings=plot_settings, line_segments=eval_args.line_segments, lines=lines,
                 line_colors=line_colors, climits=climits, rdp_name_suffix=eval_args.rdp_name_suffix)
    elif eval_args.mode == 'preview_reduced_decoders':
        preview_reduced_decoders(eval_args)
    elif eval_args.mode == 'compare_universal':
        compare_e2e_vs_universal(eval_args)
    elif eval_args.mode == 'compare_refinement':
        compare_e2e_vs_refined(eval_args)
    elif eval_args.mode == 'testfunc':
        testfunc(eval_args)
