import os
import json
from copy import deepcopy
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
from quantizer_entropy import base_estimate_dim_entropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def latent_sampler(data):
    n = data.shape[0]
    de = torch.LongTensor(random_derangement(n)).to(device)

    return data.index_select(0, de)

def visualize_with_dims(dims, encoder, decoder, data_loader, ls_sample_dir):
    x, _ = next(iter(data_loader))
    x = x[:100].view(-1, 784).to(device)
    z = encoder(x)
    z_fill = latent_sampler(z).to(device)
    latent_selected = []
    latent_unselected = set(range(len(dims)))
    for i, dim in enumerate(dims):
        latent_selected.append(dim)
        latent_unselected -= {dim,}
        z_fill[:,dim] = z[:,dim]
        x_reconst_dim = decoder(z_fill)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), x_reconst_dim.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(ls_sample_dir, f'reconstruct-{i+1}.png'))

def selection_criterion(x, x_hat, args, D):
    lambda_select = args.Lambda_select
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        distortion_loss = mse_loss(x, x_hat)
        if lambda_select > 0:
            real_validity = D(x)
            fake_validity = D(x_hat)
            perception_loss = torch.mean(real_validity) - torch.mean(fake_validity)
        else:
            perception_loss = 0
        loss = distortion_loss + lambda_select*perception_loss

    return loss

def plot_losses(errors, latent_selected, savedir, args):
    x = np.array(range(len(latent_selected))) + 1
    errors_selected = np.choose(latent_selected, errors.T)

    plt.scatter(x, errors_selected)
    plt.title('WGAN-AE Latent Dimension Reduction')
    plt.xlabel('Number of latent dimensions in use')
    plt.ylabel(f'D + λP with λ={args.Lambda_select}')
    for xi, yi, latent_dim in zip(x, errors_selected, latent_selected):
        plt.annotate(f'{latent_dim}', (xi, yi))
    plt.savefig(f'{savedir}/selected_losses.png')

def latent_selection_reconst(args, device):
    # Note: this is stochastic due to the latent sampler
    # Initialize decoder and discriminator
    encoder1 = Encoder1(args).to(device)
    decoder1 = Decoder1(args).to(device)
    discriminator1 = Discriminator(args).to(device)
    encoder1.eval() # Disable batch norm, or else results inconsistent
    decoder1.eval()
    discriminator1.eval()

    with open(os.path.join(args.load_base_model_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        assert_args_match(base_model_args, vars(args), ('L_1', 'latent_dim_1', 'limits', 'enc_layer_scale'))

    encoder1.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'encoder1.ckpt')))
    decoder1.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'decoder1.ckpt')))
    discriminator1.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'discriminator1.ckpt')))

    with open(f'{args.load_base_model_path}/_settings.json', 'r') as f:
        base_settings = json.load(f)

    assert base_settings['L_1'] == args.L_1
    assert base_settings['latent_dim_1'] == args.latent_dim_1

    # Configure data loader
    # os.makedirs("data/mnist", exist_ok=True)
    # dataloader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "data/mnist",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.Resize(28), transforms.ToTensor()]
    #         ),
    #     ),
    #     batch_size=args.batch_size,
    #     shuffle=False,
    # )
    # assert dataloader.dataset.targets.size(0) % args.batch_size == 0
    # n_iters = dataloader.dataset.targets.size(0) // args.batch_size

    train_dataloader, _ = load_dataset(args.dataset, args.batch_size,
                                       args.batch_size, shuffle_train=False)
    n_iters = train_dataloader.dataset.targets.size(0) // args.batch_size

    # Find best order across all dimensions, for flexibility in choosing the size of reduced model later on
    latent_dim_0 = args.latent_dim_0 if args.latent_dim_0 > 0 else args.latent_dim_1

    criterion = lambda x, x_reconst: selection_criterion(x, x_reconst, args, discriminator1)
    latent_selected = []
    latent_unselected = set(range(args.latent_dim_1))
    errors = np.zeros((latent_dim_0, args.latent_dim_1))
    with torch.no_grad():
        for t in range(latent_dim_0):
            for x, _ in train_dataloader:
                x = x.to(device)
                z = encoder1(x)
                for dim in latent_unselected:
                    z_copy = torch.zeros_like(z).to(device)
                    latent_selected_with_dim = deepcopy(latent_selected).append(dim)
                    latent_unselected_minus_dim = list(deepcopy(latent_unselected) - {dim,})
                    # For z (n_datapoints x latent_dim), keep the first few latent dims fixed
                    z_copy[:,latent_selected_with_dim] = z[:,latent_selected_with_dim]
                    # Now permute the last few dimensions to simulate sampling from the latent distribution
                    z_copy[:,latent_unselected_minus_dim] = \
                        latent_sampler(z[:,latent_unselected_minus_dim])

                    x_reconst_dim = decoder1(z_copy)
                    loss = criterion(x, x_reconst_dim)
                    errors[t,dim] += loss / n_iters

            dim_ranking = np_argsort_excluding(errors[t], latent_selected)
            min_error_dim = dim_ranking[0]
            latent_selected.append(int(min_error_dim)) # Can't JSON Serialize numpy objects
            latent_unselected -= {min_error_dim,}

        print(latent_selected)
        # print(errors)
        print('Final loss:', errors[t,min_error_dim])

    ls_sample_dir = f'{args.load_base_model_path}/_reduced/reconst'
    os.makedirs(ls_sample_dir, exist_ok=True)
    visualize_with_dims(latent_selected, encoder1, decoder1, train_dataloader,
                        ls_sample_dir)
    plot_losses(errors, latent_selected, ls_sample_dir, args)

    # if os.path.isfile(os.path.join(args.load_base_model_path, '_latent_selection.json')):
    #     with open(os.path.join(args.load_base_model_path, '_latent_selection.json'), 'r') as f:
    #         data = json.load(f)
    #         data['reconst'] = {'latent_selected': latent_selected, 'Lambda_select': args.Lambda_select}
    # else:
    #     data = {'reconst': {'latent_selected': latent_selected, 'Lambda_select': args.Lambda_select}}
    #
    # with open(os.path.join(args.load_base_model_path, '_latent_selection.json'), 'w') as f:
    #     json.dump(data, f)

    info_dict = {'latent_selected': latent_selected, 'Lambda_select': args.Lambda_select}
    save_info('reconst', info_dict, args)
    return latent_selected

def latent_select_entropy(args, device):
    # Initialize decoder and discriminator
    # encoder1 = Encoder1(args).to(device)
    # decoder1 = Decoder1(args).to(device)
    # discriminator1 = Discriminator(args).to(device)

    # with open(os.path.join(args.load_base_model_path, '_settings.json'), 'r') as f:
    #     base_model_args = json.load(f)
    #     assert_args_match(base_model_args, vars(args), ('L_1', 'latent_dim_1', 'limits', 'enc_layer_scale'))

    # encoder1.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'encoder1.ckpt')))
    # decoder1.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'decoder1.ckpt')))
    # discriminator1.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'discriminator1.ckpt')))

    # with open(f'{args.load_base_model_path}/_settings.json', 'r') as f:
    #     base_settings = json.load(f)

    # assert base_settings['L_1'] == args.L_1
    # assert base_settings['latent_dim_1'] == args.latent_dim_1

    # # Configure data loader
    # os.makedirs("data/mnist", exist_ok=True)
    # dataloader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "data/mnist",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.Resize(28), transforms.ToTensor()]
    #         ),
    #     ),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    # )
    # assert 10000 % args.batch_size == 0
    # n_test_iters = 10000 // args.batch_size

    # # Find best order across all dimensions, for flexibility in choosing the size of reduced model later on
    # latent_dim_0 = args.latent_dim_0 if args.latent_dim_0 > 0 else args.latent_dim_1

    latent_dim_0 = args.latent_dim_0 if args.latent_dim_0 > 0 else args.latent_dim_1

    vars(args)['experiment_path'] = args.load_base_model_path
    h_by_dim = base_estimate_dim_entropy(args, device)
    latent_selected = np.argsort(h_by_dim).tolist()[:latent_dim_0] # index sort
    save_info('entropy', latent_selected, args)
    # with torch.no_grad():
    #     for t in range(latent_dim_0):
    #         for x, _ in dataloader:
    #             x = x.to(device)
    #             z = encoder1(x)

    # ls_sample_dir = f'{args.load_base_model_path}/_reduced'
    # os.makedirs(ls_sample_dir, exist_ok=True)
    # visualize_with_dims(latent_selected, encoder1, decoder1, dataloader,
    #                     ls_sample_dir)
    # plot_losses(errors, latent_selected, ls_sample_dir, args)


    return latent_selected

def latent_select_random(args, device):
    latent_dim_0 = args.latent_dim_0 if args.latent_dim_0 > 0 else args.latent_dim_1

    # latent_selected = list(reversed(range(latent_dim_0))) # reversed 0, 1, 2, ...
    latent_selected = random_derangement(latent_dim_0)
    save_info('random', latent_selected, args)

    return latent_selected

def latent_select_identity(args, device):
    # assert args.latent_dim_0 == args.latent_dim_1
    latent_dim_0 = args.latent_dim_1

    latent_selected = list(range(latent_dim_0))
    info_dict = {'latent_selected': latent_selected}
    save_info('identity', info_dict, args)

    return latent_selected

def save_info(selection_method, info_dict, args):
    if os.path.isfile(os.path.join(args.load_base_model_path, '_latent_selection.json')):
        with open(os.path.join(args.load_base_model_path, '_latent_selection.json'), 'r') as f:
            data = json.load(f)
            data[selection_method] = info_dict
    else:
        data = {selection_method: info_dict}

    with open(os.path.join(args.load_base_model_path, '_latent_selection.json'), 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Selection')
    parser.add_argument('--n_channel', type=int, default=1, help='hidden dimension of z (default: 8)')
    parser.add_argument('--latent_dim_1', type=int, default=8, help='hidden dimension of z (default: 8)')
    parser.add_argument('--latent_dim_0', type=int, default=-1, help='selection size of latent dimension')
    parser.add_argument('--quantize', type=int, default=1, help='DOES NOTHING RIGHT NOW')
    parser.add_argument('--stochastic', type=int, default=1, help='add noise below quantization threshold (default: True)')
    parser.add_argument('--L_1', type=int, default=4, help='number of quantization levels for base model (default: -1)')
    parser.add_argument('--limits', nargs=2, type=float, default=[-1,1], help='quanitzation limits (default: (-1,1))')
    parser.add_argument("--Lambda_select", type=float, default=0.0, help="coefficient for selection loss")
    parser.add_argument("--Lambda_base", type=float, default=0.0, help="coefficient for selection loss")
    parser.add_argument("--batch_size", type=int, default=5000, help="batch size (default: 5000)")
    parser.add_argument("--enc_layer_scale", type=int, default=1, help="Scale layer size of encoder by factor")
    parser.add_argument("--train_or_test", type=str, default='train', help="use train set or test set (DOES NOTHING; always uses train)")
    parser.add_argument("--dataset", type=str, default='mnist', help="dataset to use (default: mnist)")
    parser.add_argument("-load_base_model_path", type=str, help="Load pretrained base model for selection")
    parser.add_argument("-method", type=str, help="Method (reconstruction loss or dimensionwise entropy)")
    args = parser.parse_args()

    if args.method == 'reconstruction_loss' or args.method == 'reconst':
        latent_selected = latent_selection_reconst(args, device)
    elif args.method == 'entropy':
        latent_selected = latent_select_entropy(args, device)
    elif args.method == 'random': # random
        latent_selected = latent_select_random(args, device)
    elif args.method == 'identity': # random
        latent_selected = latent_select_identity(args, device)
    else:
        raise ValueError(f'Unknown selection method: {args.method}')
    print(f'{args.method}:', latent_selected)
