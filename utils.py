from typing import Tuple

import os
import re
import random
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import StepLR


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def load_dataset(dataset, batch_size, test_batch_size, shuffle_train=True, shuffle_test=False):
    if dataset.lower() == 'mnist':
        train_dataloader, test_dataloader, unnormalizer = \
            load_mnist(batch_size, test_batch_size, shuffle_train=shuffle_train, shuffle_test=shuffle_test)

    elif dataset.lower() == 'fashion_mnist':
        train_dataloader, test_dataloader, unnormalizer = \
            load_fashion_mnist(batch_size, test_batch_size, shuffle_train=shuffle_train, shuffle_test=shuffle_test)
    elif dataset.lower() == 'svhn':
        train_dataloader, test_dataloader, unnormalizer = \
            load_svhn(batch_size, test_batch_size, shuffle_train=shuffle_train, shuffle_test=shuffle_test)
    elif dataset.lower() == 'lsun_bedrooms':
        train_dataloader, test_dataloader, unnormalizer = \
            load_lsun(batch_size, test_batch_size, shuffle_train=shuffle_train, shuffle_test=shuffle_test)
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return train_dataloader, test_dataloader, unnormalizer

def load_mnist(batch_size, test_batch_size, shuffle_train=True, shuffle_test=False):
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor()]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=shuffle_test,
    )
    assert test_dataloader.dataset.targets.size(0) % test_batch_size == 0, \
        f'{test_dataloader.dataset.targets.size(0)} test_batch_size:{test_batch_size}'

    return train_dataloader, test_dataloader, UnNormalize(0, -1, identity=True)

def load_fashion_mnist(batch_size, test_batch_size, shuffle_train=True, shuffle_test=False):
    train_dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data/fashion_mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data/fashion_mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor()]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=shuffle_test,
    )
    assert test_dataloader.dataset.targets.size(0) % test_batch_size == 0

    return train_dataloader, test_dataloader, UnNormalize(0, -1, identity=True)

def load_svhn(batch_size, test_batch_size, shuffle_train=True, shuffle_test=False):
    train_dataloader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data/svhn",
            split='train',
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasets.SVHN(
            "data/svhn",
            split='test',
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(32), transforms.ToTensor()]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=shuffle_test,
    )
    # assert test_dataloader.dataset.data.size(0) % test_batch_size == 0

    return train_dataloader, test_dataloader, UnNormalize(0, -1, identity=True)

def load_lsun(batch_size, test_batch_size, shuffle_train=True, shuffle_test=False):
    mu, std = 0.5, 0.5
    train_dataloader = torch.utils.data.DataLoader(
        datasets.LSUN(
            root="data/lsun_bedrooms",
            classes=['bedroom_train'],
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((mu, mu, mu), (std, std, std)),
            ]),
        ),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasets.LSUN(
            root="data/lsun_bedrooms",
            classes=['bedroom_val'],            # What was done in dplc paper
            transform=transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((mu, mu, mu), (std, std, std)),
            ]),
        ),
        batch_size=test_batch_size,
        shuffle=shuffle_test,
    )

    return train_dataloader, test_dataloader, UnNormalize(mu, std)

class UnNormalize(object):
    def __init__(self, mean, std, identity=False):
        self.mean = mean
        self.std = std
        self.identity = identity

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.identity:
            # Do nothing
            return tensor

        for t, m, s in zip(tensor, self.mean, self.std):
            # The normalize code -> t.sub_(m).div_(s)
            t.mul_(s).add_(m)
        return tensor

# def compute_gradient_penalty(D, real_samples, fake_samples, device=torch.device('cuda')):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake samples
#     alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)

#     # Get random interpolation between real and fake samples
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#     d_interpolates = D(interpolates)
#     fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)

#     # Get gradient w.r.t. interpolates
#     gradients = torch.autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

#     return gradient_penalty

def uniform_noise(size, alpha: float):
    # Want this to be in [-alpha/2, +alpha/2)
    return torch.rand(*size)*alpha - alpha/2

def uniform_noise_like(x, alpha: float, epsilon: float=0, device=torch.device('cuda')):
    return torch.FloatTensor(*(x.shape)).uniform_(-alpha/2+epsilon, +alpha/2-epsilon).to(device)

def generate_centers(L: int, limits: Tuple[float, float]):
    # Uniformly distributed between [limits[0], limits[1]]
    lower, upper = limits[0], limits[1]
    assert lower < upper
    interval = upper - lower
    centers = [lower + l/(L-1)*interval for l in range(0, L)]

    return centers

def evaluate_losses(real_imgs, recon_imgs, discriminator):
    real_validity = discriminator(real_imgs)
    fake_validity = discriminator(recon_imgs)

    perception_loss = torch.mean(real_validity) - torch.mean(fake_validity)
    distortion_loss = F.mse_loss(real_imgs, recon_imgs)

    return distortion_loss, perception_loss

def group_floats_1f(floats):
    bins = np.linspace(0, 30, 300)
    digitized = sorted(np.digitize(floats, bins) / 10)
    values = [0,]
    current_value = 0
    for i in range(1, len(digitized)):
        if abs(digitized[i-1] - digitized[i]) > 0.001:
            current_value += 1
        values.append(current_value)

    return digitized, values

def np_argsort_excluding(np_array, exclusions):
    ordering = list(np.argsort(np_array))
    # Include only dimensions not in the exclusion
    ordering = list(filter(lambda x: x not in exclusions, ordering))

    return ordering

def isfloat(value, non_negativity=False):
    # https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
    try:
        v = float(value)
        return v >= 0 if non_negativity else True
    except (ValueError, TypeError):
        return False

def isLambda(dir_name: str):
    x = dir_name.split('=')
    return len(x) == 2 and x[0] == 'Lambda' and isfloat(x[1], non_negativity=True)

def get_base_model_dirs(parent_dir="experiments", return_model_dirs=False):
    # TODO: Generate tree like structure first then combine into single list of full paths
    root, dirs, files = next(os.walk(parent_dir))

    search_re = r'^\d+-\d+$|^\d+-\d+-\d+-\d+$'

    # Base models
    model_dirs = []
    _dirs = []
    for dir_ in dirs:
        is_model = re.search(search_re, dir_) # Of the form 4-4, or 2-16, etc...
        if is_model:
            dir_ = os.path.join(parent_dir, dir_)
            root, dirs_, files = next(os.walk(dir_))
            dirs_ = [os.path.join(dir_, dir__) for dir__ in dirs_]
            _dirs.extend(dirs_)

    experiment_dirs = []
    for dir_ in _dirs:
        dir__ = os.path.split(dir_)[-1]

        # Check validity
        if not isLambda(dir__):
            continue
        # x = dir__.split('=')
        # if len(x) != 2:
        #     continue
        # if x[0] != 'Lambda' or not isfloat(x[1]):
        #     continue

        try:
            if dir__ == '_MSE':
                Lambda = 0
            else:
                _, Lambda = dir__.split('=')
                Lambda = float(Lambda)
        except Exception as ex:
            print(ex)
            print(f'Not a model directory: {dir_}')
            continue

        # print(f'model directory: {dir_}')
        model_dirs.append(dir_)
        root, dirs_, files = next(os.walk(dir_))
        dirs_ = [os.path.join(dir_, dir__) for dir__ in dirs_]
        experiment_dirs.extend(dirs_)

    if return_model_dirs:
        return model_dirs, experiment_dirs

    return experiment_dirs

def get_secondary_model_dirs(Lambda_dirs, filtering=None):
    # File storing names of all the different reduction methods
    with open('reduction_methods.txt', 'r') as f:
        reduction_methods = ['[' + m.strip() + ']' for m in f.readlines()]

    # Of the form 4-4, or 2-16, etc...
    # or 4-4-[entropy], 6-4-[reconst], etc
    search_re = r'^\d+-\d+$|^\d+-\d+-\[\w+\]$'

    # Refined/Reduced models
    secondary_dirs = []
    for secondary_root in Lambda_dirs:
        primary_dir_full = os.path.normpath(secondary_root).split(os.sep)
        primary_latent_dir = primary_dir_full[-2]
        primary_latent_dir_details = primary_latent_dir.split('-')
        latent_dim_1, L_1 = int(primary_latent_dir_details[0]), int(primary_latent_dir_details[1])
        try:
            _, latent_dirs, _ = next(os.walk(secondary_root))
        except StopIteration as si:
            print('\nWarning: No secondary directories exist')
            print(si)
            print('Returning empty list.\n')
            return []

        for latent_dir in latent_dirs:
            latent_dir_full = os.path.join(secondary_root, latent_dir)

            is_model = re.search(search_re, latent_dir)
            if is_model:
                # Filter out certain subdirectories corresponding to model types
                # Note: 'reduced_only' corresponds to two stage reduction, this
                # will filter out joint reduction models.
                if filtering == 'reduced_only':
                    latent_dir_details = latent_dir.split('-')
                    if len(latent_dir_details) == 2:
                        print(f'Incompatible directory: {latent_dir}')
                    elif len(latent_dir_details) == 3:
                        if latent_dir_details[-1] not in reduction_methods:
                            continue
                elif filtering == 'reduced_only_same_dim': # reduced_only with same-dimension latents
                    latent_dir_details = latent_dir.split('-')
                    if len(latent_dir_details) == 2:
                        print(f'Incompatible directory: {latent_dir}')
                    elif len(latent_dir_details) == 3:
                        if latent_dir_details[-1] not in reduction_methods:
                            continue

                        latent_dim_0, L_0 = int(latent_dir_details[0]), int(latent_dir_details[1])
                        if latent_dim_1 != latent_dim_0 or L_1 != L_0: # if not same dimension/rate decoder
                            continue
                elif filtering == 'refined_only':
                    latent_dir_details = latent_dir.split('-')
                    if len(latent_dir_details) == 2:
                        print(f'Incompatible directory: {latent_dir}')
                    elif len(latent_dir_details) == 3:
                        if latent_dir_details[-1] != '[refined]':
                            continue

                _, sub_Lambda_dirs, _ = next(os.walk(latent_dir_full))
                sub_Lambda_dirs_filtered = filter(isLambda, sub_Lambda_dirs)
                # for sub_Lambda_dir in sub_Lambda_dirs:
                #     x = sub_Lambda_dir.split('=')
                #     if len(x) != 2:
                #         continue
                #     if x[0] != 'Lambda' or not isfloat(x[1]):
                #         continue
                #     sub_Lambda_dirs_filtered.append(sub_Lambda_dir)

                sub_Lambda_dirs_full = [os.path.join(latent_dir_full, sub_Lambda_dir)
                                        for sub_Lambda_dir in sub_Lambda_dirs_filtered]
                secondary_dirs.extend(sub_Lambda_dirs_full)

    return secondary_dirs

def get_model_dirs(parent_dir, filtering=None):
    model_dirs, experiment_dirs = get_base_model_dirs(parent_dir=parent_dir, return_model_dirs=True)

    secondary_dirs = get_secondary_model_dirs(model_dirs, filtering=filtering)
    model_dirs.extend(secondary_dirs)

    # TODO: check validity of model directory like in second block

    return model_dirs

def random_derangement(n):
    # Source: https://stackoverflow.com/questions/25200220/generate-a-random-derangement-of-a-list
    random.seed(0)
    while True:
        v = [i for i in range(n)]
        for j in range(n - 1, -1, -1):
            p = random.randint(0, j)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return tuple(v)

def assert_args_match(args1, args2, matching_args):
    # Check that every arg in matching_dict matches between args1 and args2
    for arg in matching_args:
        assert arg in args1 and arg in args2, f'{arg} in args1: {arg in args1}, {arg} in args2: {arg in args2}'
        assert args1[arg] == args2[arg], f'Args don\'t match: {args1[arg]} != {args2[arg]}, {arg}'

def dict_to_namedtuple(x):
    return namedtuple('TrainArgs', x.keys())(**x)

def str_values_to_tensor(s, delimiter=','):
    return torch.LongTensor([int(dim) for dim in s.split(delimiter)])

def str_values_to_list(s, delimiter=',', dtype=int):
    return [dtype(value) for value in s.split(delimiter)]

def reformat_lines(lines):
    # the lines are formatted ((xa_1,ya_1),(xb_1,yb_1),...) right now
    # but they need to be reformatted as (xa_1,xb_1,xa_2,xb_2,...), (ya_1,yb_1,ya_2,yb_2,...)
    reformatted_x, reformatted_y = [], []
    for p1, p2 in lines:
        reformatted_x.extend([p1[0], p2[0]])
        reformatted_y.extend([p1[1], p2[1]])

    return reformatted_x, reformatted_y

def calculate_rate(mode, latent_dim_1, L_1, latent_dim_2=-1, L_2=-1, latent_dim_0=-1, L_0=-1):
    n_bits = None
    if mode == 'base':
        n_bits = latent_dim_1 * math.log2(L_1)
    elif mode == 'refined':
        if L_1 > 0 and L_2 > 0:
            n_bits = latent_dim_1 * math.log2(L_1) \
                + latent_dim_2 * math.log2(L_2)
    elif mode == 'reduced':
        if L_1 > 0 and L_0 > 0:
            n_bits = latent_dim_0 * math.log2(L_0)

    return n_bits

if __name__ == '__main__':
    
    get_model_dirs(parent_dir="experiments/1")
