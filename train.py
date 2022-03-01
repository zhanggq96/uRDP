import argparse
import os
import sys
import math
import json
from shutil import copyfile
from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from models import *
from utils import *
from quantizer_entropy import base_estimate_entropy_, \
    refined_estimate_entropy_, reduced_estimate_entropy_


# TODO: Rename Lambda_base/refined/reduced to Lambda_1/2/0 ?

def is_progress_interval(args, epoch):
    return epoch == args.n_epochs-1 or (args.progress_intervals > 0 and epoch % args.progress_intervals == 0)

def _lr_factor(epoch, dataset, mode=None):
    if dataset == 'mnist':
        if epoch < 20:
            return 1
        elif epoch < 40:
            return 1/5
        else:
            return 1/50
    elif dataset == 'fasion_mnist':
        if epoch < 20:
            return 1
        elif epoch < 35:
            return 1/5
        else:
            return 1/50
    elif dataset == 'svhn':
        if epoch < 25:
            return 1
        else:
            return 1/5
    else:
        return 1

def compute_lambda_anneal(Lambda, epoch, Lambda_init=0.0005, end_epoch=12):
    assert Lambda == 0 and epoch >= 0
    e = min(epoch, end_epoch)

    return Lambda_init*(end_epoch-e)/end_epoch

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Source: https://github.com/andreaferretti/wgan/blob/master/train.py
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_base(args, device):
    experiment_path = args.experiment_path
    N = 60000

    assert (args.L_1 > 0 or not args.quantize) and args.latent_dim_1 > 0
    assert not (args.L_1 > 0 and not args.quantize), f'Quantization disabled, yet args.L_1={args.L_1}'

    # Loss weight for gradient penalty
    lambda_gp = args.Lambda_gp

    # Initialize decoder and discriminator
    encoder1 = Encoder1(args).to(device)
    decoder1 = Decoder1(args).to(device)
    discriminator1 = Discriminator(args).to(device)
    alpha1 = encoder1.alpha

    if args.initialize_mse_model:
        # Load pretrained models to continue from if directory is provided
        if args.Lambda_base > 0:
            assert isinstance(args.load_mse_model_path, str)

            # Check args match
            with open(os.path.join(args.load_mse_model_path, '_settings.json'), 'r') as f:
                mse_model_args = json.load(f)
                assert_args_match(mse_model_args, vars(args), ('L_1', 'latent_dim_1', 'limits', 'enc_layer_scale'))
                assert mse_model_args['Lambda_base'] == 0
                # No need to assert args match for "stochastic" and "quantize"?

        if isinstance(args.load_mse_model_path, str):
            assert args.Lambda_base > 0, args.load_mse_model_path
            encoder1.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'encoder1.ckpt')))
            decoder1.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'decoder1.ckpt')))
            discriminator1.load_state_dict(torch.load(os.path.join(args.load_mse_model_path, 'discriminator1.ckpt')))

    # Configure data loader
    train_dataloader, test_dataloader, unnormalizer = \
        load_dataset(args.dataset, args.batch_size, args.test_batch_size, shuffle_train=True)
    test_set_size = len(test_dataloader.dataset)

    # Optimizers
    optimizer_E1 = torch.optim.Adam(encoder1.parameters(), lr=args.lr_encoder, betas=(args.beta1_encoder, args.beta2_encoder))
    optimizer_G1 = torch.optim.Adam(decoder1.parameters(), lr=args.lr_decoder, betas=(args.beta1_decoder, args.beta2_decoder))
    optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=args.lr_critic, betas=(args.beta1_critic, args.beta2_critic))

    lr_factor = lambda epoch: _lr_factor(epoch, args.dataset)

    scheduler_E1 = LambdaLR(optimizer_E1, lr_factor)
    scheduler_G1 = LambdaLR(optimizer_G1, lr_factor)
    scheduler_D1 = LambdaLR(optimizer_D1, lr_factor)

    criterion = nn.MSELoss()

    # ----------
    #  Prep
    # ----------
    # copyfile('models.py', 'experiments/models.txt')

    os.makedirs(f"{experiment_path}", exist_ok=True)
    with open(f'{experiment_path}/_settings.json', 'w') as f:
        json.dump(vars(args), f)

    with open(f'{experiment_path}/_losses.csv', 'w') as f:
        f.write('epoch,distortion_loss,perception_loss\n')

    # ----------
    #  Training
    # ----------

    batches_done = 0
    n_cycles = 1 + args.n_critic
    disc_loss = torch.Tensor([-1])
    distortion_loss = torch.Tensor([-1])
    saved_original_test_image = False

    for epoch in range(args.n_epochs):
        Lambda = args.Lambda_base
        if Lambda == 0:
            # Give an early edge to training discriminator for Lambda = 0
            Lambda = compute_lambda_anneal(Lambda, epoch)

        for i, (x, _) in enumerate(train_dataloader):
            # Configure input
            x = x.to(device)

            if i % n_cycles != 1:

                # ---------------------
                #  Train Discriminator
                # ---------------------

                free_params(discriminator1)
                frozen_params(encoder1)
                frozen_params(decoder1)

                optimizer_D1.zero_grad()

                # Noise batch_size x latent_dim
                u1 = uniform_noise([x.size(0), args.latent_dim_1], alpha1).to(device)
                z1 = encoder1(x, u1)
                x_recon = decoder1(z1, u1)
                # Real images
                real_validity = discriminator1(x)
                # Fake images
                fake_validity = discriminator1(x_recon)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator1, x.data, x_recon.data)
                # Adversarial loss
                disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                disc_loss.backward()

                optimizer_D1.step()

            else: # if i % n_cycles == 1:

                # -----------------
                #  Train Generator
                # -----------------

                frozen_params(discriminator1)
                free_params(encoder1)
                free_params(decoder1)

                optimizer_E1.zero_grad()
                optimizer_G1.zero_grad()

                u1 = uniform_noise([x.size(0), args.latent_dim_1], alpha1).to(device)
                z1 = encoder1(x, u1)
                x_recon = decoder1(z1, u1)

                # real_validity = discriminator(x)
                fake_validity = discriminator1(x_recon)

                perception_loss = -torch.mean(fake_validity) # + torch.mean(real_validity)
                distortion_loss = criterion(x, x_recon)

                loss = args.Lambda_distortion*distortion_loss + Lambda*perception_loss
                loss.backward()

                optimizer_G1.step()
                optimizer_E1.step()

            if batches_done % 100 == 0:
                with torch.no_grad(): # use most recent results
                    real_validity = discriminator1(x)
                    perception_loss = -torch.mean(fake_validity) + torch.mean(real_validity)
                print(
                    "[Epoch %d/%d] [Batch %d/%d (batches_done: %d)] [Disc loss: %f] [Perception loss: %f] [Distortion loss: %f]"
                    % (epoch, args.n_epochs, i, len(train_dataloader), batches_done, disc_loss.item(),
                    perception_loss.item(), distortion_loss.item())
                )

            batches_done += 1

        # ---------------------
        # Evaluate losses on test set
        # ---------------------
        with torch.no_grad():
            encoder1.eval()
            decoder1.eval()
            discriminator1.eval()

            is_entropy_interval = args.entropy_intervals > 0 and (epoch < 5 or epoch % args.entropy_intervals == 0)
            if is_entropy_interval or ((epoch == args.n_epochs - 1 or epoch == 0) and args.entropy_intervals != -2):
                # use test batch size on training set for efficiency
                base_estimate_entropy_(encoder1, args.latent_dim_1, args.L_1, args.Lambda_base, 'train',
                                       args.test_batch_size, experiment_path, args.dataset, device)

            distortion_loss_avg, perception_loss_avg = 0, 0

            for j, (x_test, _) in enumerate(test_dataloader):
                x_test = x_test.to(device)
                u1_test = uniform_noise([x_test.size(0), args.latent_dim_1], alpha1).to(device)
                x_test_recon = decoder1(encoder1(x_test, u1_test), u1_test)
                distortion_loss, perception_loss = evaluate_losses(x_test, x_test_recon, discriminator1)
                distortion_loss_avg += x_test.size(0) * distortion_loss
                perception_loss_avg += x_test.size(0) * perception_loss

                if j == 0 and is_progress_interval(args, epoch):
                    save_image(unnormalizer(x_test_recon.data[:120]), f"{experiment_path}/{epoch}_recon.png", nrow=10, normalize=True)
                    if not saved_original_test_image:
                        save_image(unnormalizer(x_test.data[:120]), f"{experiment_path}/{epoch}_real.png", nrow=10, normalize=True)
                        saved_original_test_image = True

            distortion_loss_avg /= test_set_size
            perception_loss_avg /= test_set_size

            with open(f'{experiment_path}/_losses.csv', 'a') as f:
                f.write(f'{epoch},{distortion_loss_avg},{perception_loss_avg}\n')

            encoder1.train()
            decoder1.train()
            discriminator1.train()

        scheduler_E1.step()
        scheduler_D1.step()
        scheduler_G1.step()

    # ---------------------
    #  Save
    # ---------------------

    encoder1_file = f'{experiment_path}/encoder1.ckpt'
    decoder1_file = f'{experiment_path}/decoder1.ckpt'
    discriminator1_file = f'{experiment_path}/discriminator1.ckpt'

    torch.save(encoder1.state_dict(), encoder1_file)
    torch.save(decoder1.state_dict(), decoder1_file)
    torch.save(discriminator1.state_dict(), discriminator1_file)


def train_refined(args, device):
    experiment_path = args.experiment_path
    N = 60000

    assert args.Lambda_refined >= 0 and args.Lambda_reduced == -1
    assert args.L_1 > 0 and args.latent_dim_1 > 0 \
        and args.L_2 > 0 and args.latent_dim_2 > 0

    # Loss weight for gradient penalty
    lambda_gp = args.Lambda_gp

    # Initialize encoder, decoder and discriminator
    # no need to load decoder1
    encoder1 = Encoder1(args).to(device)
    encoder2 = Encoder2(args).to(device)
    decoder2 = Decoder2(args).to(device)
    discriminator2 = Discriminator(args).to(device)
    alpha1 = encoder1.alpha
    alpha2 = encoder2.alpha

    # Check args match
    with open(os.path.join(args.load_base_model_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        assert_args_match(base_model_args, vars(args), ('L_1', 'latent_dim_1', 'limits', 'enc_layer_scale'))

    # Load base model to continue from
    assert os.path.isfile(os.path.join(args.load_base_model_path, 'encoder1.ckpt')), \
        f'No file {os.path.join(args.load_base_model_path, "encoder1.ckpt")} found!'
    encoder1.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'encoder1.ckpt')))
    if args.initialize_base_discriminator:
        # Load base discriminator into discriminator2
        assert os.path.isfile(os.path.join(args.load_base_model_path, 'discriminator1.ckpt')), \
            f'No file {os.path.join(args.load_base_model_path, "discriminator1.ckpt")} found!'
        discriminator2.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'discriminator1.ckpt')))
        print('Successfully loaded base discriminator (parameters not frozen - will continue training).')

    frozen_params(encoder1) # Only train refining models

    # Configure data loader
    train_dataloader, test_dataloader, unnormalizer = \
        load_dataset(args.dataset, args.batch_size, args.test_batch_size, shuffle_train=True)
    test_set_size = len(test_dataloader.dataset)

    # Optimizers
    optimizer_E2 = torch.optim.Adam(encoder2.parameters(), lr=args.lr_encoder, betas=(args.beta1_encoder, args.beta2_encoder))
    optimizer_G2 = torch.optim.Adam(decoder2.parameters(), lr=args.lr_decoder, betas=(args.beta1_decoder, args.beta2_decoder))
    optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=args.lr_critic, betas=(args.beta1_critic, args.beta2_critic))

    lr_factor = lambda epoch: _lr_factor(epoch, args.dataset)

    scheduler_E2 = LambdaLR(optimizer_E2, lr_factor)
    scheduler_G2 = LambdaLR(optimizer_G2, lr_factor)
    scheduler_D2 = LambdaLR(optimizer_D2, lr_factor)

    criterion = nn.MSELoss()

    # ----------
    #  Prep
    # ----------

    os.makedirs(f"{experiment_path}", exist_ok=True)
    with open(f'{experiment_path}/_settings.json', 'w') as f:
        json.dump(vars(args), f)

    with open(f'{experiment_path}/_losses.csv', 'w') as f:
        f.write('epoch,distortion_loss,perception_loss\n')

    # ----------
    #  Training
    # ----------

    batches_done = 0
    n_cycles = 1 + args.n_critic
    disc_loss = torch.Tensor([-1])
    distortion_loss = torch.Tensor([-1])
    saved_original_test_image = False

    for epoch in range(args.n_epochs):
        Lambda = args.Lambda_refined

        for i, (x, _) in enumerate(train_dataloader):
            # Configure input
            x = x.to(device)

            if i % n_cycles != 1:

                # ---------------------
                #  Train Discriminator
                # ---------------------

                free_params(discriminator2)
                frozen_params(decoder2)
                frozen_params(encoder2)

                optimizer_D2.zero_grad()

                u1 = uniform_noise([x.size(0), args.latent_dim_1], alpha1).to(device)
                u2 = uniform_noise([x.size(0), args.latent_dim_2], alpha2).to(device)
                z1 = encoder1(x, u1)
                z2 = encoder2(x, u2)
                u = torch.cat((u1, u2), dim=1)
                z = torch.cat((z1, z2), dim=1)
                x_recon = decoder2(z, u)
                # Real images
                real_validity = discriminator2(x)
                # Fake images
                fake_validity = discriminator2(x_recon)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator2, x.data, x_recon.data)
                # Adversarial loss
                disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                disc_loss.backward()

                optimizer_D2.step()

            else: # if i % n_cycles == 1:

                # -----------------
                #  Train Generator
                # -----------------

                frozen_params(discriminator2)
                free_params(decoder2)
                free_params(encoder2)

                optimizer_G2.zero_grad()
                optimizer_E2.zero_grad()

                u1 = uniform_noise([x.size(0), args.latent_dim_1], alpha1).to(device)
                u2 = uniform_noise([x.size(0), args.latent_dim_2], alpha2).to(device)
                z1 = encoder1(x, u1)
                z2 = encoder2(x, u2)
                u = torch.cat((u1, u2), dim=1)
                z = torch.cat((z1, z2), dim=1)
                x_recon = decoder2(z, u)

                # real_validity = discriminator(x)
                fake_validity = discriminator2(x_recon)

                perception_loss = -torch.mean(fake_validity) # + torch.mean(real_validity)
                distortion_loss = criterion(x, x_recon)

                loss = args.Lambda_distortion*distortion_loss + Lambda*perception_loss
                loss.backward()

                optimizer_G2.step()
                optimizer_E2.step()

            if batches_done % 100 == 0:
                with torch.no_grad(): # use most recent results
                    real_validity = discriminator2(x)
                    perception_loss = -torch.mean(fake_validity) + torch.mean(real_validity)
                print(
                    "[Epoch %d/%d] [Batch %d/%d (batches_done: %d)] [Disc loss: %f] [Perception loss: %f] [Distortion loss: %f]"
                    % (epoch, args.n_epochs, i, len(train_dataloader), batches_done, disc_loss.item(),
                    perception_loss.item(), distortion_loss.item())
                )
            batches_done += 1

        # ---------------------
        # Evaluate losses on test set
        # ---------------------
        with torch.no_grad():
            encoder2.eval()
            decoder2.eval()
            discriminator2.eval()

            distortion_loss_avg, perception_loss_avg = 0, 0

            is_entropy_interval = args.entropy_intervals > 0 and (epoch < 5 or epoch % args.entropy_intervals == 0)
            if is_entropy_interval or ((epoch == args.n_epochs - 1 or epoch == 0) and args.entropy_intervals != -2):
                # use test batch size on training set for efficiency
                refined_estimate_entropy_(encoder1, args.latent_dim_1, args.L_1, args.Lambda_base,
                                          encoder2, args.latent_dim_2, args.L_2, args.Lambda_refined,
                                          'train', args.test_batch_size, experiment_path, args.dataset, device)

            for j, (x_test, _) in enumerate(test_dataloader):
                x_test = x_test.to(device)
                u1_test = uniform_noise([x_test.size(0), args.latent_dim_1], alpha1).to(device)
                u2_test = uniform_noise([x_test.size(0), args.latent_dim_2], alpha2).to(device)
                u_test = torch.cat([u1_test, u2_test], dim=1)
                z = torch.cat((encoder1(x_test, u1_test), encoder2(x_test, u2_test)), dim=1)
                x_test_recon = decoder2(z, u_test)
                distortion_loss, perception_loss = evaluate_losses(x_test, x_test_recon, discriminator2)
                distortion_loss_avg += x_test.size(0) * distortion_loss
                perception_loss_avg += x_test.size(0) * perception_loss

                if j == 0 and is_progress_interval(args, epoch):
                    save_image(unnormalizer(x_test_recon.data[:120]), f"{experiment_path}/{epoch}_recon.png", nrow=10, normalize=True)
                    if not saved_original_test_image:
                        save_image(unnormalizer(x_test.data[:120]), f"{experiment_path}/{epoch}_real.png", nrow=10, normalize=True)
                        saved_original_test_image = True

            distortion_loss_avg /= test_set_size
            perception_loss_avg /= test_set_size

            with open(f'{experiment_path}/_losses.csv', 'a') as f:
                f.write(f'{epoch},{distortion_loss_avg},{perception_loss_avg}\n')

            encoder2.train()
            decoder2.train()
            discriminator2.train()

        scheduler_E2.step()
        scheduler_D2.step()
        scheduler_G2.step()

    # ---------------------
    #  Save
    # ---------------------

    torch.save(decoder2.state_dict(), f'{experiment_path}/decoder2.ckpt')
    torch.save(encoder2.state_dict(), f'{experiment_path}/encoder2.ckpt')
    torch.save(discriminator2.state_dict(), f'{experiment_path}/discriminator2.ckpt')


def train_reduced(args, device):
    experiment_path = args.experiment_path

    assert args.L_1 > 0 and args.latent_dim_1 > 0 \
        and args.L_0 > 0 and args.latent_dim_0 > 0 \
        and args.L_0 == args.L_1
    assert args.Lambda_reduced >= 0 and args.Lambda_refined == -1
    assert args.latent_dim_0 <= args.latent_dim_1, f'Cannot reduce {args.latent_dim_1} to {args.latent_dim_0}.'

    # Loss weight for gradient penalty
    lambda_gp = args.Lambda_gp

    # Initialize decoder and discriminator
    encoder1 = Encoder1(args).to(device)
    decoder0 = Decoder0(args).to(device)
    discriminator0 = Discriminator(args).to(device)
    alpha1 = encoder1.alpha

    # Check args match
    with open(os.path.join(args.load_base_model_path, '_settings.json'), 'r') as f:
        base_model_args = json.load(f)
        assert args.L_0 == base_model_args['L_1']
        assert_args_match(base_model_args, vars(args), ('L_1', 'latent_dim_1', 'limits', 'enc_layer_scale'))

    # reduced_dims = torch.LongTensor([int(dim) for dim in args.reduced_dims.split(',')]).to(device)
    reduced_dims = str_values_to_tensor(args.reduced_dims, delimiter=',').to(device)
    print('Selected dimensions:', args.reduced_dims)

    # Load pretrained base models to continue from
    assert os.path.isfile(os.path.join(args.load_base_model_path, 'encoder1.ckpt')), \
        f'No file {os.path.join(args.load_base_model_path, "encoder1.ckpt")} found!'
    encoder1.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'encoder1.ckpt')))
    if args.initialize_base_discriminator:
        # Load base discriminator into discriminator0
        assert os.path.isfile(os.path.join(args.load_base_model_path, 'discriminator1.ckpt')), \
            f'No file {os.path.join(args.load_base_model_path, "discriminator1.ckpt")} found!'
        discriminator0.load_state_dict(torch.load(os.path.join(args.load_base_model_path, 'discriminator1.ckpt')))
        print('Successfully loaded base discriminator (parameters not frozen - will continue training).')

    frozen_params(encoder1) # Only train reduced models

    # Configure data loader
    train_dataloader, test_dataloader, unnormalizer = \
        load_dataset(args.dataset, args.batch_size, args.test_batch_size, shuffle_train=True)
    test_set_size = len(test_dataloader.dataset)

    # Optimizers
    optimizer_G0 = torch.optim.Adam(decoder0.parameters(), lr=args.lr_decoder, betas=(args.beta1_decoder, args.beta2_decoder))
    optimizer_D0 = torch.optim.Adam(discriminator0.parameters(), lr=args.lr_critic, betas=(args.beta1_critic, args.beta2_critic))

    lr_factor = lambda epoch: _lr_factor(epoch, args.dataset)

    scheduler_G0 = LambdaLR(optimizer_G0, lr_factor)
    scheduler_D0 = LambdaLR(optimizer_D0, lr_factor)

    criterion = nn.MSELoss()

    # ----------
    #  Prep
    # ----------

    os.makedirs(f"{experiment_path}", exist_ok=True)
    with open(f'{experiment_path}/_settings.json', 'w') as f:
        json.dump(vars(args), f)

    with open(f'{experiment_path}/_losses.csv', 'w') as f:
        f.write('epoch,distortion_loss,perception_loss\n')

    # ----------
    #  Training
    # ----------

    batches_done = 0
    n_cycles = 1 + args.n_critic
    disc_loss = torch.Tensor([-1])
    distortion_loss = torch.Tensor([-1])
    saved_original_test_image = False

    for epoch in range(args.n_epochs):
        Lambda = args.Lambda_reduced

        for i, (x, _) in enumerate(train_dataloader):
            # Configure input
            x = x.to(device)

            if i % n_cycles != 1:

                # ---------------------
                #  Train Discriminator
                # ---------------------

                free_params(discriminator0)
                frozen_params(decoder0)

                optimizer_D0.zero_grad()

                u1 = uniform_noise([x.size(0), args.latent_dim_1], alpha1).to(device)
                u0 = u1.index_select(1, reduced_dims) # Shared randomness over selected dims
                z1 = encoder1(x, u1)
                z0 = z1.index_select(1, reduced_dims) # Use only selected dimensions
                x_recon = decoder0(z0, u0)
                # Real images
                real_validity = discriminator0(x)
                # Fake images
                fake_validity = discriminator0(x_recon)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator0, x.data, x_recon.data)
                # Adversarial loss
                disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                disc_loss.backward()

                optimizer_D0.step()

            else:
                # -----------------
                #  Train Generator
                # -----------------

                frozen_params(discriminator0)
                free_params(decoder0)

                optimizer_G0.zero_grad()

                u1 = uniform_noise([x.size(0), args.latent_dim_1], alpha1).to(device)
                u0 = u1.index_select(1, reduced_dims) # Shared randomness over selected dims
                z1 = encoder1(x, u1)
                z0 = z1.index_select(1, reduced_dims) # Use only selected dimensions
                x_recon = decoder0(z0, u0)

                fake_validity = discriminator0(x_recon)

                perception_loss = -torch.mean(fake_validity) # + torch.mean(real_validity)
                distortion_loss = criterion(x, x_recon)

                loss = args.Lambda_distortion*distortion_loss + Lambda*perception_loss
                loss.backward()

                optimizer_G0.step()

            if batches_done % 100 == 0:
                with torch.no_grad(): # use most recent results
                    real_validity = discriminator0(x)
                    perception_loss = -torch.mean(fake_validity) + torch.mean(real_validity)
                print(
                    "[Epoch %d/%d] [Batch %d/%d (batches_done: %d)] [Disc loss: %f] [Perception loss: %f] [Distortion loss: %f]"
                    % (epoch, args.n_epochs, i, len(train_dataloader), batches_done, disc_loss.item(),
                    perception_loss.item(), distortion_loss.item())
                )

            batches_done += 1

        # ---------------------
        # Evaluate losses on test set
        # ---------------------
        with torch.no_grad():
            encoder1.eval()
            decoder0.eval()
            discriminator0.eval()

            is_entropy_interval = args.entropy_intervals > 0 and (epoch < 5 or epoch % args.entropy_intervals == 0)
            if is_entropy_interval or ((epoch == args.n_epochs - 1 or epoch == 0) and args.entropy_intervals != -2):
                # use test batch size on training set for efficiency
                reduced_estimate_entropy_(encoder1, args.latent_dim_1, args.L_1, args.Lambda_base,
                                          args.latent_dim_0, args.L_0, args.Lambda_reduced,
                                          'train', args.test_batch_size, experiment_path, args.dataset, device)

            distortion_loss_avg, perception_loss_avg = 0, 0
            for j, (x_test, _) in enumerate(test_dataloader):
                x_test = x_test.to(device)
                u1_test = uniform_noise([x_test.size(0), args.latent_dim_1], alpha1).to(device)
                u0_test = u1_test.index_select(1, reduced_dims) # Shared randomness over selected dims
                z0 = encoder1(x_test, u1_test).index_select(1, reduced_dims)
                x_test_recon = decoder0(z0, u0_test)
                distortion_loss, perception_loss = evaluate_losses(x_test, x_test_recon, discriminator0)
                distortion_loss_avg += x_test.size(0) * distortion_loss
                perception_loss_avg += x_test.size(0) * perception_loss

                if j == 0 and is_progress_interval(args, epoch):
                    save_image(unnormalizer(x_test_recon.data[:120]), f"{experiment_path}/{epoch}_recon.png", nrow=10, normalize=True)
                    if not saved_original_test_image:
                        save_image(unnormalizer(x_test.data[:120]), f"{experiment_path}/{epoch}_real.png", nrow=10, normalize=True)
                        saved_original_test_image = True

            distortion_loss_avg /= test_set_size
            perception_loss_avg /= test_set_size

            with open(f'{experiment_path}/_losses.csv', 'a') as f:
                f.write(f'{epoch},{distortion_loss_avg},{perception_loss_avg}\n')

            encoder1.train() # Params frozen still
            decoder0.train()
            discriminator0.train()
        # frozen_params(encoder1)

        scheduler_G0.step()
        scheduler_D0.step()

    torch.save(decoder0.state_dict(), f'{experiment_path}/decoder0.ckpt')
    torch.save(discriminator0.state_dict(), f'{experiment_path}/discriminator0.ckpt')



if __name__ == '__main__':
    os.makedirs("experiments", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=25, help="number of epochs of training")
    parser.add_argument("--n_channel", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument('--quantize', type=int, default=1, help='DOES NOTHING RIGHT NOW')
    parser.add_argument('--stochastic', type=int, default=1, help='add noise below quantization threshold (default: True)')
    parser.add_argument("--latent_dim_1", type=int, default=8, help="dimensionality of the latent space")
    parser.add_argument("--latent_dim_2", type=int, default=-1, help="dimensionality of the latent space for refinement model")
    parser.add_argument("--latent_dim_0", type=int, default=-1, help="dimensionality of the latent space for reduced model")
    parser.add_argument("--latent_dim_M1", type=int, default=-1, help="dimensionality of the latent space for joint_reduced_reduced model")
    parser.add_argument('--L_1', type=int, default=-1, help='number of quantization levels for base model (default: -1)')
    parser.add_argument('--L_2', type=int, default=-1, help='number of quantization levels for refined model (default: -1)')
    parser.add_argument('--L_0', type=int, default=-1, help='number of quantization levels for reduced model (default: -1)')
    parser.add_argument('--L_M1', type=int, default=-1, help='number of quantization levels for joint_reduced_reduced model (default: -1)')
    parser.add_argument('--limits', nargs=2, type=float, default=[-1,1], help='quanitzation limits (default: (-1,1))')
    parser.add_argument("--Lambda_base", type=float, default=0.0, help="coefficient for perception loss for training base model (default: 0.0)")
    parser.add_argument("--Lambda_refined", type=float, default=-1, help="coefficient for perception loss for training refined model (default: -1)")
    parser.add_argument("--Lambda_reduced", type=float, default=-1, help="coefficient for perception loss for training reduced model (default: -1)")
    parser.add_argument("--Lambda_reduced_reduced", type=float, default=-1, help="coefficient for perception loss for training joint_reduced_reduced model (default: -1)")
    parser.add_argument("--Lambda_distortion", type=float, default=1.0, help="coefficient for distortion loss (default: 1.0)")
    parser.add_argument("--Lambda_joint", type=float, default=1.0, help="coefficient for distortion loss (default: 1.0)")
    parser.add_argument("--Lambda_gp", type=float, default=10.0, help="coefficient for gradient penalty")
    parser.add_argument("--lr_encoder", type=float, default=1e-2, help="encoder learning rate")
    parser.add_argument("--lr_decoder", type=float, default=1e-2, help="decoder learning rate")
    parser.add_argument("--lr_critic", type=float, default=2e-4, help="critic learning rate")
    parser.add_argument("--beta1_encoder", type=float, default=0.5, help="encoder beta 1")
    parser.add_argument("--beta1_decoder", type=float, default=0.5, help="decoder beta 1")
    parser.add_argument("--beta1_critic", type=float, default=0.5, help="critic beta 1")
    parser.add_argument("--beta2_encoder", type=float, default=0.9, help="encoder beta 2")
    parser.add_argument("--beta2_decoder", type=float, default=0.9, help="decoder beta 2")
    parser.add_argument("--beta2_critic", type=float, default=0.9, help="critic beta 2")
    parser.add_argument("--test_batch_size", type=int, default=5000, help="test set batch size (default: 5000)")
    parser.add_argument("--load_mse_model_path", type=str, default=None, help="directory from which to preload enc1/dec1+disc1 models to start training at")
    parser.add_argument("--load_base_model_path", type=str, default=None, help="directory from which to preload enc1/dec1+disc1 models to start training at")
    parser.add_argument("--load_reduced_model_path", type=str, default=None, help="(for joint_reduced_reduced) directory from which to preload discriminator models to start training at")
    parser.add_argument("--initialize_base_discriminator", type=int, default=0, help="For refined or reduced models: whether to start from base model disc.")
    parser.add_argument("--initialize_mse_model", type=int, default=0, help="For base model: whether or not to continue training from Lambda=0 model.")
    parser.add_argument("--enc_layer_scale", type=float, default=1.0, help="Scale layer size of encoder by factor")
    parser.add_argument("--enc_2_layer_scale", type=float, default=1.0, help="Scale layer size of encoder 2 by factor")
    parser.add_argument("--reduced_dims", type=str, default='', help="Reduced dims")
    parser.add_argument("--reduced_path", type=str, help="If joint reduced training, where to save the secondary model")
    parser.add_argument("--refined_path", type=str, help="If joint refined training, where to save the secondary model")
    parser.add_argument("--dataset", type=str, default='mnist', help="dataset to use (default: mnist)")
    parser.add_argument("--progress_intervals", type=int, default=-1, help="periodically show progress of training")
    parser.add_argument("--entropy_intervals", type=int, default=-1, help="periodically calculate entropy of model. -1 only end, -2 for never")
    parser.add_argument("--submode", type=str, default=None, help="generic submode of mode")
    parser.add_argument("-mode", type=str, default='base', help="base, refined or reduced training mode")
    parser.add_argument("-experiment_path", type=str, help="name of the subdirectory to save")

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device]: {device}')
    # torch.manual_seed(1)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        vars(args)['input_size'] = 784
    elif args.dataset == 'svhn':
        vars(args)['input_size'] = 3*32*32
    else:
        raise ValueError(f'Invalid dataset: {args.dataset}')

    if args.mode == 'base':
        train_base(args, device)
    elif args.mode == 'refined':
        train_refined(args, device)
    elif args.mode == 'reduced':
        train_reduced(args, device)
    elif args.mode == 'joint_reduced':
        train_joint_reduced(args, device)
    elif args.mode == 'joint_refined':
        train_joint_refined(args, device)
    elif args.mode == 'joint_reduced_reduced':
        train_reduced_reduced(args, device)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')
