# Run these in "run_with_params.py".

# ====================================
# MNIST

# End-to-end models for Figure 4(a). Change latent_dim_1, L_1 = 3, 3 accordingly to get tradeoffs at various rates.
settings = []
experiment_number = 'M1' # This is the folder which will be created to contain the results
mode = 'base'            # Indicated end-to-end models
latent_dim_1, L_1 = 3, 3 # latent dim + quantization levels. Controls the rate (R = latent_dim_1*log2(L_1))
Lambda_base = [0, 0.0033, 0.005, 0.0066, 0.008, 0.01, 0.011, 0.013, 0.015] # Tradeoff points
for Lambda in Lambda_base:
    settings.append({'dataset': 'mnist', 'latent_dim_1': latent_dim_1, 'L_1': L_1,
                     'Lambda_base': Lambda, 'n_critic': 1, 'n_epochs': 30, 'progress_intervals': 6,
                     'enc_layer_scale': 1, 'initialize_mse_model': 0, 'test_batch_size': 5000})
# n_critic controls the number of discriminator iterations per generator (encoder) iteration
# progress_intervals periodically saves sample images. Training loss is recorded for every iteration regardless.
# enc_layer_scale changes the width of the encoder. It's not really needed.
# initialize_mse_model initializes model parameters from the model trained on MSE loss if 1. Not used.
train_with_params(mode, settings, experiment_number, overwrite=False)

# -----------------------

# For training universal models in Figure 4(a).
# Select dimensions to use, then run universal models.

# This code was originally designed to select a subset of the latent variables.
# In the universality experiments, the size of the subset is just the
# size of the original.
settings = []
experiment_number = 'M1'
mode = 'select'
method = 'identity'
latent_dim_1, L_1 = 3, 3 # set latent_dim_1, L_1 = 3, 4 in (1)
Lambda_base = 0.015
settings.append({'dataset': 'mnist', 'latent_dim_1': latent_dim_1, 'latent_dim_0': -1, 'L_1': L_1,
                 'Lambda_base': Lambda_base, 'Lambda_select': 0})
# latent_dim_0 selects the number of latent variables from the original model to use. If -1, use all.
# The selection procedure chooses the dimensions which minimize (distortion loss) + Lambda_select*(perception loss)
# If -1, all dimensions are selected so Lambda_select doesn't matter anyway. In this case, we are essentially
# reusing the encoder (which will have its weights frozen) and its outputs in full.
select_dims(mode, settings, method, experiment_number)

# This portion trains a new decoder on top of the encoder from (1) 
# of the same size as the decoder produced by the end-to-end model. Uses dimensions
# (latent_dim_0, L_0=L_1) selected by previous block.
settings = []
experiment_number = 'M1'
mode = 'reduced'
selection_method = 'identity'
latent_dim_1, L_1 = 3, 3
latent_dim_0, L_0 = 3, 3 # L_0 must be equal to L_1
Lambda_base = 0.015
Lambda_reduced = [0, 0.0025, 0.004, 0.005, 0.006, 0.008, 0.009, 0.01, 0.011, 0.013]
for Lambda in Lambda_reduced:
    settings.append({'dataset': 'mnist', 'latent_dim_1': latent_dim_1, 'latent_dim_0': latent_dim_0, 'L_1': L_1, 'L_0': L_0,
                     'Lambda_base': Lambda_base, 'Lambda_reduced': Lambda, 'n_epochs': 30, 'progress_intervals': 6,
                     'initialize_base_discriminator': 1, 'initialize_mse_model': 0, 'test_batch_size': 5000})
train_with_params(mode, settings, experiment_number, selection_method=selection_method)

# -----------------------

# Refinement model for Figure 6(a). Trains an auxilliary refining encoder 
# and decoder on top of the uuiversal encoder

settings = []
experiment_number = 'M1'
mode = 'refined'
latent_dim_1, L_1 = 3, 3
latent_dim_2, L_2 = 3, 3 # Refining encoder settings, 3 additional dimensions
Lambda_base = 0
Lambdas_refined = [0, 0.0025, 0.004, 0.005, 0.006, 0.008, 0.009, 0.01, 0.011, 0.013]
for Lambda_refined in Lambdas_refined:
    settings.append({'dataset': 'mnist', 
                     'latent_dim_1': latent_dim_1, 'L_1': L_1, 'Lambda_base': Lambda_base, 
                     'latent_dim_2': latent_dim_2, 'L_2': L_2, 'Lambda_refined': Lambda_refined,
                     'initialize_base_discriminator': 1, 'n_epochs': 30, 'test_batch_size': 5000, 
                     'progress_intervals': 6})
train_with_params(mode, settings, experiment_number)

# ====================================
# SVHN

# End-to-end models for Figure 4(c). Change latent_dim_1, L_1 = 10, 8 accordingly.

experiment_number = 'S1'
settings = []
mode = 'base'
latent_dim_1, L_1 = 10, 8
Lambda_bases = [0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002]
for Lambda_base in Lambda_bases:
    settings.append({'dataset': 'svhn', 'n_channel': 3, 'latent_dim_1': latent_dim_1, 'L_1': L_1,
                     'Lambda_base': Lambda_base, 'n_critic': 1, 'n_epochs': 80, 'progress_intervals': 10,
                     'enc_layer_scale': 1, 'initialize_mse_model': 0, 'test_batch_size': 5000,
                     'lr_encoder': 1e-4, 'lr_decoder': 1e-4, 'lr_critic': 1e-4,
                     'beta1_encoder': 0.5, 'beta1_decoder': 0.5, 'beta1_critic': 0.5,
                     'beta2_encoder': 0.999, 'beta2_decoder': 0.999, 'beta2_critic': 0.999})
train_with_params(mode, settings, experiment_number, overwrite=True)

# -----------------------

# For training universal models in Figure 4(c).

experiment_number = 'S1'
settings = []
mode = 'select'
method = 'identity'
latent_dim_1, L_1 = 10, 8
Lambda_base = 0.002
settings.append({'dataset': 'svhn', 'latent_dim_1': latent_dim_1, 'latent_dim_0': -1, 'L_1': L_1,
                 'Lambda_base': Lambda_base, 'Lambda_select': 0})
select_dims(mode, settings, method, experiment_number)

experiment_number = 'S1'
settings = []
mode = 'reduced'
selection_method = 'identity'
latent_dim_1, L_1 = 10, 8
latent_dim_0, L_0 = 10, 8
Lambda_base = 0.002
Lambdas_reduced = [0, 0.0003, 0.0005, 0.0008, 0.001, 0.0012, 0.0017]
for Lambda_reduced in Lambdas_reduced:
    settings.append({'dataset': 'svhn', 'n_channel': 3, 'latent_dim_1': latent_dim_1, 'latent_dim_0': latent_dim_0, 'L_1': L_1, 'L_0': L_0,
                        'Lambda_base': Lambda_base, 'Lambda_reduced': Lambda_reduced, 'n_critic': 1, 'n_epochs': 80, 'progress_intervals': 10,
                        'enc_layer_scale': 1.0, 'initialize_mse_model': 0, 'initialize_base_discriminator': 1, 'test_batch_size': 5000,
                        'lr_encoder': 1e-4, 'lr_decoder': 1e-4, 'lr_critic': 1e-4,
                        'beta1_encoder': 0.5, 'beta1_decoder': 0.5, 'beta1_critic': 0.5,
                        'beta2_encoder': 0.999, 'beta2_decoder': 0.999, 'beta2_critic': 0.999})
train_with_params(mode, settings, experiment_number, selection_method=selection_method)

# -----------------------

# Refined model example in Figure 6(c).
settings = []
experiment_number = 'S1'
mode = 'refined'
latent_dim_1, L_1 = 10, 8
latent_dim_2, L_2 = 10, 8
Lambda_base = 0.00
Lambdas_refined = [0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002]
for Lambda_refined in Lambdas_refined:
    settings.append({'dataset': 'svhn', 'n_channel': 3, 'latent_dim_1': latent_dim_1, 'latent_dim_2': latent_dim_2, 'L_1': L_1, 'L_2': L_2,
                     'Lambda_base': Lambda_base, 'Lambda_refined': Lambda_refined, 'n_critic': 1, 'n_epochs': 80, 'progress_intervals': 10,
                     'enc_2_layer_scale': 1, 'initialize_mse_model': 0, 'initialize_base_discriminator': 1, 'test_batch_size': 5000,
                     'lr_encoder': 1e-4, 'lr_decoder': 1e-4, 'lr_critic': 1e-4,
                     'beta1_encoder': 0.5, 'beta1_decoder': 0.5, 'beta1_critic': 0.5,
                     'beta2_encoder': 0.999, 'beta2_decoder': 0.999, 'beta2_critic': 0.999})
train_with_params(mode, settings, experiment_number)