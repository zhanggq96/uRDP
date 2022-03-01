import os
import subprocess
import datetime
import time
import json
import shutil


def run(experiment_path_pre, settings, commands):
    with open(f'{experiment_path_pre}/_status.tmp', 'w') as f:
        now = datetime.datetime.now()
        f.write(now.strftime(f"Experiment: %d/%m/%Y %H:%M:%S\n"))
        f.write('exp,' + ','.join(f'{key}' for key, value in settings[0].items()) + f',timing\n')

    for i, (setting, command) in enumerate(zip(settings, commands)):
        print(f'--- Beginning experiment {i} ---')
        start_time = time.process_time()
        subprocess.call(command, shell=True) # The experiment will save results on its own
        end_time = time.process_time()
        minutes = (end_time - start_time) / 60

        with open(f'{experiment_path_pre}/_status.tmp', 'a+') as f:
            f.write(f'{i},' + ','.join(f'{value}' for key, value in setting.items()) + f',{minutes}\n')

def train_with_params(mode, settings, experiment_number, selection_method=None, submode=None, overwrite=False):

    commands = []
    experiment_path = ''
    experiment_path_pre = f'experiments/{experiment_number}'
    shutil.copy('models.py', experiment_path_pre)

    for setting in settings:
        # if setting['dataset'] == 'mnist' and setting['n_epochs'] != 30:
        #     raise ValueError(f'mnist epochs: ? {setting["n_epochs"]}')
        # elif setting['dataset'] == 'fashion_mnist' and setting['n_epochs'] != 50:
        #     raise ValueError(f'fashion_mnist epochs: ? {setting["n_epochs"]}')

        experiment_path_base = f'{setting["latent_dim_1"]}-{setting["L_1"]}'
        base_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_base'])

        if mode == 'base':
            experiment_path = os.path.join(experiment_path_pre,
                                           experiment_path_base,
                                           base_Lambda_dir)
            if setting['Lambda_base'] > 0:
                # experiment_path = os.path.join(experiment_path_base, '_MSE')
                # setting['load_mse_model_dir'] = None
                setting['load_mse_model_path'] = os.path.join(experiment_path_pre,
                                                             experiment_path_base,
                                                             'Lambda=0.00000')
        elif mode == 'refined':
            refined_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_refined'])
            experiment_path = os.path.join(experiment_path_pre,
                                           experiment_path_base,
                                           base_Lambda_dir,
                                           f'{setting["latent_dim_2"]}-{setting["L_2"]}-[refined]',
                                           refined_Lambda_dir)
            setting['load_base_model_path'] = os.path.join(experiment_path_pre,
                                                          experiment_path_base,
                                                          base_Lambda_dir)
        elif mode == 'reduced':
            assert isinstance(selection_method, str)
            reduced_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_reduced'])
            experiment_path = os.path.join(experiment_path_pre,
                                           experiment_path_base,
                                           base_Lambda_dir,
                                           f'{setting["latent_dim_0"]}-{setting["L_0"]}-[{selection_method}]',
                                           reduced_Lambda_dir)
            setting['load_base_model_path'] = os.path.join(experiment_path_pre,
                                                          experiment_path_base,
                                                          base_Lambda_dir)
            with open(os.path.join(setting['load_base_model_path'],
                                   '_latent_selection.json'), 'r') as f:
                data = json.load(f)
                # selection_method = setting['selection_method']
                all_selected_dims = data[selection_method]['latent_selected'] # List of all selected dims
                reduced_dims = all_selected_dims[:setting['latent_dim_0']] # Choose how many
            reduced_dims = ','.join([str(dim) for dim in reduced_dims])
            setting['reduced_dims'] = reduced_dims
        elif mode == 'joint_reduced':
            assert setting['Lambda_reduced'] == setting['Lambda_base'] # otherwise will overwrite base file
            experiment_path_base += f'-{setting["latent_dim_0"]}-{setting["L_0"]}' # z_dim_0-L_0-z_dim_1-L_1
            reduced_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_reduced'])
            experiment_path = os.path.join(experiment_path_pre,
                                           experiment_path_base,
                                           base_Lambda_dir)
            reduced_path = os.path.join(experiment_path,
                                        f'{setting["latent_dim_0"]}-{setting["L_0"]}-[joint_reduced]',
                                        reduced_Lambda_dir)
            setting['reduced_path'] = reduced_path
        elif mode == 'joint_refined':
            assert setting['Lambda_refined'] == setting['Lambda_base'] # otherwise will overwrite base file
            experiment_path_base += f'-{setting["latent_dim_2"]}-{setting["L_2"]}' # z_dim_0-L_0-z_dim_1-L_1
            refined_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_refined'])
            experiment_path = os.path.join(experiment_path_pre,
                                           experiment_path_base,
                                           base_Lambda_dir)
            refined_path = os.path.join(experiment_path,
                                        f'{setting["latent_dim_2"]}-{setting["L_2"]}-[joint_refined]',
                                        refined_Lambda_dir)
            setting['refined_path'] = refined_path
        elif mode == 'joint_reduced_reduced':
            assert isinstance(submode, str), f'{submode}'
            setting['submode'] = submode
            assert setting['Lambda_reduced'] == setting['Lambda_base']
            # Reduce the higher rate models produced by joint reduced training
            # Only works for reducing to same dimension (i.e. replacing with a new decoder)
            # So the field specifying the new settings (latent_dim_-1, L_-1) is equal to (latent_dim_0, L_0).
            # The fields (latent_dim_0, L_0) are used to indicate the dimension of the old model.

            experiment_path_base += f'-{setting["latent_dim_0"]}-{setting["L_0"]}' # z_dim_0-L_0-z_dim_1-L_1
            reduced_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_reduced'])
            reduced_reduced_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_reduced_reduced'])
            setting['load_base_model_path'] = os.path.join(experiment_path_pre,
                                                           experiment_path_base,
                                                           base_Lambda_dir)
            if submode == 'joint_reduced_reduced_1':
                assert (setting['latent_dim_M1'], setting['L_M1']) == (setting["latent_dim_1"], setting["L_1"]) # only use for reducing to same dim for now
                experiment_path = os.path.join(setting['load_base_model_path'],
                                               f'{setting["latent_dim_M1"]}-{setting["L_M1"]}-[joint_reduced_reduced_1]',
                                               reduced_reduced_Lambda_dir)
            elif submode == 'joint_reduced_reduced_0':
                assert (setting['latent_dim_M1'], setting['L_M1']) == (setting["latent_dim_0"], setting["L_0"]) # only use for reducing to same dim for now
                setting['load_reduced_model_path'] = os.path.join(setting['load_base_model_path'],
                                                                  f'{setting["latent_dim_0"]}-{setting["L_0"]}-[joint_reduced]',
                                                                  reduced_Lambda_dir,)
                experiment_path = os.path.join(setting['load_reduced_model_path'],
                                               f'{setting["latent_dim_M1"]}-{setting["L_M1"]}-[joint_reduced_reduced_0]',
                                               reduced_reduced_Lambda_dir)
            # print(setting['load_base_model_path'])
            # if submode == 'joint_reduced_reduced_0': print(setting['load_reduced_model_path'])
            # print(experiment_path)
            # exit()
        if not overwrite and os.path.exists(experiment_path):
            raise ValueError('Overwritting!')

        commands.append(
            'python train.py ' + ' '.join(f'--{key} {value}' for key, value in setting.items()) \
                + f' -experiment_path {experiment_path}' + f' -mode {mode}'
        )

    for command in commands:
        print(command)
    print('Number of commands to execute:', len(commands))

    with open(f'{experiment_path_pre}/_status.tmp', 'w') as f:
        now = datetime.datetime.now()
        f.write(now.strftime(f"Experiment: %d/%m/%Y %H:%M:%S\n"))
        f.write('exp,' + ','.join(f'{key}' for key, value in settings[0].items()) + f',timing\n')

        total_time = 0
    for i, (setting, command) in enumerate(zip(settings, commands)):
        print(f'--- Beginning experiment {i} ---')
        start_time = time.process_time()
        subprocess.call(command, shell=True) # The experiment will save results on its own
        end_time = time.process_time()
        minutes = (end_time - start_time) / 60
        total_time += minutes

        with open(f'{experiment_path_pre}/_status.tmp', 'a+') as f:
            f.write(f'{i},' + ','.join(f'{value}' for key, value in setting.items()) + f',{minutes}\n')

    with open(f'{experiment_path_pre}/_status.tmp', 'a+') as f:
        f.write(f'Total time taken (minutes): {total_time}')

    print(f'Finished running experiment {experiment_number}')

def select_dims(mode, settings, method, experiment_number):
    commands = [] # ['dir', 'echo foo']*5
    experiment_path_pre = f'experiments/{experiment_number}'
    for setting in settings:
        base_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_base'])
        experiment_path_base = f'{setting["latent_dim_1"]}-{setting["L_1"]}'
        load_model_dir = os.path.join(experiment_path_pre,
                                      experiment_path_base,
                                      base_Lambda_dir)

        commands.append(
            'python latent_selection.py ' + ' '.join(f'--{key} {value}' for key, value in setting.items()) \
                + f' -load_base_model_path {load_model_dir}' + f' -method {method}'
        )

    for command in commands:
        print(command)
    print(len(commands))

    with open(f'{experiment_path_pre}/_status.tmp', 'w') as f:
        now = datetime.datetime.now()
        f.write(now.strftime(f"Experiment: %d/%m/%Y %H:%M:%S\n"))
        f.write('exp,' + ','.join(f'{key}' for key, value in settings[0].items()) + f',timing\n')

    for i, (setting, command) in enumerate(zip(settings, commands)):
        print(f'--- Beginning experiment {i} ---')
        start_time = time.process_time()
        subprocess.call(command, shell=True) # The experiment will save results on its own
        end_time = time.process_time()
        minutes = (end_time - start_time) / 60

        with open(f'{experiment_path_pre}/_status.tmp', 'a+') as f:
            f.write(f'{i},' + ','.join(f'{value}' for key, value in setting.items()) + f',{minutes}\n')

def evaluate_quantized_entropy(mode, settings, method, experiment_number):
    commands = []
    experiment_path_pre = f'experiments/{experiment_number}'
    for setting in settings:
        experiment_path_base = f'{setting["latent_dim_1"]}-{setting["L_1"]}'
        base_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_base'])
        load_base_model_path = os.path.join(experiment_path_pre,
                                           experiment_path_base,
                                           base_Lambda_dir)
        setting['load_base_model_path'] = load_base_model_path
        if mode == 'base':
            setting['experiment_path'] = load_base_model_path
        else:
            pass

        commands.append(
            'python quantizer_entropy.py ' + ' '.join(f'--{key} {value}' for key, value in setting.items()) \
                + f' -mode {mode}' + f' -method {method}'
        )

    for command in commands:
        print(command)
    print(len(commands))

    run(experiment_path_pre, settings, commands)
    print(f'Done evaluate_quantized_entropy({mode}, {str(settings)[:25]}... , {experiment_number})')

def evaluate_latent_variance(mode, settings, experiment_number):
    commands = []
    experiment_path_pre = f'experiments/{experiment_number}'
    for setting in settings:
        experiment_path_base = f'{setting["latent_dim_1"]}-0'
        base_Lambda_dir = 'Lambda={:.5f}'.format(setting['Lambda_base'])
        load_base_model_path = os.path.join(experiment_path_pre,
                                            experiment_path_base,
                                            base_Lambda_dir)
        setting['load_base_model_path'] = load_base_model_path
        if mode == 'base':
            setting['experiment_path'] = load_base_model_path
        else:
            pass

        commands.append(
            'python latent_variance.py ' + ' '.join(f'--{key} {value}' for key, value in setting.items()) \
                + f' -mode {mode}'
        )

    for command in commands:
        print(command)
    print(len(commands))

    run(experiment_path_pre, settings, commands)
    print(f'Done evaluate_latent_variance({mode}, {str(settings)[:25]}... , {experiment_number})')


if __name__ == '__main__':

    # Base mode example FOR USUAL BASE MODEL
    settings = []
    experiment_number = 'EB/M1'
    mode = 'base'
    latent_dim_1, L_1 = 3, 3
    Lambda_base = [0.015,]
    for Lambda in Lambda_base:
        settings.append({'dataset': 'mnist', 'latent_dim_1': latent_dim_1, 'L_1': L_1,
                         'Lambda_base': Lambda, 'n_critic': 1, 'n_epochs': 30, 'progress_intervals': 6,
                         'enc_layer_scale': 1, 'initialize_mse_model': 0, 'test_batch_size': 5000})
    # train_with_params(mode, settings, experiment_number, overwrite=True)
    train_with_params(mode, settings, experiment_number, overwrite=False)