import argparse
import sys
from timeit import default_timer as timer
from typing import List, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torch.optim as optim
from tqdm import trange, tqdm

from channel_generators import sample_channel_realizations
from minn import construct_minn, Minn, TransmissionVariables
from parameters import Parameters, change_param_value
from utils_misc import dBm_to_Watt
from utils_torch import DataAndChannelsLoader, select_torch_device
from utils_training import load_MNIST_data, load_FashionMNIST_data, load_KMNIST_data, load_CIFAR10_data, \
    SingleTrainingLogger


def load_channels(params: Parameters):
    rng   = params.Auxiliary.rng if params.Auxiliary.rng != 'auto' else np.random.default_rng()
    dtype = params.Auxiliary.complex_dtype

    H_tx_ris_train, H_ris_rx_train, H_tx_rx_train = sample_channel_realizations(params,
                                                                                params.Training.preload_channels,
                                                                                params.Training.verbose_level > 0,
                                                                                rng)
    H_tx_ris_val, H_ris_rx_val, H_tx_rx_val = sample_channel_realizations(params,
                                                                          params.Training.preload_channels_val,
                                                                          params.Training.verbose_level > 0,
                                                                          rng)
    train_datasets = [
        TensorDataset(torch.from_numpy(H_tx_ris_train).to(dtype)),
        TensorDataset(torch.from_numpy(H_ris_rx_train).to(dtype)),
        TensorDataset(torch.from_numpy(H_tx_rx_train).to(dtype))
    ]
    val_datasets = [
        TensorDataset(torch.from_numpy(H_tx_ris_val).to(dtype)),
        TensorDataset(torch.from_numpy(H_ris_rx_val).to(dtype)),
        TensorDataset(torch.from_numpy(H_tx_rx_val).to(dtype))
    ]
    return train_datasets, val_datasets


def load_data(params: Parameters) -> Tuple[DataAndChannelsLoader, DataAndChannelsLoader]:
    train_channel_datasets, val_channel_datasets = load_channels(params)

    if params.Training.dataset == 'MNIST':
        trainloader, testloader, _, _ = load_MNIST_data(params.Training.batch_size, params.Paths.data_rootdir)
        change_param_value('Auxiliary.data_shape', (1, 28, 28) )

    elif params.Training.dataset == 'FMNIST':
        trainloader, testloader, _, _ = load_FashionMNIST_data(params.Training.batch_size, params.Paths.data_rootdir)
        change_param_value('Auxiliary.data_shape', (1, 28, 28) )

    elif params.Training.dataset == 'KMNIST':
        trainloader, testloader, _, _ = load_KMNIST_data(params.Training.batch_size, params.Paths.data_rootdir)
        change_param_value('Auxiliary.data_shape', (1, 28, 28) )

    elif params.Training.dataset == 'CIFAR10':
        trainloader, testloader, _, _ = load_CIFAR10_data(params.Training.batch_size, params.Paths.data_rootdir)
        change_param_value('Auxiliary.data_shape', (3, 32, 32) )

    else:
        raise NotImplementedError(f'Unknown dataset name "{params.Training.dataset}"')

    data_channels_train_loader = DataAndChannelsLoader(trainloader, train_channel_datasets)
    data_channels_val_loader = DataAndChannelsLoader(testloader, val_channel_datasets)

    return data_channels_train_loader, data_channels_val_loader


def determine_current_power_value(params, epoch_num, verbose):
    current_P_dBm    = params.Channels.P
    schedule         = params.Training.P_value_schedule_dBm
    scheduled_epochs = sorted(schedule.keys(), reverse=True)

    for scheduled_epoch in scheduled_epochs:
        if epoch_num >= scheduled_epoch:
            current_P_dBm = schedule[scheduled_epoch]
            break

    if verbose >= 3 and epoch_num > 0:
        prev_P_dBm = params.Channels.P
        for scheduled_epoch in scheduled_epochs:
            if (epoch_num - 1) >= scheduled_epoch:
                prev_P_dBm = schedule[scheduled_epoch]
                break

        if current_P_dBm != prev_P_dBm:
            tqdm.write(f"Epoch {epoch_num}: Transmission power set to {current_P_dBm:} dBm.")

    P_watt = dBm_to_Watt(current_P_dBm)
    return P_watt


def apply_noise_to_channel(params, C):
    if params.Channels.csi_noise_dB is None:
        return C

    R_dB               = params.Channels.csi_noise_dB
    R_linear_magnitude = 10 ** (R_dB / 20.0)
    norm_C             = torch.linalg.norm(C)
    norm_N_desired     = norm_C / R_linear_magnitude
    N_0_real           = torch.randn_like(C.real) * (0.5 ** 0.5)
    N_0_imag           = torch.randn_like(C.real) * (0.5 ** 0.5)
    N_0                = torch.complex(N_0_real, N_0_imag)
    norm_N_0           = torch.linalg.norm(N_0)
    scaling_N          = norm_N_desired / norm_N_0
    N                  = scaling_N * N_0
    C_temp             = C + N
    norm_C_temp        = torch.linalg.norm(C_temp)
    alpha_scaling      = norm_C / norm_C_temp
    C_noise            = alpha_scaling * C_temp
    return C_noise

def prepare_transmission_batch(params, epoch, batch_data_and_channels, P_curr, device):
    transmission_vars = TransmissionVariables(
        inputs   = batch_data_and_channels[0],
        targets  = batch_data_and_channels[1],
        H_ue_ris = batch_data_and_channels[2],
        H_ris_bs = batch_data_and_channels[3],
        H_ue_bs  = batch_data_and_channels[4],
        P_curr   = P_curr,
    )
    transmission_vars.H_ue_bs_noise  = apply_noise_to_channel(params, transmission_vars.H_ue_bs)
    transmission_vars.H_ue_ris_noise = apply_noise_to_channel(params, transmission_vars.H_ue_ris)
    transmission_vars.H_ris_bs_noise = apply_noise_to_channel(params, transmission_vars.H_ris_bs)
    transmission_vars                = transmission_vars.to(device)
    return transmission_vars



def train_model(params                    : Parameters,
                minn                      : Minn,
                data_channels_train_loader: DataAndChannelsLoader,
                data_channels_val_loader  : DataAndChannelsLoader,
                device                    : torch.device,
                logger                    : SingleTrainingLogger,
                verbose                   : int=5):

    minn      = minn.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(minn.parameters(recurse=True),
                           lr=params.Training.learning_rate,
                           weight_decay=params.Training.weight_decay,
                           )

    if verbose > 3: print('Started training. You may press Ctrl+C anytime after the first epoch to finish early.')

    num_epochs      = params.Training.epochs
    best_acc        = -1
    P_best_acc      = -1
    last_train_loss = -1.
    iterator        = range(num_epochs) if verbose < 2 else trange(num_epochs, desc="Training")

    for epoch in iterator:
        try:
            minn.train()
            P_curr     = determine_current_power_value(params, epoch, verbose)
            train_loss = 0.

            for batch_i, batch_data_and_channels in enumerate(data_channels_train_loader):
                transmission_vars = prepare_transmission_batch(params, epoch, batch_data_and_channels, P_curr, device)
                y                 = minn(transmission_vars)
                loss              = criterion(y, transmission_vars.targets)
                batch_loss        = loss.detach().cpu().item()
                train_loss       += batch_loss / len(data_channels_train_loader)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            last_train_loss = train_loss

            val_loss, val_metric = evaluate_model(params, minn, data_channels_val_loader, P_curr, epoch, criterion, device)
            logger.log_epoch(epoch, acc=val_metric, train_loss=train_loss, val_loss=val_loss, P_curr=P_curr)

            if verbose >= 3 and (epoch + 1) % params.Training.epoch_print_freq == 0:
                tqdm.write(f"Epoch {epoch + 1}/{num_epochs} train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_metric:.3f}")

            if val_metric > best_acc:
                best_acc   = val_metric
                P_best_acc = P_curr

        except KeyboardInterrupt:
            print('\nTraining aborted by user...')
            break

    logger.log_final(best_acc, P_best_acc, final_acc=val_metric, P_at_final_acc=P_curr, final_iter=epoch)

    return minn, best_acc, last_train_loss




def evaluate_model(params                    : Parameters,
                   minn                      : Minn,
                   data_channels_val_loader  : DataAndChannelsLoader,
                   P_curr                    : float,
                   epoch                     : int,
                   criterion                 : Callable,
                   device                    : torch.device,
                   ):
    minn.eval()
    val_loss    = 0.
    val_correct = 0
    val_total   = 0
    with torch.no_grad():
        for batch_i, batch_data_and_channels in enumerate(data_channels_val_loader):
            transmission_vars = prepare_transmission_batch(params, epoch, batch_data_and_channels, P_curr, device)
            y                 = minn(transmission_vars)
            loss              = criterion(y, transmission_vars.targets)
            batch_loss        = loss.detach().cpu().item()
            val_loss         += batch_loss / len(data_channels_val_loader)
            pred              = y.argmax(dim=1)
            val_correct      += pred.eq(transmission_vars.targets).sum().item()
            val_total        += transmission_vars.targets.size(0)

        val_accuracy = val_correct / val_total

    return val_loss, val_accuracy

def configure_logger_setup_info(params: Parameters):
    logger_info = {
        'Nt'         : params.Channels.Nt,
        'Nr'         : params.Channels.Nr,
        'N'          : params.Channels.N,
        'MS'         : params.MINN.metasurface_type,
        'P'          : params.Channels.P,
        'dataset'    : params.Training.dataset,
        'CSI'        : params.MINN.csi_knowledge,
        'Control'    : params.MINN.metasurface_control,
        'Sim_layers' : params.Channels.n_sim_layers,
        'TpF'        : params.Channels.TpF,
        'CSI_noise'  : params.Channels.csi_noise_dB,
    }

    if params.Training.verbose_level >= 2:
        max_len = max(len(key) for key in logger_info)
        title   = 'System Parameters'
        width   = max(max_len + 5 + 10, len(title) + 4)
        print(f"\n{'-' * width}")
        print(f" {title}")
        print(f"{'-' * width}")
        for key, value in logger_info.items():
            print(f" {key.ljust(max_len)} : {value}")
        print(f"{'-' * width}\n")

    return logger_info


def change_parameter_values_from_command_line_arguments():
    def cast_value(value):
        if value == "True": return True
        if value == "False": return False
        if value == "None": return None
        try:
            f = float(value)
            if f == int(f): return int(f)
            return f
        except ValueError:
            return value

    parser = argparse.ArgumentParser(description="Change parameters and run training.")
    parser.add_argument('params', nargs='*', help="Parameter name and value pairs (e.g., Channels.Nt 8 Training.dataset MNIST)")
    args = parser.parse_args()

    if len(args.params) % 2 != 0:
        print("Error: Each parameter must be followed by a value.")
        sys.exit(1)

    for i in range(0, len(args.params), 2):
        param_name = args.params[i]
        value = args.params[i + 1]
        casted_value = cast_value(value)
        change_param_value(param_name, casted_value)


def main():
    change_parameter_values_from_command_line_arguments()
    start_time                                           = timer()
    params                                               = Parameters()
    rng                                                  = np.random.default_rng(params.Auxiliary.numpy_seed)
    change_param_value('Auxiliary.rng', rng)
    data_channels_train_loader, data_channels_val_loader = load_data(params)
    minn                                                 = construct_minn(params)
    device                                               = select_torch_device(params.Training.preferred_device, params.Training.verbose_level>0)
    system_params_to_log                                 = configure_logger_setup_info(params)
    logger                                               = SingleTrainingLogger(params, system_params_to_log, verbose=params.Training.verbose_level>0)
    minn, best_acc, final_train_loss                     = train_model(params, minn,
                                                                       data_channels_train_loader,
                                                                       data_channels_val_loader,
                                                                       device,
                                                                       logger,
                                                                       params.Training.verbose_level)
    end_time                                             = timer()
    elapsed_minutes                                      = int((end_time - start_time) / 60)

    logger.save_training_results()
    logger.save_final_results()
    logger.plot_training_curve()

    if params.Training.verbose_level > 0:
        print(f'Done...\nFinal train loss: {final_train_loss:.4f}\nBest accuracy: {best_acc:.3f}')
    if params.Training.verbose_level > 1:
        print(f'[Took {elapsed_minutes} mins.]')


# pip install torch torchvision numpy scipy tqdm pandas matplotlib
if __name__ == '__main__':
    main()
