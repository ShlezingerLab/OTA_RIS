from typing import Any
import numpy as np
from scipy.constants import speed_of_light
import torch


class Parameters:

    class Channels:
        Nt                         = 8                      # Transmit antennas
        Nr                         = 12                       # Receive antennas
        P                          = 30                     # Transmission power (per TX antenna symbol) in dBm. Note that this value may change dynamically during training.
        N                          = 8*8                    # Elements in RIS or per SIM layer. Only used if `metasurface_type = 'RIS'` or `metasurface_type ='SIM'`. The elements are arranged in a rectangular-like grid automatically
        n_sim_layers               = 3                      # Number of SIM layers. Only used if `metasurface_type = 'SIM'`
        TpF                        = 1                      # Transmissions per channel frame: `TpF * Nt` are produced at the Encoder module which are transmitted over the same channel realization, emulating multiple transmissions.
        csi_noise_dB               = None                   # If not set to None, it applies AWGN with variance of the specified value to the channel instances the modules observe

        freq                       = 28e9                   # Hz
        noise_sigma_sq             = -90                    # dBm
        pathloss_exp               = 2
        tx_rx_rice_factor          = 3                      # dB
        tx_ris_rice_factor         = 13
        ris_rx_rice_factor         = 7
        extra_tx_rx_attenuation_dB = None                   # If nor None, it applied extra attenuation to the TX-RX link
        tx_position                = np.array([-2, +2, -0.5])
        rx_position                = np.array([10, 16, 4.])
        ris_position               = np.array([0, 0, 0])
        sim_layers_distance        = 5                      # Value multiplied by λ. The SIM model does not hold for small distances (eg < 10 * `sim_elem_width`). Larger distances (eg > 30 *`sim_elem_width`) may result to severe attenuation and NaN.
        sim_elem_width             = 0.5                    # Value multiplied by λ. Implies the element area as `sim_elem_width`^2. If set to >λ/2, it leads to superficial signal amplification.


    class MINN:
        csi_knowledge              = 'agnostic'                # Use 'aware' or 'agnostic' depending on whether the Encoder and Decoder modules will receive CSI as input.
        metasurface_type           = None                 # Use: 'RIS' or 'SIM' or None (for baseline)
        metasurface_control        = 'static'       # Use 'reconfigurable' or 'static' depending on whether a controller module for reconfigurable MSs will be used or MS weights will be treated as DNN weights


    class Training:
        method_name                = 'testing'
        dataset                    = 'MNIST'                # Use 'MNIST' or 'KMNIST' or 'FMNIST' or 'CIFAR10'. Changes the DNN structure of the Encoder and Decoder modules.
        learning_rate              = 1e-4
        weight_decay               = 1e-7                   # Equivalent to regularization
        epochs                     = 400
        batch_size                 = 256
        preferred_device           = 'cuda'                 # 'cuda' or 'cpu' (or maybe 'mps' for Mac but some operations may not be supported)
        preload_channels           = 10000                  # Number of channel realizations used for training
        preload_channels_val       = 3000                   # Number of channel realizations used for testing/validation
        verbose_level              = 5
        epoch_print_freq           = 1                      # Print train/val loss and accuracy every `epoch_print_freq` epochs

        P_value_schedule_dBm       = {                      # Dynamically change the value of transmission power `P` during training epochs. Format: [epoch number] : [P value in dB]
            300+00: 30,
            300+10: 25,
            300+20: 20,
            300+30: 15,
            300+40: 10,
            300+50: 5,
            300+60: 0,
            300+70: -5,
            300+80: -10,
            300+90: -15
        }


    class Paths:
        data_rootdir               = './data'
        output_rootdir             = './outputs/'
        output_plots_subdir        = 'training_plots/'   # Use None to deactivate saving output plots to save disk space
        results_file               = 'results.csv'
        training_log_file          = 'training_log.json'

    class Auxiliary:
        complex_dtype              = torch.complex64  # keep this - many pytorch functions are not implemented for other complex types
        rng                        = 'auto'           # do not change. value is determined automatic
        data_shape                 = 'auto'           # do not change. value is determined automatic
        numpy_seed                 = 1234





    @staticmethod
    def wavelength():
        return speed_of_light / Parameters.Channels.freq


def change_param_value(param_name: str, new_value: Any) -> None:
    """
    Change the value of a specified parameter in the Parameters class.

    Parameters:
    - param_name: A string specifying the parameter in the format 'ClassName.VarName'.
    - new_value: The new value to set for the parameter.
    """
    try:
        class_name, var_name = param_name.split('.')
        cls = getattr(Parameters, class_name)
        setattr(cls, var_name, new_value)
    except AttributeError as e:
        print(f"Error: {e}. Ensure '{param_name}' is in the format 'ClassName.VarName' and exists in Parameters.")
    except ValueError as e:
        print(f"Error: {e}. Ensure '{param_name}' is in the format 'ClassName.VarName'.")
