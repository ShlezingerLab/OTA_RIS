from typing import Union

import torch
import torch.nn as nn
import copy
from dataclasses import dataclass, fields
from typing import Optional, Any
import numpy as np

from parameters import Parameters, change_param_value
from utils_misc import split_to_close_to_square_factors, dBm_to_Watt
from utils_torch import matmul_with_diagonal_right, sample_awgn_torch


@dataclass
class TransmissionVariables:
    """
    A class containing all variables that are needed for the forward pass of the MINN.
    """
    H_ue_bs:          Optional[torch.Tensor] = None
    H_ue_ris:         Optional[torch.Tensor] = None
    H_ris_bs:         Optional[torch.Tensor] = None
    inputs:           Optional[torch.Tensor] = None
    targets:          Optional[torch.Tensor] = None
    transmit_signal:  Optional[torch.Tensor] = None
    received_signal:  Optional[torch.Tensor] = None
    signal_at_ris:    Optional[torch.Tensor] = None
    cascaded_channel: Optional[torch.Tensor] = None
    full_channel:     Optional[torch.Tensor] = None
    H_ue_bs_noise:    Optional[torch.Tensor] = None
    H_ue_ris_noise:   Optional[torch.Tensor] = None
    H_ris_bs_noise:   Optional[torch.Tensor] = None
    P_curr:           Optional[Union[float,torch.Tensor]] = None

    def set(self, **kwargs):
        """
        Safely update attributes.
        """
        valid_fields = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                raise KeyError(f"'{key}' is not a valid field of TransmissionVariables")
        return self

    def to(self, device):
        """
        Transfer all tensors to a specified torch device.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                setattr(self, field.name, value.to(device))
        return self

    def to_numpy(self, inplace=False):
        """
        Convert all tensors to numpy arrays.
        """
        target_obj = self if inplace else copy.deepcopy(self)

        for field in fields(target_obj):
            value = getattr(target_obj, field.name)
            if torch.is_tensor(value):
                setattr(target_obj, field.name, value.cpu().detach().numpy())

        return target_obj






class Minn(torch.nn.Module):
    def __init__(self,
                 parameters   : Parameters,
                 encoder      : nn.Module,
                 MS_controller: Union[nn.Module, None],
                 decoder      : nn.Module):
        super(Minn, self).__init__()

        self.params        = parameters
        self.encoder       = encoder
        self.decoder       = decoder
        self.MS_controller = MS_controller
        self.MS_variation  = self.params.MINN.metasurface_type
        self.Nt            = self.params.Channels.Nt
        self.Nr            = self.params.Channels.Nr
        self.N             = self.params.Channels.N
        self.TpF           = self.params.Channels.TpF
        noise_var          = dBm_to_Watt(self.params.Channels.noise_sigma_sq)
        self.noise_std     = torch.tensor(np.sqrt(noise_var))

        if self.MS_variation == 'SIM':
            from metasurface_modules import ReconfigurableSimNet, RisLayer
            n_x, n_y                   = split_to_close_to_square_factors(self.N)
            lam                        = self.params.wavelength()
            self.SIM_propagation_model = ReconfigurableSimNet(
                layers=[RisLayer(n_x,n_y) for _ in range(self.params.Channels.n_sim_layers)],
                layer_dist=lam/2,
                wavelength=lam,
                elem_area=(self.params.Channels.sim_elem_width * lam) ** 2,
                elem_dist=self.params.Channels.sim_layers_distance * lam,
                inp_shape=(self.params.Channels.TpF, self.params.Channels.Nr, self.params.Channels.N),
                layers_orientation_plane='yz',
                first_layer_central_coords=self.params.Channels.ris_position,
                complex_dtype=self.params.Auxiliary.complex_dtype
            )
        else:
            self.SIM_propagation_model = None

        if self.MS_variation == 'RIS':
            self.N_tot = self.N
        elif self.MS_variation == 'SIM':
            self.N_tot = self.N * self.params.Channels.n_sim_layers
        else:
            self.N_tot = None
            self.N     = None


    def _apply_tx_coding_and_power_norm(self, tv: TransmissionVariables) -> TransmissionVariables:
        s                  = self.encoder(tv)
        #n_symbols          = s.shape[-1]                                     # for debug only
        s_norm             = torch.norm(s, dim=2, keepdim=True)
        s_unit_mag         = s / s_norm
        s_scaled           = tv.P_curr * s_unit_mag
        #_s_scaled_norm     = torch.norm(s_scaled, dim=2, keepdim=False)      # for debug only
        s_scaled           = s_scaled.unsqueeze(3)                            # (B, T, Nt, 1)
        s_scaled           = s_scaled.to(self.params.Auxiliary.complex_dtype)

        tv.set(transmit_signal=s_scaled)
        return tv


    def _fix_channel_dimensions(self, tv: TransmissionVariables) -> TransmissionVariables:
        T                 = tv.transmit_signal.shape[1]                        # T: TpF (Transmissions per Frame)
        tv.H_ue_bs        = tv.H_ue_bs.unsqueeze(1).repeat(1, T, 1, 1)         # (b, T, Nr, Nt)
        tv.H_ue_ris       = tv.H_ue_ris.unsqueeze(1).repeat(1, T, 1, 1)        # (b, T, N, Nt)
        tv.H_ris_bs       = tv.H_ris_bs.unsqueeze(1).repeat(1, T, 1, 1)        # (b, T, Nr, N)
        tv.H_ue_bs_noise  = tv.H_ue_bs_noise.unsqueeze(1).repeat(1, T, 1, 1)   # (b, T, Nr, Nt)
        tv.H_ue_ris_noise = tv.H_ue_ris_noise.unsqueeze(1).repeat(1, T, 1, 1)  # (b, T, N, Nt)
        tv.H_ris_bs_noise = tv.H_ris_bs_noise.unsqueeze(1).repeat(1, T, 1, 1)  # (b, T, Nr, N)
        return tv


    def _apply_RIS_cascaded_channel(self, tv: TransmissionVariables) -> TransmissionVariables:
        if self.MS_variation != 'RIS': raise RuntimeError('This function is only for a MINN using a RIS.')

        T         = tv.transmit_signal.shape[1] if tv.transmit_signal is not None else 1
        ris_conf  = self.MS_controller(tv)                               # (b, N)
        phi       = torch.exp(-1j*ris_conf)
        phi       = phi.unsqueeze(1).repeat(1, T, 1)                      # (b, T, N)
        C_casc    = matmul_with_diagonal_right(tv.H_ris_bs, phi)          # C_ris_rx @ Phi : (b, T, Nr, N)
        C_casc    = torch.matmul(C_casc, tv.H_ue_ris)                     # C_ris_rx @ Phi @ C_tx_ris : (b, T, Nr, Nt)
        C_full    = C_casc + tv.H_ue_bs

        tv.set(cascaded_channel=C_casc)
        tv.set(full_channel=C_full)
        return tv


    def _apply_SIM_cascaded_channel(self, tv: TransmissionVariables) -> TransmissionVariables:
        if self.MS_variation != 'SIM': raise RuntimeError('This function is only for a MINN using a SIM.')

        T              = tv.transmit_signal.shape[1]
        all_ris_confs  = self.MS_controller(tv)                                  # list of (b, N)
        all_phis       = [torch.exp(-1j*ris_conf) for ris_conf in all_ris_confs]
        all_phis       = [phi.unsqueeze(1).repeat(1, T, 1) for phi in all_phis]  # list of (b, T, N)
        self.SIM_propagation_model.set_all_phis(all_phis)
        C_casc         = self.SIM_propagation_model(tv.H_ris_bs)
        #C_casc         = matmul_with_diagonal_right(tv.H_ris_bs, phi)           # C_ris_rx @ Phi : (b, T, Nr, N)
        C_casc         = torch.matmul(C_casc, tv.H_ue_ris)                       # C_ris_rx @ Phi @ C_tx_ris : (b, T, Nr, Nt)
        C_full         = C_casc + tv.H_ue_bs

        tv.set(cascaded_channel=C_casc)
        tv.set(full_channel=C_full)
        return tv


    def _construct_received_signal(self, tv: TransmissionVariables) -> TransmissionVariables:
        y_receive = torch.matmul(tv.full_channel, tv.transmit_signal)  # (b, T, Nr, 1)
        y_receive = y_receive + sample_awgn_torch(0.0, self.noise_std, y_receive.size(), y_receive.device)
        y_receive = y_receive.squeeze(dim=3)
        y_receive = torch.cat((y_receive.real, y_receive.imag), dim=2)  # (b, T, 2*Nr)

        tv.set(received_signal=y_receive)
        return tv


    def forward(self, tv: TransmissionVariables):

        tv = self._apply_tx_coding_and_power_norm(tv)  # sets `transmitted_signal`
        tv = self._fix_channel_dimensions(tv)          # sets `H_*_*` (changes shape)

        if self.MS_variation == 'RIS':
            tv = self._apply_RIS_cascaded_channel(tv)  # sets `cascaded_channel` and `full_channel`
        elif self.MS_variation == 'SIM':
            tv = self._apply_SIM_cascaded_channel(tv)  # sets `cascaded_channel` and `full_channel`
        else:
            C_full = tv.H_ue_bs
            tv.set(full_channel=C_full)

        tv = self._construct_received_signal(tv)       # sets `received_signal`
        z  = self.decoder(tv)

        return z





def construct_minn(params: Parameters) -> Minn:

    from transceiver_modules import ChannelAwareEncoder, ChannelAgnosticEncoder, AdvancedChannelAgnosticEncoder, \
        ChannelAwareDecoder, ChannelAgnosticDecoder
    from metasurface_modules import TrainableFixedRis, RisController, TrainableFixedSim, SimController

    assert params.MINN.csi_knowledge in {'aware', 'agnostic'}
    assert params.MINN.metasurface_type in {'RIS', 'SIM', None}
    assert params.MINN.metasurface_control in {'reconfigurable', 'static'}
    assert not isinstance(params.Auxiliary.data_shape, str) and params.Auxiliary.data_shape is not None, "Load dataset (via `training.load_data()`) before instantiating a SIM object"

    C = params.Channels

    if params.MINN.csi_knowledge == 'aware':
        decoder = ChannelAwareDecoder(C.Nt, C.Nr, C.N, C.TpF)
        if params.Training.dataset == 'CIFAR10':
            raise NotImplementedError("Channel-aware TX encoder for CIFAR-10 has not been implemented.")
        else:
            encoder = ChannelAwareEncoder(C.Nt, C.Nr, C.N, C.TpF, params.Auxiliary.data_shape)

    elif params.MINN.csi_knowledge == 'agnostic':
        decoder = ChannelAgnosticDecoder(C.Nt, C.Nr, C.N, C.TpF)

        if params.Training.dataset == 'CIFAR10':
            encoder = AdvancedChannelAgnosticEncoder(C.Nt, C.Nr, C.N, C.TpF, params.Auxiliary.data_shape)
        else:
            encoder = ChannelAgnosticEncoder(C.Nt, C.Nr, C.N, C.TpF, params.Auxiliary.data_shape)

    if params.MINN.metasurface_type is None:
        MS_module = None

    else:
        n_ris_rows, n_ris_cols = split_to_close_to_square_factors(C.N)
        sim_layer_dimensions   = [(n_ris_rows, n_ris_cols)] * C.n_sim_layers
        if params.MINN.metasurface_type == 'RIS' and params.MINN.metasurface_control == 'static':
            MS_module = TrainableFixedRis(n_ris_rows, n_ris_cols)
        elif params.MINN.metasurface_type == 'RIS' and params.MINN.metasurface_control == 'reconfigurable':
            MS_module = RisController(n_ris_rows, n_ris_cols, C.Nt, C.Nr, C.TpF)
        elif params.MINN.metasurface_type == 'SIM' and params.MINN.metasurface_control == 'static':
            MS_module = TrainableFixedSim(sim_layer_dimensions)
        elif params.MINN.metasurface_type == 'SIM' and params.MINN.metasurface_control == 'reconfigurable':
            MS_module = SimController(sim_layer_dimensions, C.Nt, C.Nr, C.TpF)


    minn = Minn(params, encoder, MS_module, decoder)
    return minn
