from collections import namedtuple
from typing import Tuple, List, Union, Iterator
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

from minn import TransmissionVariables
from utils_misc import repeat_num_to_list_if_not_list_already
from utils_torch import matmul_by_diag_as_vector


def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.ndim != 2 or B.ndim != 2: raise ValueError
    if A.shape[1] != 3 or B.shape[1] != 3: raise ValueError

    def euclidean_distance(a, b):
        dist_sq = np.sum((a-b)**2, axis=2)
        return np.sqrt(dist_sq)

    A = A[:, None, :]  # Shape: (n, 1, 3)
    B = B[None, :, :]  # Shape: (1, m, 3)

    dists = euclidean_distance(A,B)  # Shape: (n,m)
    return dists


def pairwise_vectors(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.ndim != 2 or B.ndim != 2: raise ValueError
    if A.shape[1] != 3 or B.shape[1] != 3: raise ValueError

    def vector_subtraction(a,b):
        return b - a

    A = A[:, None, :]  # Shape: (n, 1, 3)
    B = B[None, :, :]  # Shape: (1, m, 3)

    endpoints = vector_subtraction(A, B)  # Shape: (n, m, 3)
    return endpoints


def mag(v: np.ndarray, axis=-1):
    return np.sqrt(np.sum(v**2, axis=axis))


normal_direction_vectors_along_plane = {
    'xy': np.array([0,0,1]),
    'yx': np.array([0,0,1]),
    'yz': np.array([1,0,0]),
    'zy': np.array([1,0,0]),
    'zx': np.array([0,1,0]),
    'xz': np.array([0,1,0]),
}

def align_coords_to_plane(X: Union[float, np.ndarray],
                          Y: Union[float, np.ndarray],
                          Z: Union[float, np.ndarray],
                          plane: str
                          ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:

    if isinstance(X, np.ndarray) or isinstance(Y, np.ndarray) or isinstance(Z, np.ndarray):
        if not all([len(item) == len(X) for item in (X,Y,Z)]):
            raise ValueError("If any of the items is an array, all items must be arrays of the same length.")


    convert_to = {
        'xy': lambda x,y,z: (x,y,z),
        'yx': lambda x,y,z: (x,y,z),
        'yz': lambda x,y,z: (z,x,y),
        'zy': lambda x,y,z: (z,x,y),
        'zx': lambda x,y,z: (y,z,x),
        'xz': lambda x,y,z: (y,z,x),
    }
    return convert_to[plane](X,Y,Z)



def frob_norm(X: np.ndarray):
    if not X.ndim == 2: raise ValueError

    return (np.abs(X)**2).sum()


class Metasurface(nn.Module):
    """
    Models a single SIM layers where the phase shift parameters theta (ω in the paper) are trainable.
    The responses φ = exp(-jπω) are computed dynamically.
    This is to be used in the SimNet class when `metasurface_control` is set to 'static' in `Parameters.MINN`
    """
    def __init__(self,
                 num_elems,
                 elems_per_row,
                 elem_area,
                 elem_dist,
                 central_point_coords: Tuple,
                 elem_placement_plane='yz',
                 complex_dtype=torch.complex64,
                 ):
        super(Metasurface, self).__init__()

        self.num_elems            = num_elems
        self.num_cols             = elems_per_row
        self.elem_area            = float(elem_area)
        self.elem_dist            = float(elem_dist)
        self.complex_dtype        = complex_dtype
        self.num_rows             = self.num_elems // self.num_cols
        self.elem_placement_plane = elem_placement_plane
        self.central_point_coords = np.array(central_point_coords, dtype=float)
        self.elem_positions       = self._get_element_positions()

        self.theta                = nn.Parameter(0.5*torch.ones(self.num_elems))
        # self.phi                = ... # <---- phi is a @property method to perform the calculations during every call
                                        #       This is slower, but it allows autograd to work.

        assert self.num_elems % self.num_cols == 0


    @staticmethod
    def to_0_2pi(x):
        """
        for a given number/array x, transform it into the interval [0, 2π]
        by first passing x through sigmoid(x) -> [0, 1] and then multiply it by 2π
        """
        return nn.functional.sigmoid(x) * 2*torch.pi


    @property
    def phi(self):
        phi_ = torch.exp(1j * self.to_0_2pi(self.theta))
        phi_ = phi_.to(self.complex_dtype)
        return phi_


    def forward(self):
        return self.phi


    def _get_element_positions(self):
        """
        Get a (n,3) matrix of coordinates of the (centers of the) elements of the RIS layer,
        where n is the number of elements.
        This function supports placing the elements at any plane and at any origin point.
        """
        X, Y = np.meshgrid(np.arange(self.num_cols), np.arange(self.num_rows))
        X    = self.elem_dist * X
        Y    = self.elem_dist * Y
        Z    = np.zeros_like(X)

        surface_width    = self.elem_dist * self.num_cols
        surface_height   = self.elem_dist * self.num_rows
        central_point    = self.central_point_coords


        if self.elem_placement_plane in {'xy', 'yx'}:
            coords           = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
            central_point[0] = central_point[0] + surface_width / 2
            central_point[1] = central_point[1] + surface_height / 2

        elif self.elem_placement_plane in {'yz', 'zy'}:
            coords           = np.column_stack((Z.flatten(), X.flatten(), Y.flatten()))
            central_point[1] = central_point[1] + surface_width / 2
            central_point[2] = central_point[2] + surface_height / 2

        elif self.elem_placement_plane in {'zx', 'xz'}:
            coords           = np.column_stack((Y.flatten(), Z.flatten(), X.flatten()))
            central_point[0] = central_point[0] + surface_width / 2
            central_point[2] = central_point[2] + surface_height / 2

        else:
            raise ValueError("Invalid plane parameter. Use 'xy', 'yz', or 'zx'.")

        coords -= central_point
        #coords[:,0], coords[:,1], coords[:,2] = align_coords_to_plane(coords[:,0], coords[:,1], coords[:,2], self.elem_placement_plane)

        return coords


    def visualize_surface(self, color='k', ax=None, show=True):
        import matplotlib
        import matplotlib.pyplot as plt

        original_backend = matplotlib.get_backend()
        matplotlib.use('Qt5Agg')

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        X, Y, Z       = self.elem_positions[:, 0], self.elem_positions[:, 1], self.elem_positions[:, 2]

        ax.scatter(X, Y, Z, marker='s', s=20, c=color)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if show:
            plt.show(block=True)

        matplotlib.use(original_backend)
        return ax




class MetasurfaceWithoutPhaseShifts(Metasurface):
    """
    Models a single SIM layers where the phase shift parameters theta (ω in the paper) are **not trainable**.
    Their values are supposed to be set dynamically by a controller module.
    The responses φ = exp(-jπω) are again computed dynamically.
    This is to be used in the ReconfigurableSimNet class when `metasurface_control` is set to 'reconfigurable' in `Parameters.MINN`
    """
    def __init__(self,
                 num_elems,
                 elems_per_row,
                 elem_area,
                 elem_dist,
                 central_point_coords: Tuple,
                 elem_placement_plane='yz',
                 complex_dtype=torch.complex64,
                 ):
        super().__init__(num_elems, elems_per_row, elem_area, elem_dist,
                         central_point_coords, elem_placement_plane, complex_dtype)
        self.theta = None

    def phi(self):
        raise NotImplementedError("This is a dummy implementation for SIM integration. Specify phi externally.")



class Surface2SurfaceTransmission(nn.Module):
    """
    Models the Ξ matrices from the paper using the Rayleigh-Sommerfeld equation.
    Implemented as a torch module to allow backpropagation.
    """
    def __init__(self,
                 prev_ris_layer: Metasurface,
                 next_ris_layer: Metasurface,
                 lam: float,
                 ):

        super().__init__()

        self.prev_ris_layer = prev_ris_layer
        self.next_ris_layer = next_ris_layer
        self.lam            = lam

        if self.prev_ris_layer.elem_placement_plane in {'xy', 'yx'}:
            self.layer_dist = abs(self.prev_ris_layer.elem_positions[0][2] - self.next_ris_layer.elem_positions[0][2])
        elif self.prev_ris_layer.elem_placement_plane in {'yz', 'zy'}:
            self.layer_dist = abs(self.prev_ris_layer.elem_positions[0][0] - self.next_ris_layer.elem_positions[0][0])
        elif self.prev_ris_layer.elem_placement_plane in {'zx', 'xz'}:
            self.layer_dist = abs(self.prev_ris_layer.elem_positions[0][1] - self.next_ris_layer.elem_positions[0][1])

        W_values            = torch.from_numpy(self._construct_propagation_coefficient_matrix())
        W_values            = W_values.to(self.prev_ris_layer.complex_dtype)
        self.register_buffer('W',W_values,persistent=True)

        assert self.prev_ris_layer.complex_dtype == self.next_ris_layer.complex_dtype
        assert self.prev_ris_layer.elem_placement_plane == self.next_ris_layer.elem_placement_plane




    def forward(self):
        return self.W


    def _construct_propagation_coefficient_matrix(self) -> np.ndarray:
        assert self.prev_ris_layer.elem_area == self.next_ris_layer.elem_area

        atom2atom_dists            = pairwise_distances(self.prev_ris_layer.elem_positions,
                                                        self.next_ris_layer.elem_positions)
        atom2atom_vectors          = pairwise_vectors(self.prev_ris_layer.elem_positions,
                                                      self.next_ris_layer.elem_positions)
        propagation_normal_vector  = normal_direction_vectors_along_plane[self.prev_ris_layer.elem_placement_plane]
        propagation_normal_vector  = self.layer_dist * propagation_normal_vector
        propagation_angle_cosines  = mag(propagation_normal_vector) / mag(atom2atom_vectors, axis=2) # <---------

        w_factor_1                 = self.prev_ris_layer.elem_area * propagation_angle_cosines / atom2atom_dists
        w_factor_2                 = 1/(2*np.pi*atom2atom_dists) - 1j/self.lam
        w_factor_3                 = np.exp(1j*2*np.pi * atom2atom_dists / self.lam)
        W                          = w_factor_1 * w_factor_2 * w_factor_3

        W_mag = frob_norm(W)
        #W = W/W_mag
        riemann_rho = ((self.lam/4)**2)

        assert W.ndim == 2
        assert W.shape[0] == self.prev_ris_layer.num_elems and W.shape[1] == self.next_ris_layer.num_elems

        return W


    def visualize_transmission_matrix(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()

        ax1 = fig.add_subplot(221)
        im = ax1.imshow(self.W.real)
        ax1.set_title('Real{W}')

        ax2 = fig.add_subplot(222)
        im = ax2.imshow(self.W.imag)
        ax2.set_title('Imag{W}')

        ax1 = fig.add_subplot(223)
        im = ax1.imshow(np.abs(self.W))
        ax1.set_title('|W|')

        ax2 = fig.add_subplot(224)
        im = ax2.imshow(np.angle(self.W))
        ax2.set_title('angle(W)')

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(im, cax=cbar_ax)

        # plt.colorbar()
        plt.show()





RisLayer = namedtuple('RisLayer', ['n_x', 'n_y'])



class SimNet(nn.Module):
    """
    Implements the SIM model Π (Ω_m @ Ξ_m) Ω_1 as a torch.nn.Module to allow for automatic differentiation.
    This class uses metasurfaces whose phase shifts are treated as trainable parameters.
    """
    def __init__(self,
                 layers: List[RisLayer],
                 layer_dist: Union[float, List[float]],
                 wavelength: float,
                 elem_area: float,
                 elem_dist: float,
                 inp_shape: Union[Tuple, List, torch.Size],
                 layers_orientation_plane: str = 'yz',
                 first_layer_central_coords: Union[tuple, list, np.ndarray] = (0,0,0),
                 input_module: nn.Module = None,
                 output_module: nn.Module = None,
                 complex_dtype=torch.complex64,
                 ):
        super(SimNet, self).__init__()

        self.input_module          = input_module
        self.output_module         = output_module
        self.wavelength            = wavelength
        self.complex_dtype         = complex_dtype
        self.layer_distances       = repeat_num_to_list_if_not_list_already(layer_dist, len(layers)-1)
        self.num_layers            = len(layers)
        self.inp_shape             = inp_shape
        (self.ris_layers,
         self.transmission_layers) = self._construct_layers(layers, self.layer_distances, wavelength, elem_area,
                                                            elem_dist, layers_orientation_plane,
                                                            first_layer_central_coords, complex_dtype)
        #self.layer_norm1           = ComplexLayerNorm(self.inp_shape, elementwise_affine=False)
        self.layer_norm1           = lambda x: x

    def _construct_layers(self,
                          layers: List[RisLayer],
                          layer_distances,
                          lam,
                          elem_area,
                          elem_dist,
                          orientation,
                          first_layer_central_coords,
                          dtype,
                          metasurface_constructor=Metasurface
                          ) -> Tuple[List[Metasurface], List[Surface2SurfaceTransmission]]:

        def calculate_max_surface_dimensions(layers_: List[RisLayer], elem_dist) -> Tuple[float,float]:
            max_rows = max([layer.n_x for layer in layers_])
            max_cols = max([layer.n_y for layer in layers_])
            return max_rows*elem_dist, max_cols*elem_dist


        max_surface_x, max_surface_y = calculate_max_surface_dimensions(layers, elem_dist)
        z_offset                     = 0
        ris_center_x                 = max_surface_x / 2
        ris_center_y                 = max_surface_y / 2
        ris_layers                   = nn.ModuleList()
        transmission_layers          = nn.ModuleList()
        prev_ris                     = None

        for i, ris_params in enumerate(layers):
            num_elems           = ris_params.n_x * ris_params.n_y
            elems_per_row       = ris_params.n_x
            ris_z               = z_offset
            ris_x, ris_y, ris_z = align_coords_to_plane(ris_center_x, ris_center_y, ris_z, orientation)
            ris                 = metasurface_constructor(num_elems,
                                                          elems_per_row,
                                                          elem_area,
                                                          elem_dist,
                                                          (ris_x, ris_y, ris_z),
                                                          orientation,
                                                          dtype)
            ris_layers.append(ris)

            if prev_ris:
                transmission = Surface2SurfaceTransmission(prev_ris, ris, lam)
                transmission_layers.append(transmission)

            prev_ris = ris
            if i < self.num_layers-1:
                z_offset += layer_distances[i]

        return ris_layers, transmission_layers


    def _get_RIS_configuration(self, layer_idx) -> torch.Tensor:
        return self.ris_layers[layer_idx]()

    def activation(self, x):
        return x   # no activation function in each SIM -> The whole SIMNet is a linear layer

    def forward(self, x):

        #x = self.layer_norm1(x)

        if self.input_module:
            x = self.input_module(x)

        x           = x.to(self.complex_dtype)
        phi_antenna = self._get_RIS_configuration(0)
        #x           = x * phi_antenna.view(x.shape)
        #x           = x.view(-1, len(phi_antenna))
        #x           = phi_antenna * x
        x = matmul_by_diag_as_vector(x, phi_antenna.to(x.device))
        x = self.activation(x)

        curr_device = x.device
        for l in self.transmission_layers:
            l.to(curr_device)

        for i in range(1, self.num_layers):

            W   = self.transmission_layers[i-1]()
            phi = self._get_RIS_configuration(i).to(x.device)
            x   = torch.matmul(x, W)
            x   = matmul_by_diag_as_vector(x, phi)
            x   = self.activation(x) # this does nothing but it's left for extensions


        if self.output_module:
            x = self.output_module(x)

        return x


    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        params = {}

        if self.input_module:
            for name, W in self.input_module.named_parameters():
                params['SimNet_input_'+name] = W

        for i, layer in enumerate(self.ris_layers):
            for name, W in layer.named_parameters():
                params[f"SimNet_{name}_{i+1}"] = W

        if self.output_module:
            for name, W in self.output_module.named_parameters():
                params['SimNet_output_'+name] = W

        return params.values()


    def visualize(self, backend='Qt5Agg', ax=None, show=True):
        matplotlib.use(backend)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')


        for l in self.ris_layers[:-1]:
            l.visualize_surface(ax=ax, show=False)

        self.ris_layers[-1].visualize_surface(ax=ax, show=show)





class ReconfigurableSimNet(SimNet):
    """
    Implements the SIM model Π (Ω_m @ Ξ_m) Ω_1 as a torch.nn.Module to allow for automatic differentiation.
    This class uses metasurfaces whose phase shifts are treated as .
    """
    def __init__(self,
                 layers: List[RisLayer],
                 layer_dist: Union[float, List[float]],
                 wavelength: float,
                 elem_area: float,
                 elem_dist: float,
                 inp_shape: Union[Tuple, List, torch.Size],
                 layers_orientation_plane: str = 'yz',
                 first_layer_central_coords: Union[tuple, list, np.ndarray] = (0, 0, 0),
                 input_module: nn.Module = None,
                 output_module: nn.Module = None,
                 complex_dtype=torch.complex64,
                 ):
        super().__init__(layers, layer_dist, wavelength, elem_area, elem_dist, inp_shape, layers_orientation_plane,
                         first_layer_central_coords, input_module, output_module, complex_dtype)

        (self.ris_layers,
         self.transmission_layers) = self._construct_layers(layers, self.layer_distances, wavelength, elem_area,
                                                            elem_dist, layers_orientation_plane,
                                                            first_layer_central_coords, complex_dtype,
                                                            metasurface_constructor=MetasurfaceWithoutPhaseShifts)
        self.all_phis = None

    def set_all_phis(self, all_phis):
        self.all_phis = all_phis

    def _get_RIS_configuration(self, layer_idx) -> torch.Tensor:
        return self.all_phis[layer_idx]




class RisController(nn.Module):
    """
    Implements the RIS controller module: Observes instantaneous CSI and decides on reconfigurable RIS profiles
    """
    def __init__(self,
                 n_ris_rows,
                 n_ris_cols,
                 n_tx,
                 n_rx,
                 TpF,
                 ):
        super(RisController, self).__init__()
        self.n_ris_rows = n_ris_rows
        self.n_ris_cols = n_ris_cols
        self.n_tx       = n_tx
        self.n_rx       = n_rx
        self.TpF        = TpF
        self.N          = n_ris_rows * n_ris_cols

        self.input_width  = self.N * n_tx + self.N * n_rx + self.n_rx * self.n_tx
        self.input_height = 2

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d((1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            # nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.flat_dim = int(np.prod(self._get_feature_dimensions()))

        self.head = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 2*self.N),
            nn.Tanh(),
            nn.Linear(2*self.N, self.N),
            nn.Tanh()
        )


    def _get_feature_dimensions(self):
        inp       = torch.randn(1, 1, self.input_height, self.input_width)
        out       = self.feature_extractor(inp)
        out_shape = out.size()
        return out_shape[1:]


    def forward(self, transmission_variables: TransmissionVariables) -> torch.Tensor:
        C_tx_rx   = transmission_variables.H_ue_bs[:,0,:,:].view(-1, 1, self.n_tx * self.n_rx)
        C_tx_ris  = transmission_variables.H_ue_ris[:,0,:,:].view(-1, 1, self.N * self.n_tx)
        C_ris_rx  = transmission_variables.H_ris_bs[:,0,:,:].view(-1, 1, self.n_rx * self.N)

        C_tx_rx   = torch.cat((torch.real(C_tx_rx), torch.imag(C_tx_rx)), dim=1)
        C_tx_ris  = torch.cat((torch.real(C_tx_ris), torch.imag(C_tx_ris)), dim=1)
        C_ris_rx  = torch.cat((torch.real(C_ris_rx), torch.imag(C_ris_rx)), dim=1)

        C         = torch.cat((C_tx_rx, C_tx_ris, C_ris_rx), dim=2)
        C         = C.unsqueeze(1)
        features  = self.feature_extractor(C)
        features  = features.view(-1, self.flat_dim)
        theta     = self.head(features)
        theta     = (theta + 1) * torch.pi

        return theta

class SimController(RisController):
    def __init__(self,
                 sim_layers_dimensions: List[Tuple[int,int]],
                 n_tx,
                 n_rx,
                 TpF,
                 ):
        n_ris_rows = sim_layers_dimensions[0][0]
        n_ris_cols = sim_layers_dimensions[0][1]
        super(SimController, self).__init__(n_ris_rows, n_ris_cols, n_tx, n_rx, TpF)
        self.N           = [x*y for (x,y) in sim_layers_dimensions]
        self.input_width = self.N[0] * n_tx + self.N[0] * n_rx + self.n_rx * self.n_tx
        self.n_layers    = len(sim_layers_dimensions)

        self.head       = None

        self.heads = []
        for i in range(self.n_layers):
            head = nn.Sequential(
                nn.Linear(self.flat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.N[i]),
                nn.Tanh()
            )
            self.heads.append(head)



    def forward(self, transmission_variables: TransmissionVariables) -> List[torch.Tensor]:
        C_tx_rx  = transmission_variables.H_ue_bs[:,0,:,:].view(-1, 1, self.n_tx * self.n_rx)
        C_tx_ris = transmission_variables.H_ue_ris[:,0,:,:].view(-1, 1, self.N[0] * self.n_tx)
        C_ris_rx = transmission_variables.H_ris_bs[:,0,:,:].view(-1, 1, self.n_rx * self.N[0])

        C_tx_rx  = torch.cat((torch.real(C_tx_rx), torch.imag(C_tx_rx)), dim=1)
        C_tx_ris = torch.cat((torch.real(C_tx_ris), torch.imag(C_tx_ris)), dim=1)
        C_ris_rx = torch.cat((torch.real(C_ris_rx), torch.imag(C_ris_rx)), dim=1)

        C        = torch.cat((C_tx_rx, C_tx_ris, C_ris_rx), dim=2)
        C        = C.unsqueeze(1)

        features = self.feature_extractor(C)
        features = features.view(-1, self.flat_dim)

        thetas   = []


        curr_device = C_tx_rx.device
        for h in self.heads:
            h.to(curr_device)

        for i in range(self.n_layers):
            theta = self.heads[i](features)
            theta = (theta + 1) * torch.pi
            thetas.append(theta)

        return thetas


class TrainableFixedRis(nn.Module):
    def __init__(self, n_ris_rows, n_ris_cols):
        super(TrainableFixedRis, self).__init__()
        self.N          = n_ris_rows * n_ris_cols
        initial_theta   = torch.rand(size=(self.N,)) * -2 * torch.pi
        self._theta     = torch.nn.Parameter(initial_theta)

    def forward(self, transmission_variables: TransmissionVariables):
        b      = transmission_variables.inputs.shape[0]
        thetas = (torch.tan(self._theta) + 1 ) * torch.pi        # (N,)
        thetas = thetas.unsqueeze(0).repeat(b , 1)               # (b, N)
        return thetas

class TrainableFixedSim(nn.Module):
    def __init__(self, sim_layers_dimensions: List[Tuple[int,int]],):
        super(TrainableFixedSim, self).__init__()
        self.N        = [x*y for (x,y) in sim_layers_dimensions]
        self.n_layers = len(sim_layers_dimensions)
        self._thetas  = nn.ParameterList()

        for i in range(self.n_layers):
            initial_theta = torch.rand(size=(self.N[i],)) * -2 * torch.pi
            theta         = torch.nn.Parameter(initial_theta)
            self._thetas.append(theta)

    def forward(self, transmission_variables: TransmissionVariables) -> List[torch.Tensor]:
        b          = transmission_variables.inputs.shape[0]
        all_thetas = []

        for i in range(self.n_layers):
            thetas = (torch.tan(self._thetas[i]) + 1) * torch.pi  # (N,)
            thetas = thetas.unsqueeze(0).repeat(b, 1)             # (b, N)
            all_thetas.append(thetas)

        return all_thetas