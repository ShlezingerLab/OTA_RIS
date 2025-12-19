from collections import namedtuple
from typing import Tuple, List, Union, Iterator
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

def repeat_num_to_list_if_not_list_already(num_or_list, expected_len) -> List:

    if isinstance(num_or_list, list) or isinstance(num_or_list, tuple) or isinstance(num_or_list, np.ndarray):
        if len(num_or_list) != expected_len:
            raise ValueError
        else:
            mylist = num_or_list
    else:
        mylist = [num_or_list]*expected_len

    return mylist


def matmul_by_diag_as_vector(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Given a 2D (or batched) matrix M and a vector (or batched matrix) v, perform the operation X = M @ diag(v),
    where diag(v) gives a square matrix with elements of v along its main diagonal,
    WITHOUT constructing the (sparse) diagonal matrix for memory and time efficiency.
    This exploits the fact that each i-th row of X, X_i can be seen as an elementwise product
    of M_i (i-th row of M) with v, which is more efficient.

    This function accepts batched versions of the arguments and performs the operations as such:
    - If M is of shape (m,n) and v is an n-dimensional vector, then the output is of shape (m,n) - i.e. normal matmul.
    - If M is of shape (b1,b2,...,bk,m,n) and v is of shape (b1,b2,...,bk,n), then the output is of
      shape (b1,b2,...,bk,m,n) - i.e. batched matrix multiplications are applied across the first k dimensions.
    """
    return M * v[..., None, :]


def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Get an array of point-to-point Euclidean distances among all elements (seen as 3D coordinates) of the two arrays.
    :param A: First array of points of shape (n, 3)
    :param B: Second array of points of shape (m, 3)
    :return: Array of point-to-point distances of shape (n, m)
    """
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
    """
    Similar to `pairwise_distances(A, B)` but returns vectors of the form `b-a`
    for each `b` point (row) in B and each `a` point in B.
    :param A:
    :param B:
    :return:
    """
    if A.ndim != 2 or B.ndim != 2: raise ValueError
    if A.shape[1] != 3 or B.shape[1] != 3: raise ValueError

    def vector_subtraction(a,b):
        return b - a

    A = A[:, None, :]  # Shape: (n, 1, 3)
    B = B[None, :, :]  # Shape: (1, m, 3)

    endpoints = vector_subtraction(A, B)  # Shape: (n, m, 3)
    return endpoints


def mag(v: np.ndarray, axis=-1):
    """
    Magnitude of a vector (of a batched matrix of vectors)
    :param v:
    :param axis:
    :return:
    """
    return np.sqrt(np.sum(v**2, axis=axis))


# vectors to be used to perform 3D rotations in order to align the placement of SIM elements with each of the axes
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
    """
    Align (x,y,z) coordinates (or arrays of coordinates) to the desired 2D plane by simply transposing the appropriate
    x, y, or z axes.
    :param X:
    :param Y:
    :param Z:
    :param plane: one of 'xy', 'yx', 'yz', 'zy', 'zx', 'xz'
    :return: Three floats corresponding to the new x,y,z coordinates or three arrays of x,y,z coordinates.
    """

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

class Metasurface(nn.Module):
    """
    This class models the "physical" metasurface (i.e. RIS) which can be used as single layer of the SIM neuralnet.
    It includes geometrical information of its elements (area, and xyz coordinates)
    as well as the internal state (θ) of the elements. This is a trainable component (torch.nn.Parameter).
    The phase shifts (φ) can be obtained by the self.phi @property.
    """
    def __init__(self,
                 num_elems: int,
                 elems_per_row: int,
                 elem_area: float,
                 elem_dist: float,
                 central_point_coords: Tuple[float, float, float],
                 elem_placement_plane='yz',
                 complex_dtype=torch.complex64,
                 ):
        """
        Construct a metasurface using geometrical information about its elements
        :param num_elems: Total number of elements of the metasurface.
        :param elems_per_row: Number of elements per row. Must divide evenly `num_elems`.
        :param elem_area: The area (in meters^2) each element occupies.
        :param elem_dist: How far apart (along any axis) are the centers of consecutive elements.
                          Physically consistent value must be <= sqrt(elem_area) for non overlapping square elements
        :param central_point_coords: (x,y,z) coordinates of its center.
                                     The positions of the elements are determined accordignly.
        :param elem_placement_plane: One of 'xy', 'yx', 'yz', 'zy', 'zx', 'xz', to determine to which plane
                                     the elements will be aligned, which affects their 3d coordinates
        :param complex_dtype: either torch.complex64 or torch.complex128 (note that not
                              all torch functions may be defined for both types).
        """
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
        """
        Obtain the phase shift response (φ) of the metsurface.
        :return:
        """
        phi_ = torch.exp(1j * self.to_0_2pi(self.theta))
        phi_ = phi_.to(self.complex_dtype)
        return phi_


    def forward(self):
        """
        Make this layer class callable as a torch.nn.Module that returns its phase shifts.
        :return:
        """
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

        X, Y, Z  = self.elem_positions[:, 0], self.elem_positions[:, 1], self.elem_positions[:, 2]

        ax.scatter(X, Y, Z, marker='s', s=20, c=color)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if show:
            plt.show(block=True)

        matplotlib.use(original_backend)
        return ax



class Surface2SurfaceTransmission(nn.Module):
    """
    This class models the radiation propagation between two arbitrary metsurfaces that are assumed to be closely packed
    so that the Rayleigh-Sommerfeld diffraction equation holds.
    The modeling follows:
    - https://arxiv.org/pdf/2305.08079 (Eq. 9) and/or equivalently:
    - https://arxiv.org/pdf/2408.04837 (Eq. 2)
    This is a *static* torch.nn.Module, since once the transmission matrix W is determined during initialization,
    it should remain fixed in all forward/backward calls.
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




# Just a convenient data type that holds information about the number of elements along the x and y axes (n_x and n_y).
# It can be used in the following SimNet class as a list of layer parameters that all have the same element spacing,
# element distances, and aligned centers, to easily create SimNet instances.
# the n_x and n_y could correspond to any of the x,y,z axes.
RisLayer = namedtuple('RisLayer', ['n_x', 'n_y'])



class SimNet(nn.Module):
    """
    Implementation of a SIM as a torch.nn.Module.
    It contains trainable phase shifts (using the Metasurface module)
    and fixed propagation equations (using the Surface2SurfaceTransmission module).

    This class supports any number of metasurface layers of any number of units,
    but they are all parallel to the specified plane with their centers being aligned.
    The element distances and areas are the same for all layers.

    Additionally, this module may have arbitrary preprocessing and postprocessing torch.nn.Modules.
    Those can be used for example to model data transmission (input_module)
    and a head module to perform classification or regression.

    The modeling follows:
    - https://arxiv.org/pdf/2305.08079 (Eq. 10) and/or equivalently:
    - https://arxiv.org/pdf/2408.04837 (Eq. 3)
    which is generally of the form of
    Y = Φ_L @ W_L @ Φ_{L-1} @ W_{L-1} @ ... @ Φ_2 @ W_2 @ Φ_1 @ Χ
    where X is the input matrix, Φ_i are the trainable phase shifts and W_i are the fixed layer-to-layer
    propagation coefficients.
    If input and output modules are present, say f() and g(), respectively, the forward call is implemented instead as
    Y = g(Φ_L @ W_L @ Φ_{L-1} @ W_{L-1} @ ... @ Φ_2 @ W_2 @ Φ_1 @ f(Χ))
    """
    def __init__(self,
                 layers: List[RisLayer],
                 layer_dist: Union[float, List[float]],
                 wavelength: float,
                 elem_area: float,
                 elem_dist: float,
                 layers_orientation_plane: str = 'yz',
                 first_layer_central_coords: Union[tuple, list, np.ndarray] = (0,0,0),
                 input_module: nn.Module = None,
                 output_module: nn.Module = None,
                 complex_dtype=torch.complex64,
                 ):
        """

        :param layers: A list of RisLayers (essentially (n_x, n_y) tuples specifying the number
                       of elements per row and column for each layer. The length of the list corresponds to the number
                       of SIM layers.
        :param layer_dist: The distance (in meters) between two consecutive SIM layers.
        :param wavelength: The wavelength of the carrier frequency (used to calculate the layer-to-layer propagation).
        :param elem_area: To be passed to Metasurface.elem_area (common for all layers).
        :param elem_dist: To be passed to Metasurface.elem_dist (common for all layers).
        :param layers_orientation_plane: One of 'xy', 'yx', 'yz', 'zy', 'zx', 'xz', to determine to which plane all
                                         layers will be aligned.
        :param first_layer_central_coords: (x,y,z) coordinates of the first layer. The centers of the rest of layers
                                           will be determined automatically by increasing one of the x,y,z coordinates
                                           (as implied by the `layers_orientation_plane` value) by `layer_dist` each
                                           time, while keeping the other two coordinates the same.
        :param input_module: An optional torch.nn.Module that operates on the input data `x` during `self.forward(x)`
                             and passes its output to be multiplied with the phase shifts of the first SIM layer.
                             If present, it must be compatible with `x`'s shape and must output a tensor (either complex
                             or real of shape (b, n_x1, n_y1) where b is the batch size, n_x1 and n_y1 are the number of
                             elements across the rows and columns of the first SIM layer.
                            It can be used to model wireless channel transmission for example.
        :param output_module: An optional torch.nn.Module that operates on the result after the phase shifts of the last
                              SIM layer have been applied.
                              If present, it must accept a complex tensor of shape (b, n_xL, n_yL) where b is the batch
                              size, n_x1 and n_y1 are the number of elements across the rows and columns of the last
                              SIM layer. Its output shape can be arbitrary.
                              It can be used for example to convert the response at the last layer to a multi-class
                              classification problem by measuring the amplitudes of the responses of each of the
                              elements of the last SIM layer and/or apply softmax or other activation.
        :param complex_dtype: either torch.complex64 or torch.complex128 (note that not
                              all torch functions may be defined for both types).

        """
        super(SimNet, self).__init__()

        self.input_module          = input_module
        self.output_module         = output_module
        self.wavelength            = wavelength
        self.complex_dtype         = complex_dtype
        self.layer_distances       = repeat_num_to_list_if_not_list_already(layer_dist, len(layers)-1)
        self.num_layers            = len(layers)
        # IMPORTANT:
        # `ris_layers` and `transmission_layers` must be registered as submodules so that
        # `model.to(device)` / `model.cuda()` correctly moves all parameters & buffers.
        # Plain Python lists will NOT be traversed by PyTorch, which leads to CPU/CUDA
        # device mismatches (e.g., `phi_antenna` on CPU while `x` is on CUDA).
        ris_layers, transmission_layers = self._construct_layers(
            layers,
            self.layer_distances,
            wavelength,
            elem_area,
            elem_dist,
            layers_orientation_plane,
            first_layer_central_coords,
            complex_dtype,
        )
        self.ris_layers = nn.ModuleList(ris_layers)
        self.transmission_layers = nn.ModuleList(transmission_layers)


    def _construct_layers(self,
                          layers: List[RisLayer],
                          layer_distances,
                          lam,
                          elem_area,
                          elem_dist,
                          orientation,
                          first_layer_central_coords,
                          dtype
                          ) -> Tuple[List[Metasurface], List[Surface2SurfaceTransmission]]:

        def calculate_max_surface_dimensions(layers_: List[RisLayer], elem_dist) -> Tuple[float,float]:
            max_rows = max([layer.n_x for layer in layers_])
            max_cols = max([layer.n_y for layer in layers_])
            return max_rows*elem_dist, max_cols*elem_dist


        max_surface_x, max_surface_y = calculate_max_surface_dimensions(layers, elem_dist)
        z_offset                     = 0
        ris_center_x                 = max_surface_x / 2
        ris_center_y                 = max_surface_y / 2
        ris_layers                   = []
        transmission_layers          = []
        prev_ris                     = None

        for i, ris_params in enumerate(layers):
            num_elems           = ris_params.n_x * ris_params.n_y
            elems_per_row       = ris_params.n_x
            ris_z               = z_offset
            ris_x, ris_y, ris_z = align_coords_to_plane(ris_center_x, ris_center_y, ris_z, orientation)
            ris_x              -= first_layer_central_coords[0]
            ris_y              -= first_layer_central_coords[1]
            ris_z              -= first_layer_central_coords[2]
            ris                 = Metasurface(num_elems,
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


    def forward(self, x):

        if self.input_module:
            x = self.input_module(x)

        x           = x.to(self.complex_dtype)
        phi_antenna = self.ris_layers[0]()
        #x           = x * phi_antenna.view(x.shape)
        x           = x.view(-1, len(phi_antenna))
        x           = phi_antenna * x

        for i in range(1, self.num_layers):

            W   = self.transmission_layers[i-1]()
            phi = self.ris_layers[i]()
            x   = torch.matmul(x, W)
            x   = matmul_by_diag_as_vector(x, phi)


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




class ClassificationHead(nn.Module):
    def __init__(self, in_complex_features, out_values=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_complex_features, out_values)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.num_out_values = out_values

    def forward(self, x):
        x = torch.sqrt(torch.real(x)**2 + torch.imag(x)**2)
        out = self.fc1(x)

        if self.num_out_values == 1:
            out = self.sigmoid(out)
        else:
            out = self.softmax(out)

        return out


class RegressionHead(nn.Module):
    def __init__(self, in_complex_features, out_values=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(2*in_complex_features, out_values)


    def forward(self, x):
        x   = torch.concatenate([torch.real(x), torch.imag(x)], dim=1)
        out = self.fc1(x)
        return out



if __name__ == '__main__':

    def test_layers():
        import matplotlib
        import matplotlib.pyplot as plt


        l1 = Metasurface(8*8, 8, 0.001, 0.0005,   (0, 0, 0), 'yz')
        l2 = Metasurface(8*8, 8, 0.001, 0.0005, (0.001, 0., 0.0), 'yz')
        l3 = Metasurface(20*20, 25, 0.001, 0.0005,  (0.002, 0, 0.0), 'yz')

        t1 = Surface2SurfaceTransmission(l1, l2, lam=0.001)
        W = t1()
        print(W.shape)
        print(torch.abs(W).max(), torch.abs(W).min())
        t1.visualize_transmission_matrix()

        original_backend = matplotlib.get_backend()
        matplotlib.use('Qt5Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        l1.visualize_surface(ax=ax, show=False)
        l2.visualize_surface(ax=ax, show=False)
        l3.visualize_surface(ax=ax, show=True)


    def test_network():

        module = SimNet(
            [RisLayer(16,16),
             RisLayer(32,32),
             RisLayer(1,1),],
            1,
            2,
            1,
            1,
            'yz',
            (10,20,30),
            None,
            None,
            torch.complex64
        )

        out = module(torch.randn(10,16,16))
        print(out.shape)
        module.visualize()

    test_layers()
    test_network()
