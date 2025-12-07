import numpy as np
from typing import Union, Tuple
from scipy.constants import pi
from tqdm import tqdm

from utils_misc import dBW_to_Watt, is_iterable, sample_gaussian_standard_normal, split_to_close_to_square_factors
from parameters import Parameters


def calculate_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.ndim == 1: A = A.reshape((1, -1))
    if B.ndim == 1: B = B.reshape((1, -1))

    assert A.shape[1] == B.shape[1] == 3
    n = A.shape[0]
    m = B.shape[0]
    dists = np.empty((n, m))
    for i in range(n):
        a = A[i, :]
        dists[i, :] = np.linalg.norm(a - B)

    return dists


def ray_to_elevation_azimuth(starting_point, ending_point):
    def cart2sph(x, y, z):
        XsqPlusYsq = x ** 2 + y ** 2
        r = np.sqrt(XsqPlusYsq + z ** 2)  # r
        elev = np.arctan2(z, np.sqrt(XsqPlusYsq))  # theta
        az = np.arctan2(y, x)  # phi
        return r, elev, az

    v = ending_point - starting_point  # type: np.ndarray
    _, elev, az = cart2sph(v[0], v[1], v[2])
    return elev, az


def calculate_pathloss(dist, wavelength, pathloss_exponent=2., extra_attenuation_dB=None):
    pl_dB = - pathloss_exponent * 10 * np.log10(4 * pi * dist / wavelength)
    extra_attenuation_dB = extra_attenuation_dB if extra_attenuation_dB is not None else 0
    pl_dB -= extra_attenuation_dB
    pl_W = dBW_to_Watt(pl_dB)
    return pl_W


def URA_steering_vector(tx_position: np.ndarray,
                        rx_position: np.ndarray,
                        num_tx_antennas: Union[int, Tuple[int, int]],
                        elem_dist: float,
                        wavelength: float,
                        normalized: bool = True
                        ) -> np.ndarray:
    """
    from [Basar 2020]: "Indoor and Outdoor Physical Channel Modeling and Efficient Positioning for Reconfigurable Intelligent Surfaces in mmWave Bands"
    """
    if is_iterable(num_tx_antennas):
        N_vert, N_hor = num_tx_antennas[0], num_tx_antennas[1]
    else:
        N_vert, N_hor = split_to_close_to_square_factors(num_tx_antennas)

    d = elem_dist
    k = 2 * pi / wavelength
    theta, phi = ray_to_elevation_azimuth(tx_position, rx_position)
    coords = np.array([(x, y) for x in range(N_hor) for y in range(N_vert)])
    x = coords[:, 0]
    y = coords[:, 1]
    a = np.exp(1j * k * d * (x * np.sin(theta) + y * np.sin(phi) * np.cos(theta)))

    if normalized:
        a = a / np.linalg.norm(np.absolute(a))

    return a


def ULA_steering_vector(tx_position: np.ndarray,
                        rx_position: np.ndarray,
                        num_tx_antennas: Union[int, Tuple[int, int]],
                        elem_dist: float,
                        wavelength: float,
                        normalized: bool = True
                        ) -> np.ndarray:
    n = np.arange(num_tx_antennas)
    theta, phi = ray_to_elevation_azimuth(tx_position, rx_position)
    cos_theta = np.cos(theta)
    a = np.exp(-1j * 2 * pi * n * elem_dist * cos_theta / wavelength)

    if normalized:
        a = a / np.linalg.norm(np.absolute(a))

    return a


def MIMO_Ricean_channel(tx_position: np.ndarray,
                        rx_position: np.ndarray,
                        n_tx_antennas: int,
                        n_rx_antennas: int,
                        tx_elem_spacing: float,
                        rx_elem_spacing: float,
                        ricean_factor_dB: float,
                        wavelength: float,
                        pathloss_exponent: float = 2.,
                        tx_antenna_type: str = 'ULA',
                        rx_antenna_type: str = 'ULA',
                        extra_attenuation_dB: float = None,
                        rng: np.random.Generator = None
                        ) -> np.ndarray:

    if tx_antenna_type == 'ULA':
        tx_antenna_response_func = ULA_steering_vector
    elif tx_antenna_type == 'URA':
        tx_antenna_response_func = URA_steering_vector
    else:
        raise ValueError(f'Unexpected antenna type {tx_antenna_type} (expected "ULA" or "URA").')

    if rx_antenna_type == 'ULA':
        rx_antenna_response_func = ULA_steering_vector
    elif rx_antenna_type == 'URA':
        rx_antenna_response_func = URA_steering_vector
    else:
        raise ValueError(f'Unexpected antenna type {rx_antenna_type} (expected "ULA" or "URA").')

    kappa              = dBW_to_Watt(ricean_factor_dB)
    dist               = np.linalg.norm(tx_position - rx_position)
    pl                 = calculate_pathloss(dist, wavelength, pathloss_exponent, extra_attenuation_dB)
    tx_steering_vector = tx_antenna_response_func(tx_position, rx_position, n_tx_antennas, tx_elem_spacing, wavelength, normalized=True)
    rx_steering_vector = rx_antenna_response_func(rx_position, tx_position, n_rx_antennas, rx_elem_spacing, wavelength, normalized=True)
    a                  = np.outer(tx_steering_vector, rx_steering_vector)
    a                 *= np.sqrt(n_tx_antennas*n_rx_antennas)
    LOS_component      = np.sqrt(kappa / (kappa + 1)) * a
    awg_noise          = sample_gaussian_standard_normal(a.shape, rng)
    NLOS_component     = np.sqrt(1 / (kappa + 1)) * awg_noise
    channel_coeffs     = np.sqrt(pl) * (LOS_component + NLOS_component)

    channel_coeffs = channel_coeffs.T

    assert channel_coeffs.shape == (n_rx_antennas, n_tx_antennas)
    assert channel_coeffs.dtype == complex

    return channel_coeffs


def sample_channel_realizations(params: Parameters, n_samples, verbose=True, rng=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None: rng = np.random.default_rng()

    H_tx_ris       = np.empty((n_samples, params.Channels.N,  params.Channels.Nt), dtype=complex)
    H_ris_rx       = np.empty((n_samples, params.Channels.Nr, params.Channels.N), dtype=complex)
    H_tx_rx        = np.empty((n_samples, params.Channels.Nr, params.Channels.Nt), dtype=complex)

    it             = range(n_samples)
    if verbose: it = tqdm(it, desc='Generating channels')

    for i in it:
        H_tx_ris[i, :, :, ] = MIMO_Ricean_channel(
            params.Channels.tx_position,
            params.Channels.ris_position,
            params.Channels.Nt,
            params.Channels.N,
            params.wavelength()/2,
            params.wavelength()/2,
            params.Channels.tx_ris_rice_factor,
            params.wavelength(),
            params.Channels.pathloss_exp,
            'ULA',
            'URA',
            rng=rng
        )

        H_ris_rx[i, :, :, ] = MIMO_Ricean_channel(
            params.Channels.ris_position,
            params.Channels.rx_position,
            params.Channels.N,
            params.Channels.Nr,
            params.wavelength() / 2,
            params.wavelength() / 2,
            params.Channels.ris_rx_rice_factor,
            params.wavelength(),
            params.Channels.pathloss_exp,
            'URA',
            'ULA',
            rng=rng
        )

        H_tx_rx[i, :, :, ] = MIMO_Ricean_channel(
            params.Channels.tx_position,
            params.Channels.rx_position,
            params.Channels.Nt,
            params.Channels.Nr,
            params.wavelength() / 2,
            params.wavelength() / 2,
            params.Channels.tx_rx_rice_factor,
            params.wavelength(),
            params.Channels.pathloss_exp,
            'ULA',
            'ULA',
            params.Channels.extra_tx_rx_attenuation_dB,
            rng=rng
        )

    return H_tx_ris, H_ris_rx, H_tx_rx
