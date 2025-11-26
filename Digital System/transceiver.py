import numpy as np
from modulation import map_bits_to_symbols, hard_demapper
from channel import awgn, flat_rayleigh, equalize_mf

class Transmitter:
    def __init__(self, M:int,  rng: np.random.Generator, E_s:float=1.0):
        self.M = M
        self.k = int(np.log2(M))
        self.rng = rng
        self.E_s = E_s
    def random_bits(self, n_bits:int) -> np.ndarray:
        return self.rng.integers(0, 2, size=n_bits, dtype=np.uint8)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        return map_bits_to_symbols(bits, self.M, self.E_s)

class Receiver:
    def __init__(self, M:int):
        self.M = M
        self.k = int(np.log2(M))

    def demodulate_hard(self, y_eq: np.ndarray) -> np.ndarray:
        return hard_demapper(y_eq, self.M)

def carrier(
    s: np.ndarray,
    fc: float = 1e6,
    fs: float = 10e6,
    t0: float = 0.0,
) -> np.ndarray:
    """
    Convert a complex baseband sequence into a real-valued passband waveform.

    Parameters
    ----------
    s : np.ndarray
        Complex baseband samples (I + jQ).
    fc : float, optional
        Carrier frequency in Hz. Defaults to 1 MHz.
    fs : float, optional
        Sampling rate in Hz. Defaults to 10 MHz (10× carrier).
    t0 : float, optional
        Initial time offset in seconds. Defaults to 0.

    Returns
    -------
    np.ndarray
        Real-valued passband samples.
    """
    s = np.asarray(s, dtype=np.complex128)
    if s.ndim != 1:
        raise ValueError("Baseband sequence `s` must be one-dimensional.")

    n = np.arange(s.size, dtype=np.float64)
    t = t0 + n / fs
    return np.real(s * np.exp(1j * 2 * np.pi * fc * t))

def _design_lowpass_fir(cutoff: float, fs: float, num_taps: int) -> np.ndarray:
    if not 0 < cutoff < fs / 2:
        raise ValueError("cutoff must be between 0 and Nyquist (fs/2).")
    if num_taps < 3:
        raise ValueError("num_taps must be >= 3.")
    if num_taps % 2 == 0:
        num_taps += 1  # enforce odd length for linear phase symmetry

    n = np.arange(num_taps) - (num_taps - 1) / 2
    sinc_arg = 2 * cutoff / fs
    h = sinc_arg * np.sinc(sinc_arg * n)
    window = np.hamming(num_taps)
    h *= window
    h /= np.sum(h)
    return h.astype(np.float64)

def demodulate(
    x_rf: np.ndarray,
    fc: float = 1e6,
    fs: float = 10e6,
    t0: float = 0.0,
    lpf_cutoff: float | None = None,
    num_taps: int = 101,
) -> np.ndarray:
    """
    Convert a real-valued passband waveform back to its complex baseband equivalent.

    Parameters
    ----------
    x_rf : np.ndarray
        Real (or complex) RF samples after the channel.
    fc : float, optional
        Carrier frequency in Hz. Defaults to 1 MHz.
    fs : float, optional
        Sampling rate in Hz. Defaults to 10 MHz.
    t0 : float, optional
        Initial time offset in seconds. Defaults to 0.
    lpf_cutoff : float, optional
        Low-pass filter cutoff (Hz). Defaults to fs/5.
    num_taps : int, optional
        Number of FIR taps for the low-pass filter. Defaults to 101.

    Returns
    -------
    np.ndarray
        Complex baseband samples after mixing and low-pass filtering.
    """
    x_rf = np.asarray(x_rf, dtype=np.complex128)
    if x_rf.ndim != 1:
        raise ValueError("RF sequence `x_rf` must be one-dimensional.")

    n = np.arange(x_rf.size, dtype=np.float64)
    t = t0 + n / fs
    mixed = x_rf * np.exp(-1j * 2 * np.pi * fc * t)

    if lpf_cutoff is None:
        lpf_cutoff = fs / 5.0
    taps = _design_lowpass_fir(lpf_cutoff, fs, num_taps)
    baseband = np.convolve(mixed, taps, mode="same")
    return baseband.astype(np.complex128)

def simulate_awgn(M:int, snr_db:float, n_bits:int, rng, E_s:float=1.0):
    tx = Transmitter(M, rng, E_s)
    rx = Receiver(M)

    bits_tx = tx.random_bits(n_bits - (n_bits % tx.k))
    s = tx.modulate(bits_tx)
    s_c = carrier(s)
    y = awgn(s, snr_db, rng, Es=E_s)
    y_b = demodulate(y)
    bits_rx = rx.demodulate_hard(y)
    ber = np.mean(bits_rx != bits_tx)
    return ber

def simulate_rayleigh(M:int, snr_db:float, n_bits:int, rng, E_s:float=1.0):
    tx = Transmitter(M, rng)
    rx = Receiver(M)

    bits_tx = tx.random_bits(n_bits - (n_bits % tx.k))
    s = tx.modulate(bits_tx)
    y, h = flat_rayleigh(s, snr_db, rng, Es=E_s)
    y_eq = equalize_mf(y, h)  # perfect CSI equalization
    bits_rx = rx.demodulate_hard(y_eq)
    ber = np.mean(bits_rx != bits_tx)
    return ber
