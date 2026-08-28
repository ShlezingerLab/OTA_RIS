import numpy as np

def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator, Es: float = 1.0) -> np.ndarray:
    """
    Add complex AWGN to achieve target Es/N0 (per symbol).
    Es: average symbol energy (default 1.0).
    """
    x = np.asarray(x, dtype=np.complex128)
    snr_linear = 10**(snr_db/10.0)
    N0 = Es / snr_linear
    noise = (np.sqrt(N0/2) * (rng.standard_normal(x.shape) + 1j*rng.standard_normal(x.shape)))
    return x + noise

def flat_rayleigh(x: np.ndarray, snr_db: float, rng: np.random.Generator, Es: float = 1.0):
    """
    y = h*x + n, where h ~ CN(0,1). Returns (y, h).
    """
    x = np.asarray(x, dtype=np.complex128)
    h = (rng.standard_normal() + 1j*rng.standard_normal())/np.sqrt(2.0)
    y = h * x
    Es = np.mean(np.abs(x)**2)
    snr_linear = 10**(snr_db/10.0)
    N0 = Es / snr_linear
    n = (np.sqrt(N0/2) * (rng.standard_normal(x.shape) + 1j*rng.standard_normal(x.shape)))
    y += n
    return y, h

def equalize_mf(y: np.ndarray, h: complex) -> np.ndarray:
    """
    Coherent matched filter & phase correction: s_hat = y / h (ZF).
    """
    return y / h
