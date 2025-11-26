import numpy as np

def _gray_code(n):
    return n ^ (n >> 1)

def _gray_inv(g):
    # inverse Gray via iterative xor
    n = g.copy()
    shift = 1
    while True:
        n ^= (n >> shift)
        if (1 << shift) > g.max()+1:  # safe stop
            break
        shift += 1
    return n

def _levels_sqrqam(M):
    m_side = int(np.sqrt(M))
    assert m_side*m_side == M, "M must be a perfect square for square QAM."
    # Gray levels for each axis
    ints = np.arange(m_side)
    gray = _gray_code(ints)
    # Map to odd integers centered at 0: ..., -3, -1, +1, +3, ...
    # Then normalize average energy to 1
    vals = (2*gray - (m_side-1))
    # Normalization factor to get Es=1
    Es = ( ( (vals**2).mean() ) * 2 )  # sum of I and Q averages
    norm = np.sqrt(Es)
    return vals / norm

def map_bits_to_symbols(bits: np.ndarray, M: int, Es: float = 1.0) -> np.ndarray:
    """
    Gray-coded BPSK/QPSK/QAM mapping to complex symbols with average energy Es.
    bits: 1D array of {0,1}
    M: modulation order (2, 4, ... square QAM)
    Es: desired average symbol energy (default 1.0)
    returns complex np.ndarray of symbols
    """
    bits = np.asarray(bits, dtype=np.uint8)
    assert Es >= 0, "Symbol energy Es must be non-negative."
    if M == 2:  # BPSK on real axis
        assert bits.size % 1 == 0
        s = 1.0 - 2.0*bits.astype(np.float64)
        s = s + 0j  # ensure complex dtype
        return s * np.sqrt(Es)
    if M == 4:  # QPSK (Gray): pair bits -> I,Q
        assert bits.size % 2 == 0
        b = bits.reshape(-1, 2)
        i = b[:,0].astype(np.float64)
        q = b[:,1].astype(np.float64)
        I = -(1.0 - 2.0*i)
        Q = -(1.0 - 2.0*q)
        s = (I + 1j*Q)/np.sqrt(2)  # Es=1
        return s * np.sqrt(Es)

    k = int(np.log2(M))
    assert bits.size % k == 0
    m_side = int(np.sqrt(M))
    assert m_side*m_side == M
    axis = _levels_sqrqam(M)

    # Split k bits into two groups of k/2 for I and Q
    b = bits.reshape(-1, k)
    ki = k//2; kq = k - ki
    i_bits = b[:, :ki]
    q_bits = b[:, ki:]

    i_int = i_bits.dot(1 << np.arange(ki-1, -1, -1))
    q_int = q_bits.dot(1 << np.arange(kq-1, -1, -1))

    # Natural -> Gray index on each axis
    i_gray = _gray_code(i_int)
    q_gray = _gray_code(q_int)

    I = axis[i_gray]
    Q = axis[q_gray]
    return (I + 1j*Q) * np.sqrt(Es)

def hard_demapper(symbols: np.ndarray, M: int) -> np.ndarray:
    """
    Hard-decision demapper (Euclidean nearest) returning bits {0,1}.
    """
    s = np.asarray(symbols, dtype=np.complex128)
    if M == 2:  # BPSK on real axis
        bits = (s.real < 0).astype(np.uint8)
        return bits

    if M == 4:  # QPSK Gray: threshold
        I = (s.real > 0).astype(np.uint8)
        Q = (s.imag > 0).astype(np.uint8)
        return np.stack([I, Q], axis=1).reshape(-1)

    k = int(np.log2(M))
    m_side = int(np.sqrt(M))
    axis = _levels_sqrqam(M)

    # Quantize I and Q to nearest axis levels
    I_idx = np.argmin((s.real[:, None] - axis[None, :])**2, axis=1)
    Q_idx = np.argmin((s.imag[:, None] - axis[None, :])**2, axis=1)

    # Gray -> natural index
    def gray_inv_vec(g):
        table = np.array([_gray_inv(np.array([i]))[0] for i in range(m_side)], dtype=int)
        return table[g]

    i_nat = gray_inv_vec(I_idx)
    q_nat = gray_inv_vec(Q_idx)

    ki = k//2; kq = k - ki
    def to_bits(x, k):
        return ((x[:, None] >> np.arange(k-1, -1, -1)) & 1).astype(np.uint8)

    i_bits = to_bits(i_nat, ki)
    q_bits = to_bits(q_nat, kq)
    out = np.concatenate([i_bits, q_bits], axis=1).reshape(-1)
    return out
