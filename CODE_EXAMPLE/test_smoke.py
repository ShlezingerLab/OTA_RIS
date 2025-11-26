from src.transceiver import simulate_awgn, simulate_rayleigh

def test_runs():
    ber_awgn = simulate_awgn(M=4, snr_db=10, n_bits=10000, rng=__import__("numpy").random.default_rng(0))
    ber_ray  = simulate_rayleigh(M=4, snr_db=10, n_bits=10000, rng=__import__("numpy").random.default_rng(0))
    assert 0 <= ber_awgn <= 0.5
    assert 0 <= ber_ray  <= 0.5
