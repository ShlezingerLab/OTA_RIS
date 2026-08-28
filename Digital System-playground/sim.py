import argparse
import numpy as np
from transceiver import simulate_awgn, simulate_rayleigh
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=4, help="Modulation order (2,4,16,64,...)")
    p.add_argument("--snr-start", type=float, default=0.0)
    p.add_argument("--snr-stop", type=float, default=20.0)
    p.add_argument("--snr-step", type=float, default=2.0)
    p.add_argument("--snr", type=float, default=None, help="Single SNR value (overrides sweep)")
    p.add_argument("--n-bits", type=int, default=100)
    p.add_argument("--channel", choices=["awgn","rayleigh"], default="awgn")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--plot-constellation", action="store_true", help="Plot received constellation for the given SNR (uses small sample).")
    p.add_argument("--E_s", type=float, default=1.0, help="Signal energy per symbol (E_s)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.snr is not None and args.plot_constellation:
        from .transceiver import Transmitter
        from .channel import awgn, flat_rayleigh, equalize_mf
        tx = Transmitter(args.M, rng, E_s=args.E_s)
        n_bits = args.n_bits
        bits = tx.random_bits(n_bits - (n_bits % int(np.log2(args.M))))
        s = tx.modulate(bits, E_s=args.E_s)
        if args.channel == "awgn":
            y = awgn(s, args.snr, rng, Es=args.E_s)
            y_plot = y
        else:
            y, h = flat_rayleigh(s, args.snr, rng)
            y_plot = equalize_mf(y, h)  # post-eq constellation

    if args.snr is not None:
        snrs = [args.snr]
    else:
        snrs = np.arange(args.snr_start, args.snr_stop+1e-9, args.snr_step)

    sim_fn = simulate_awgn if args.channel == "awgn" else simulate_rayleigh

    bers = []
    for snr in snrs:
        ber = sim_fn(args.M, snr, args.n_bits,rng, E_s=args.E_s)
        print(f"SNR={snr:5.1f} dB  |  BER={ber:.3e}")
        bers.append(ber)

    if len(snrs) > 1:
        plt.figure()
        plt.plot(snrs, bers)
        #plt.semilogy(snrs, bers, marker="o")
        plt.title(f"BER vs SNR  (M={args.M}, channel={args.channel})")
        plt.xlabel("SNR (Es/N0) [dB]")
        plt.ylabel("BER")
        plt.grid(True, which="both")
        plt.show()

if __name__ == "__main__":
    main()
