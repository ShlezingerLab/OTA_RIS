# Stubs to help you integrate learned encoders/decoders later (e.g., PyTorch).

def learned_encoder_placeholder(x_bits_or_features):
    """
    Input: bits or features
    Output: complex symbols (np.ndarray of complex64/complex128) with Es≈1
    """
    raise NotImplementedError("Plug your PyTorch/JAX encoder here.")

def learned_decoder_placeholder(y_observations):
    """
    Input: received complex symbols (optionally with side-info)
    Output: recovered bits / task output
    """
    raise NotImplementedError("Plug your PyTorch/JAX decoder here.")
