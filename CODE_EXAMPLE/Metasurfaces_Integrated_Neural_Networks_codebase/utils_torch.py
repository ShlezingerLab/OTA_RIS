import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Iterator, Tuple, Any, Literal


class DataAndChannelsLoader(DataLoader):
    def __init__(self, data_loader: DataLoader, channels: List[Dataset], **kwargs):
        self.data_loader = data_loader
        self.channels = channels
        self.batch_size = data_loader.batch_size

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        for data_batch in self.data_loader:
            current_batch_size = data_batch[0].size(0)

            channel_batches = []
            for channel in self.channels:
                random_indices = torch.randint(0, len(channel), (current_batch_size,)).tolist()

                channel_samples_list = [channel[idx][0] for idx in random_indices]

                channel_batch = torch.stack(channel_samples_list, dim=0)
                channel_batches.append(channel_batch)

            yield tuple(list(data_batch) + list(channel_batches))


class ComplexLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=torch.complex64):
        super().__init__()
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
        self.device = device
        self.dtype = dtype

        weight = torch.ones(self.normalized_shape) + 1j * torch.ones(self.normalized_shape)
        weight = weight.to(self.dtype)

        bias   = torch.zeros(self.normalized_shape) + 1j * torch.zeros(self.normalized_shape)
        bias   = bias.to(self.dtype)

        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(weight)

            if self.use_bias:
                self.bias = torch.nn.Parameter(bias)
            else:
                self.bias = self.register_buffer('bias', bias)

        else:
            self.register_buffer('weight', bias)
            self.register_buffer('bias', bias)


    def forward(self, x):
        start_dim = x.ndim - len(self.normalized_shape)
        dims      = list(range(start_dim, x.ndim))
        E_x       = torch.mean(x, dim=dims)
        var_x     = torch.var(x, dim=dims, correction=0)

        y         = x - E_x.view([-1] + [1]*len(self.normalized_shape))
        y         = y / torch.sqrt(var_x.view([-1] + [1]*len(self.normalized_shape)) + self.eps)
        y         = y * self.weight
        y         = y + self.bias
        return y


def matmul_by_diag_as_vector(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Given a 2D (or batched) matrix M and a vector (or batched matrix) v, perform the operation X = M @ diag(v),
    where diag(v) gives a square matrix with elements of v along its main diagonal,
    WITHOUT constructing the (sparse) diagonal matrix for memory and time efficiency.

    This function accepts batched versions of the arguments and performs the operations as such:
    - If M is of shape (m,n) and v is n-dimensional, then the output is of shape (m,n) - i.e. normal matmul.
    - If M is of shape (b1,b2,...,bk,m,n) and v is of shape (b1,b2,...,bk,n), then the output is of
      shape (b1,b2,...,bk,m,n) - i.e. batched matrix multiplications are applied across the first k dimensions.
    """
    return M * v[..., None, :]


def matmul_with_diagonal_right(matrix, diag_vector):
    """
    :param matrix: Shape (b1, ..., bk, n, m)
    :param diag_vector: (b1, ..., bk, m) which is assumed to be (b1, ..., bk, m, m) with the last two dimensions having a diagonal matrix
    :return: (b1, ..., bk, n, m)
    """
    if matrix.shape[-1] != diag_vector.shape[-1]: raise ValueError

    diag_vector = diag_vector.unsqueeze(-2)
    res         = matrix * diag_vector

    return res


def sample_awgn_torch(mean, stdev, shape, device=None):
    noise_real = torch.normal(mean=mean, std=stdev / 2, size=shape, device=device)
    noise_imag = torch.normal(mean=mean, std=stdev / 2, size=shape, device=device)
    noise = torch.complex(noise_real, noise_imag)
    return noise



def select_torch_device(preferred_device: Literal['cpu', 'cuda', 'mps'], verbose=True) -> torch.device:
    """
    Intelligently selects a torch device based on preference and availability.

    Policy:
    1. If the preferred_device is 'cpu', use 'cpu'.
    2. If a hardware device ('cuda' or 'mps') is preferred, check that device first.
    3. If the preferred device is unavailable, check the other hardware device.
    4. If neither hardware device is available, fall back to 'cpu'.

    Args:
       preferred_device: The user's preferred device ('cpu', 'cuda', or 'mps').
       verbose: If True, prints the selection outcome.

    Returns:
       A torch.device object representing the selected device.

    Raises:
       ValueError: If the preferred_device string is invalid.
    """
    # 1. Input Validation and Normalization
    valid_devices = {'cpu', 'cuda', 'mps'}
    if preferred_device.lower() not in valid_devices:
        raise ValueError(f"Invalid device specified: '{preferred_device}'. Must be one of {valid_devices}")

    preferred_device_lower = preferred_device.lower()
    selected_device_name = 'cpu'  # Default fallback

    # Check for availability of CUDA and MPS
    cuda_available = torch.cuda.is_available()
    # Note: MPS backend availability check
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # Determine the ordered check list
    # Format: (device_name, is_available_check)
    check_order = []

    if preferred_device_lower == 'cpu':
        # Policy 1: If CPU is requested, stick to CPU.
        selected_device_name = 'cpu'

    elif preferred_device_lower == 'cuda':
        # Policy 2, 3: Check CUDA, then MPS, then fallback to CPU
        check_order.append(('cuda', cuda_available))
        if preferred_device_lower != 'mps':
            check_order.append(('mps', mps_available))

    elif preferred_device_lower == 'mps':
        # Policy 2, 3: Check MPS, then CUDA, then fallback to CPU
        check_order.append(('mps', mps_available))
        if preferred_device_lower != 'cuda':
            check_order.append(('cuda', cuda_available))

    # 2. Iterate through the check order (if applicable)

    found_hardware_device = False
    for device_name, available in check_order:
        if available:
            selected_device_name = device_name
            found_hardware_device = True
            break

    # 3. Handle verbose messaging and return

    if verbose:
        msg_parts = [f"Preferred device was '{preferred_device_lower}'.", ]

        if preferred_device_lower == selected_device_name:
            msg_parts.append(f"Using '{selected_device_name}'.")
        elif found_hardware_device:
            msg_parts.append(f"'{preferred_device_lower}' is not available. Falling back to '{selected_device_name}'.")
        else:
            # Fallback to CPU
            msg_parts.append(
                f"Neither '{preferred_device_lower}' nor the alternative (CUDA/MPS) is available. Falling back to 'cpu'.")

        print(" ".join(msg_parts))

    return torch.device(selected_device_name)
