## Example Code: Metasurfaces Integrated Neural Networks (MINN)

This document summarizes the structure and logic of the example code in
`CODE_EXAMPLE/Metasurfaces_Integrated_Neural_Networks_codebase`.

The code implements an **end‑to‑end learned communication system** with:
- A deep‑learning **transmitter (Encoder)** and **receiver (Decoder)**.
- A configurable **metasurface / SIM (RIS/SIM controller or fixed surface)**.
- A **Ricean MIMO channel simulator**.
- A full **training / evaluation loop** on MNIST‑like datasets (and CIFAR‑10).

Running `python training.py` trains the system and writes results under `./outputs/`.

---

### High‑Level Training Flow

The main script is `training.py` and its `main()` function performs:

1. **Parse CLI overrides**
   - `change_parameter_values_from_command_line_arguments()` reads argument pairs like
     `Channels.N 64 Training.dataset FMNIST` and applies them via `change_param_value()`.

2. **Set up global parameters and RNG**
   - Instantiate `Parameters()` (from `parameters.py`).
   - Create a NumPy RNG (`Auxiliary.rng`) using `Auxiliary.numpy_seed`.

3. **Load data and pre‑generate channel realizations**
   - `load_data(params)` in `training.py`:
     - Calls `load_channels(params)` → `sample_channel_realizations(...)` in `channel_generators.py` to generate
       Ricean MIMO channel tensors for:
       - TX→RIS (`H_tx_ris`)
       - RIS→RX (`H_ris_rx`)
       - TX→RX direct link (`H_tx_rx`)
     - Wraps these as `TensorDataset`s.
     - Loads the selected dataset (`MNIST`, `FMNIST`, `KMNIST`, or `CIFAR10`) via `utils_training.py`:
       - `load_MNIST_data`, `load_FashionMNIST_data`, `load_KMNIST_data`, `load_CIFAR10_data`.
     - Sets `Auxiliary.data_shape` accordingly.
     - Combines data and channel datasets into a joint loader via
       `utils_torch.DataAndChannelsLoader`.

4. **Construct the end‑to‑end MINN model**
   - `construct_minn(params)` in `minn.py`:
     - Chooses **Encoder** and **Decoder**:
       - Channel‑aware: `ChannelAwareEncoder`, `ChannelAwareDecoder`.
       - Channel‑agnostic: `ChannelAgnosticEncoder`, `ChannelAgnosticDecoder`.
       - CIFAR‑10 uses `AdvancedChannelAgnosticEncoder`.
     - Chooses metasurface module depending on:
       - `MINN.metasurface_type` in `{ 'RIS', 'SIM', None }`.
       - `MINN.metasurface_control` in `{ 'static', 'reconfigurable' }`.
       - Uses classes from `metasurface_modules.py`:
         - `TrainableFixedRis`, `TrainableFixedSim`,
         - `RisController`, `SimController`.
     - Wraps everything in a `Minn` object.

5. **Select device, configure logging, and train**
   - Device: `select_torch_device(...)` in `utils_torch.py` chooses among `cuda`, `mps`, `cpu`.
   - Logging: `SingleTrainingLogger` (in `utils_training.py`) is configured with:
     - System parameters summary (`configure_logger_setup_info`).
     - Output paths (`results.csv`, `training_log.json`, plots directory).
   - Training loop: `train_model(...)` in `training.py`:
     - Uses `Adam` optimizer, `CrossEntropyLoss`.
     - For each epoch:
       - Compute current transmit power `P_curr` via `determine_current_power_value`
         (implements the schedule `Training.P_value_schedule_dBm`).
       - For each batch from `DataAndChannelsLoader`:
         - Build a `TransmissionVariables` object (see below) with:
           - `inputs`, `targets`,
           - channel tensors (`H_ue_ris`, `H_ris_bs`, `H_ue_bs`),
           - `P_curr`.
         - Optionally inject CSI noise via `apply_noise_to_channel`.
         - Move everything to the selected device.
         - Forward through `Minn` → get logits.
         - Compute loss, backprop, optimizer step.
       - Run validation via `evaluate_model` (same `Minn`, no grad).
       - Log metrics (`acc`, `train_loss`, `val_loss`, `P_curr`) via `SingleTrainingLogger.log_epoch`.
   - After training:
     - `logger.log_final(...)`, `save_training_results()`, `save_final_results()`.
     - `logger.plot_training_curve()` (accuracy vs epoch, with setup info in the figure).

---

### Core Configuration: `parameters.py`

The `Parameters` class groups all configurable hyperparameters:

- **`Channels`**:
  - Antennas: `Nt` (TX), `Nr` (RX), `N` (RIS / per SIM layer), `n_sim_layers`.
  - Transmissions per fading frame: `TpF`.
  - CSI noise level: `csi_noise_dB` (None = perfect CSI).
  - Carrier frequency: `freq`, noise power `noise_sigma_sq`.
  - Ricean K‑factors for different links (TX–RX, TX–RIS, RIS–RX).
  - Positions: `tx_position`, `rx_position`, `ris_position`.
  - SIM geometry: `sim_layers_distance`, `sim_elem_width`.

- **`MINN`**:
  - `csi_knowledge`: `'aware'` or `'agnostic'`.
  - `metasurface_type`: `'RIS'`, `'SIM'`, or `None`.
  - `metasurface_control`: `'reconfigurable'` or `'static'`.

- **`Training`**:
  - Dataset (`'MNIST'`, `'FMNIST'`, `'KMNIST'`, `'CIFAR10'`), batch size, epochs, LR, weight decay.
  - Device preference (`preferred_device`).
  - Counts of preloaded channel realizations for train/validation.
  - Verbosity and `epoch_print_freq`.
  - Power schedule `P_value_schedule_dBm`: piecewise mapping from epoch to TX power.

- **`Paths`**:
  - `data_rootdir` and output files/dirs (`output_rootdir`, `results_file`, `training_log_file`, `output_plots_subdir`).

- **`Auxiliary`**:
  - Complex dtype, RNG handle, `data_shape`, NumPy seed.

Utility:
- `Parameters.wavelength()` returns \(c / \text{freq}\).
- `change_param_value("ClassName.VarName", new_value)` provides programmatic/CLI overrides.

---

### End‑to‑End Model: `minn.py`

**TransmissionVariables dataclass**
- Bundles all data and channel tensors required during a forward pass:
  - `H_ue_bs`, `H_ue_ris`, `H_ris_bs`, and their noisy versions.
  - `inputs`, `targets`, `transmit_signal`, `received_signal`.
  - Optional `signal_at_ris`, `cascaded_channel`, `full_channel`.
  - `P_curr` (current TX power in Watts).
- Helper methods:
  - `.set(**kwargs)` safely updates fields.
  - `.to(device)` moves all tensors to a torch device.
  - `.to_numpy(inplace=False)` converts all tensors to NumPy arrays (useful for logging/analysis).

**Minn class**
- Wraps encoder, metasurface controller (if any), SIM propagation model (if any), and decoder.
- Key internal methods:
  - `_apply_tx_coding_and_power_norm(tv)`:
    - Calls `encoder(tv)` → complex symbols of shape `(B, TpF, Nt)`.
    - Normalizes power and scales by `P_curr`, converting to complex tensor of shape `(B, TpF, Nt, 1)`.
  - `_fix_channel_dimensions(tv)`:
    - Broadcasts single‑frame channel matrices across `T = TpF` transmissions.
  - `_apply_RIS_cascaded_channel(tv)`:
    - For RIS case:
      - `MS_controller(tv)` produces per‑element phases → diagonal matrix.
      - Forms cascaded channel `C_ris_rx @ Φ @ C_tx_ris`, adds direct TX–RX link → `full_channel`.
  - `_apply_SIM_cascaded_channel(tv)`:
    - For SIM case:
      - `MS_controller(tv)` returns list of per‑layer phases; handed to a **ReconfigurableSimNet**.
      - Propagates through continuous multi‑layer SIM, then combines with TX–RIS link and direct link.
  - `_construct_received_signal(tv)`:
    - Computes `y = full_channel @ transmit_signal`.
    - Adds complex AWGN via `sample_awgn_torch`.
    - Concatenates real and imaginary parts along antenna dimension to produce real‑valued features `(B, T, 2*Nr)`.

**Forward()**
1. TX encoding + power scaling.
2. Channel dimension expansion.
3. Apply RIS or SIM cascaded channel (or just TX–RX if no metasurface).
4. Construct received signal + AWGN.
5. Pass to the decoder to produce logits over classes.

**construct_minn(params)**:
- Chooses encoder/decoder and metasurface module based on `params`.
- Enforces that `Auxiliary.data_shape` is set (data must be loaded first).
- Returns a fully constructed `Minn` instance.

---

### Metasurface and SIM Models: `metasurface_modules.py`

This file implements physical models of metasurfaces and their controllers:

- **Geometry and utilities**
  - `pairwise_distances`, `pairwise_vectors`, `mag`, `align_coords_to_plane`, `frob_norm`.
  - `normal_direction_vectors_along_plane` maps planes (`'xy'`, `'yz'`, `'zx'`, etc.) to normal vectors.

- **Metasurface**
  - Represents a single RIS/SIM layer:
    - Parameters: number of elements, grid layout, element area and spacing, position and orientation in 3D space.
    - Learns phase parameters `theta` (trainable), internally mapped to \( \phi = e^{j \cdot \text{sigmoid}(\theta) \cdot 2\pi} \).
    - Computes per‑element positions in 3D (`_get_element_positions`) and can visualize the surface.

- **MetasurfaceWithoutPhaseShifts**
  - Same geometry as `Metasurface` but phase parameters are **not** trainable.
  - Used with a separate controller module that sets the phases externally (for reconfigurable SIM).

- **Surface2SurfaceTransmission**
  - Implements the Rayleigh–Sommerfeld propagation between two metasurface layers.
  - Precomputes a complex matrix `W`:
    - Depends on pairwise distances and angles between all element pairs.
    - Encodes diffraction and pathloss.
  - `forward()` returns `W`, used to propagate fields between metasurfaces.

- **SimNet**
  - Implements a multi‑layer SIM:
    \( \Pi_m \left( \Omega_m \cdot \Xi_m \right) \Omega_1 \), where \( \Omega_m \) are metasurface phase matrices and \( \Xi_m \) are transmission matrices.
  - Builds:
    - A list of metasurface layers (`ris_layers`).
    - Corresponding surface‑to‑surface propagation layers (`transmission_layers`).
  - `forward(x)`:
    - Optionally passes `x` through an `input_module`.
    - Applies per‑layer metasurface configurations and propagation matrices.
    - Optionally uses an `output_module`.
  - `parameters()` returns a custom iterator over all learnable parameters, useful for grouping or logging.

- **ReconfigurableSimNet**
  - Inherits from `SimNet` but uses `MetasurfaceWithoutPhaseShifts` and external phase configurations:
    - `set_all_phis(all_phis)` is called before each forward pass.
    - `_get_RIS_configuration` reads phases from `self.all_phis`.

- **Controllers and fixed metasurfaces**
  - `RisController`:
    - Observes instantaneous CSI (`H_ue_bs`, `H_ue_ris`, `H_ris_bs`) and predicts a single RIS phase profile.
    - Uses CNNs over stacked real/imaginary channel features.
  - `SimController`:
    - Extends `RisController` to produce phase profiles for **multiple SIM layers** (one head per layer).
  - `TrainableFixedRis`:
    - Learns a **single, channel‑independent** RIS phase profile, shared across the batch.
  - `TrainableFixedSim`:
    - Learns fixed phase profiles for each SIM layer (no CSI dependence).

---

### Channel Simulation: `channel_generators.py`

This file implements realistic **Ricean fading** channel models:

- **Helpers**
  - `calculate_distances`, `ray_to_elevation_azimuth`.
  - `calculate_pathloss`: pathloss as a function of distance, wavelength, exponent, and extra attenuation.
  - `URA_steering_vector`, `ULA_steering_vector`:
    - Antenna array responses based on array geometry and AoA/AoD.

- **MIMO_Ricean_channel(...)**
  - Generates a complex MIMO channel matrix with:
    - Deterministic LOS component (`kappa` from Rice factor).
    - Random NLOS component (complex Gaussian).
    - Overall scaling with pathloss and antenna counts.

- **sample_channel_realizations(params, n_samples, ...)**
  - Uses `MIMO_Ricean_channel` to sample:
    - `H_tx_ris`: TX→RIS.
    - `H_ris_rx`: RIS→RX.
    - `H_tx_rx`: TX→RX direct.
  - Respects positions and Rice factors from `Parameters.Channels`.
  - Returns NumPy arrays later wrapped as torch `TensorDataset`s.

---

### Transceiver (Encoder/Decoder) Modules: `transceiver_modules.py`

Implements the neural encoders and decoders used by the `Minn` model:

- **ChannelAgnosticEncoder**
  - CNN‑based encoder for MNIST‑like datasets:
    - Convolutions + ReLU + flatten + linear layer → latent vector.
    - Reshapes to `(B, TpF, 2*Nt)` and maps to complex symbols `(B, TpF, Nt)` using real/imag parts.

- **ChannelAwareEncoder**
  - Splits processing into:
    - `source_encoder`: CNN that encodes the input image.
    - `channel_encoder`: MLP that processes noisy channel matrices (real+imag parts).
    - `final_encoder`: concatenates both embeddings and emits complex transmit symbols.

- **AdvancedChannelAgnosticEncoder**
  - Deeper CNN for CIFAR‑10 (3×32×32) with multiple conv‑BN‑ReLU blocks, pooling, dropout.
  - Ends with fully connected layers to the desired `(TpF, 2*Nt)` output.

- **ChannelAgnosticDecoder**
  - Combines per‑antenna received features over all transmissions:
    - `combiner`: linear layer from `2*Nr` to hidden dimension.
    - `classifier`: MLP over flattened `(TpF * hidden_dim)` to output class logits.

- **ChannelAwareDecoder**
  - Uses CSI as input in addition to received signal:
    - `channel_decoder` processes stacked real/imag channel matrices.
    - `classifier` combines decoded channel features with flattened received signal.

---

### Miscellaneous Utilities: `utils_misc.py` and `utils_torch.py`

**utils_misc.py**
- Types and numeric helpers:
  - `is_iterable`, `dBm_to_Watt`, `dBW_to_Watt`.
  - `sample_gaussian_standard_normal` for complex Gaussian noise.
  - `split_to_close_to_square_factors` to factor an integer into dimensions close to a square grid.
  - `repeat_num_to_list_if_not_list_already` to broadcast scalars to lists.

**utils_torch.py**
- Data utilities:
  - `DataAndChannelsLoader`:
    - Wraps a data loader and one or more channel datasets.
    - For each data batch, randomly samples matching channel realizations and yields combined batches.
- Complex layer normalization:
  - `ComplexLayerNorm` implements layer norm for complex tensors (weight/bias in complex space).
- Linear algebra helpers:
  - `matmul_by_diag_as_vector`, `matmul_with_diagonal_right` for efficient multiplication by diagonal matrices.
- Noise sampling:
  - `sample_awgn_torch` for complex AWGN.
- Device selection:
  - `select_torch_device(preferred_device, verbose)` encapsulates logic for choosing `cuda` / `mps` / `cpu`.

---

### Training Utilities and Data Loaders: `utils_training.py`

- Dataset loaders:
  - `load_MNIST_data`, `load_FashionMNIST_data`, `load_KMNIST_data`, `load_CIFAR10_data`:
    - Handle download, transforms (optionally with augmentation), and data loaders.
    - Set input shapes used by encoders via `Auxiliary.data_shape`.
  - `CIFAR10_dataset` (wrapper class) is provided for more customized usage, though the main script uses the standard torchvision dataset directly.

- Logging and outputs:
  - `SingleTrainingLogger`:
    - Collects per‑epoch metrics and final results.
    - Saves:
      - CSV of all runs (`results.csv`).
      - JSON training logs (`training_log.json`).
      - Optional training accuracy plots (PNG) with setup info.
    - Ensures the CSV schema is updated if new columns (parameters or metrics) are added.

---

### Main Script: `training.py`

In summary, `training.py` ties everything together:

1. Reads optional CLI parameter overrides.
2. Initializes `Parameters` and RNG.
3. Loads data and channels via `load_data`.
4. Builds the MINN model via `construct_minn`.
5. Chooses a torch device with `select_torch_device`.
6. Constructs a `SingleTrainingLogger` and prints system configuration.
7. Calls `train_model` to run the full training loop with periodic validation.
8. Saves logs, results, and plots to `./outputs/`.

This example code can be used as a **reference implementation** for:
- End‑to‑end differentiable communication systems.
- Learning‑based design of RIS / SIM metasurfaces.
- Studying performance under channel state uncertainty, different datasets, and transmit power schedules.
