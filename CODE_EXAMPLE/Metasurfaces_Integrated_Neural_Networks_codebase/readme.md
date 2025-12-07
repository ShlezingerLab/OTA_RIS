Requires python>=3.9 The requred libraries can be installed through
```
pip install torch torchvision numpy scipy tqdm pandas matplotlib
```

To train and evaluate on classification exampls under Ricean fading use:
```
python training.py
```

The execution of the script will download datasets under the `./data/` directory and will generate the following output files udner the `./outputs/` directory:
- `./outputs/results.csv`: A CSV file where the results of all runs will be stored. Each run contains achieved accuracy and loss function as well as system parameters. Which parameters will be saved can be changed in `training.py`.
- `./outputs/training_log.json`: A JSON file that contains accuracy and loss values per epoch for all runs. Each run is identified by a unique key present in `./outputs/results.csv`.
- `./outputs/training_plots/*.png`: Contains automatically generated training curves of validation accuracy. One filename per run, named with each unique key from `./outputs/results.csv`.

----
Details regarding the code files:

- `channel_generators.py`: Contains the simulator code for generating random channel realizations of Ricean fading with fixed UE(TX), BS(RX), and MS positions as specified in `parameters.py`.
- `metasurface_modules.py`: Contains the code that models simple metasurfaces and SIMs, as well as controller torch modules that enable training with either reconfigurable or fixed MSs.
- `minn.py`: Contains the top-level MINN architecture composed of Encoder (TX), Decoder (RX), and Channel (MS_controller, fading, and SIM propagation) module that models the system end-to-end.
- `parameters.py`: Contains values for all parameters used in the code that can be tweaked to execute all different MINN variations. Comments with explanation for each parameter are given in the file.
- `training.py`: Main script that contains training and evaluation procedures, as well as data loading.
- `transceiver_modules.py`: Contains TX and RX DNN modules. Channel-aware and channel-agnostic versions are implemented for MNIST (and variations) and a channel-agnostic version is implemented for CIFAR-10 data set.
- `utils_misc.py`: Various functions needed in the code.
- `utils_torch.py`: Various functions to support pytorch operations, including a common dataloader for data and channels.
- `utils_training.py`: Various functions to support training, including a Logger class for writing the results and loaders for the different data sets.

----
You may use CTRL+C to stop the training procedure any time after the first epoch is completed. The script will terminate normally and save the results in the specified files. No resuming functionality is implemented, though.

Note that the script contains functionalities that are not described in the current version of the paper available online, as they have been requested by the Reviewers. Those are:
- Training on different datasets. MINST, Fashion-MNIST, K-MINST, and CIFAR-10 are supported. This can be changed from `Parameters.Training.data_set`.
- Training with CSI observation error in channel-aware transceivers (and MS controller if applicable). The error in the CSI is controlled by `Parameters.Channels.csi_noise_dB`. Set to `None` to ignore erroneous CSI cases.
- Training with splitting the TX-encoded vector over multiple transmissions over the `Nt` antennas, assuming fading remains static. The number of transmissions per (static fading) frame (TpF) is controlled by` Parameters.Channels.TpF` which can be set to `1` to ignore this case.


----
To create shell scripts for different runs of the training procedure, `training.py` supports changing parameter values through command line arguments that override the values in `parameters.py`. Example of usage:

```
$ python parameters.py Channels.N 64 Channels.Nt 32 MINN.dataset FMNIST Auxiliary.numpy_seed 42
```
In such cases, `Training.verbose_level` might be set to `0` to disable printing.

