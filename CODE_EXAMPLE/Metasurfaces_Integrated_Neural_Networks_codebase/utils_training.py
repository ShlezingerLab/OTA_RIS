from collections import OrderedDict
from time import sleep
from typing import *
from typing import Union
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F

from parameters import Parameters


def load_MNIST_data(batch_size, data_dir=None):
    if data_dir is None:
        data_dir = './data'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])

    trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader, trainset, testset


def load_FashionMNIST_data(batch_size, data_dir=None):
    """
    Loads the Fashion-MNIST dataset (articles of clothing) with data augmentation
    applied to the training set for improved model generalization.
    Images: 28x28 grayscale, 10 classes.
    """
    if data_dir is None:
        data_dir = './data/'

    # Standard grayscale normalization (mean 0.5, std 0.5)
    normalize = transforms.Normalize((0.5,), (0.5,))

    # --- Training Transformations (Includes Augmentation) ---
    train_transform = transforms.Compose([
        # Augmentations for robustness
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.1),

        transforms.ToTensor(),
        normalize,
    ])

    # --- Testing Transformations (Deterministic: only Tensor conversion and Normalization) ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True,
                                                 download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False,
                                                download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    print(f"Loaded Fashion-MNIST: Train={len(trainset)}, Test={len(testset)}")
    return trainloader, testloader, trainset, testset


# --- KMNIST (Kuzushiji-MNIST) Loader ---
def load_KMNIST_data(batch_size, data_dir=None):
    """
    Loads the KMNIST (Kuzushiji-MNIST) dataset (Japanese characters).
    Images: 28x28 grayscale, 10 classes. More complex than MNIST.
    """
    if data_dir is None:
        data_dir = './data/'

    # Same transformation as MNIST as it's also 28x28 grayscale
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((0.5,), (0.5,)),
         ])

    # KMNIST is available directly in torchvision.datasets
    trainset = torchvision.datasets.KMNIST(root=data_dir, train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.KMNIST(root=data_dir, train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    print(f"Loaded KMNIST: Train={len(trainset)}, Test={len(testset)}")
    return trainloader, testloader, trainset, testset


class CIFAR10_dataset(Dataset):

    def __init__(self, data_dir, partition = "train", transform = None):

        self.partition = partition
        self.transform = transform
        if self.partition == "train":
            self.data = torchvision.datasets.CIFAR10(data_dir,
                                                     train=True,
                                                     download=True)
        else:
            self.data = torchvision.datasets.CIFAR10(data_dir,
                                                     train=False,
                                                     download=True)
    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Image
        image = self.data[idx][0]
        image_tensor = self.transform(image)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"img": image_tensor, "label": label}

def load_CIFAR10_data(batch_size, data_dir=None):
    if data_dir is None:
        data_dir = './data'

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                          download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                         download=True, transform=train_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader, trainset, testset



class SingleTrainingLogger:
    def __init__(self, params: Parameters,
                 setup_params: dict[str,Union[str,float,int]],
                 separate_output_file=None,
                 training_log_file=None,
                 verbose=True):
        custom_method_name        = params.Training.method_name
        self.params               = params
        self.verbose              = verbose
        self.training_info        = None
        self.final_info           = None
        self.output_pathname      = self.get_output_pathname(separate_output_file)
        self.training_pathname    = self.get_training_log_pathname(training_log_file)
        date_started              = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.unique_hash          = date_started if custom_method_name == '' else f"{custom_method_name}-{date_started}"
        self.info_to_log          = {
            'method_name'    : custom_method_name,
            'best_acc'       : None,  # to be filled at the end
            'P_at_best_acc'  : None,
            'final_acc'      : None,
            'P_at_final_acc' : None,
            'final_iter'     : None,
        }
        self.info_to_log = {**self.info_to_log,
                            **setup_params,  # append use-specified setup params to output dictionary
                            'date_started': date_started,
                            'date_ended'  : '',
                            'unique_hash' : self.unique_hash,}

    def get_output_pathname(self, separate_output_file):
        filename   = self.params.Paths.results_file if not separate_output_file else separate_output_file
        output_dir = self.params.Paths.output_rootdir
        file_path  = os.path.join(output_dir, filename)

        os.makedirs(output_dir, exist_ok=True)

        return file_path


    def get_training_log_pathname(self, training_log_file):
        filename   = self.params.Paths.training_log_file if not training_log_file else training_log_file
        output_dir = self.params.Paths.output_rootdir
        file_path  = os.path.join(output_dir, filename)

        os.makedirs(output_dir, exist_ok=True)

        return file_path


    def get_output_plot_pathname(self):
        output_dir = os.path.join(
            self.params.Paths.output_rootdir,
            self.params.Paths.output_plots_subdir,
        )
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, f"{self.unique_hash}.png")
        return filepath


    def log_epoch(self, epoch_num, **kwargs):
        if self.training_info is None:
            self.training_info = {'epoch': [epoch_num]}
            for key, val in kwargs.items():
                self.training_info[key] = [val]
        else:
            self.training_info['epoch'].append(epoch_num)
            for key, val in kwargs.items():
                self.training_info[key].append(val)


    def log_final(self, best_acc, P_at_best_acc, final_acc, P_at_final_acc, final_iter, **kwargs):
        self.info_to_log['best_acc']       = best_acc
        self.info_to_log['P_at_best_acc']  = P_at_best_acc
        self.info_to_log['final_acc']      = final_acc
        self.info_to_log['P_at_final_acc'] = P_at_final_acc
        self.info_to_log['final_iter']     = final_iter
        self.info_to_log['date_ended']     = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.info_to_log                   = {**self.info_to_log, **kwargs}  # append potential extra information




    def prepare_output_file_structure(self, output_info: dict):
        existing_df     = pd.read_csv(self.output_pathname)
        missing_in_file = [col for col in output_info if col not in existing_df.columns]
        extra_in_file   = [col for col in existing_df.columns if col not in output_info]

        if missing_in_file:
            if self.verbose: print(f"Warning: The following columns are missing in the existing CSV file: {missing_in_file}")
            # Add missing columns to the existing DataFrame with empty data
            for col in missing_in_file:
                existing_df[col] = None

        if extra_in_file:
            if self.verbose: print(f"Warning: The following extra columns are found in the existing CSV file: {extra_in_file}")

        # Ensure the order of columns is correct
        for col in output_info:
            if col not in existing_df.columns:
                existing_df[col] = None


    def save_final_results(self):
        if not os.path.exists(self.output_pathname):
            df = pd.DataFrame([self.info_to_log])
            df.to_csv(self.output_pathname, index=False)

        else:
            self.prepare_output_file_structure(self.info_to_log)
            existing_df = pd.read_csv(self.output_pathname)
            new_row     = pd.DataFrame([self.info_to_log])
            updated_df  = pd.concat([existing_df, new_row], ignore_index=True)
            updated_df.to_csv(self.output_pathname, index=False)

        if self.verbose: print(f'Wrote experiment details and final results to "{self.output_pathname}".')



    def save_training_results(self):
        try:
            with open(self.training_pathname, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file does not exist or is empty, start with an empty list
            data = []

        new_entry = {self.unique_hash: self.training_info}
        data.append(new_entry)
        with open(self.training_pathname, 'w') as file:
            json.dump(data, file, indent=2)


    def plot_training_curve(self, x_var='epoch', y_var='acc', ylim=(0,1)):
        x = self.training_info[x_var]
        y = self.training_info[y_var]

        plt.plot(x, y)
        plt.grid()
        plt.ylim(*ylim)
        plt.xlabel(x_var)
        plt.ylabel(y_var)

        # Prepare the setup_info text
        setup_info_text = "\n".join(f"{key}: {value}" for key, value in self.info_to_log.items())

        # Determine the location for the text box
        # Adjust the position (0.05, 0.95) as needed to fit your plot
        plt.gcf().text(0.05, 0.95, setup_info_text, fontsize=8, verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.5))

        if self.params.Paths.output_plots_subdir is not None:
            plot_filename = self.get_output_plot_pathname()
            plt.savefig(plot_filename)

        try:
            plt.show(block=False)
        except:
            if self.verbose: print('Warning: Showing training curve failed.')
