import torch
import torch.nn as nn
import numpy as np
from minn import TransmissionVariables


class ChannelAgnosticEncoder(torch.nn.Module):
    def __init__(self, Nt, Nr, N, TpF, chw):
        super(ChannelAgnosticEncoder, self).__init__()
        self.Nt           = Nt
        self.Nr           = Nr
        self.N            = N
        self.TpF          = TpF
        self.chw          = chw
        #self.hidden_dim   = hidden_dim

        assert np.prod(self.chw) == 1*28*28, 'This class works only for (1,28,28) sized inputs for MNIST'

        self.encoder = nn.Sequential(
            nn.Conv2d(self.chw[0], 32, kernel_size=4, stride=2, padding=1),
            # Input: (num_channels, height, width), Output: (32, height/2, width/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, height/4, width/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, height/8, width/8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (self.chw[1] // 8) * (self.chw[2] // 8), self.TpF * 2*self.Nt),
            #nn.Tanh(),
        )

    def forward(self, tv: TransmissionVariables):
        x       = tv.inputs
        encoded = self.encoder(x)
        encoded = encoded.view(-1, self.TpF, 2*self.Nt)
        encoded = torch.complex(encoded[:,:,:self.Nt], encoded[:,:,self.Nt:])
        return encoded




class ChannelAwareEncoder(nn.Module):
    def __init__(self, Nt, Nr, N, TpF, chw, hidden_dim=32):
        super(ChannelAwareEncoder, self).__init__()
        self.Nt           = Nt
        self.Nr           = Nr
        self.N            = N
        self.TpF          = TpF
        self.chw          = chw
        self.hidden_dim   = hidden_dim

        assert np.prod(self.chw) == 1*28*28, 'This class works only for (1,28,28) sized inputs for MNIST'



        self.source_encoder = nn.Sequential(
            nn.Conv2d(self.chw[0], 32, kernel_size=4, stride=2, padding=1),
            # Input: (num_channels, height, width), Output: (32, height/2, width/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, height/4, width/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, height/8, width/8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (self.chw[1] // 8) * (self.chw[2] // 8), self.hidden_dim),
            #nn.Softmax(),
            # nn.Tanh(),
        )

        self.channel_dim     = 2 * (self.Nt * self.N + self.Nr * self.N + self.Nt * self.Nr)
        self.channel_encoder = nn.Sequential(
            nn.LayerNorm(self.channel_dim),
            nn.Linear(self.channel_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim),
            #nn.Softmax()
        )

        self.final_encoder = nn.Sequential(
            nn.LayerNorm(2*hidden_dim),
            nn.Linear(2*hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.TpF * 2*self.Nt),
        )


    def forward(self, tv: TransmissionVariables):
        b            = tv.inputs.shape[0]
        src_encoded  = self.source_encoder(tv.inputs)
        C_ue_bs      = tv.H_ue_bs_noise.view(b, -1)
        C_ue_ris     = tv.H_ris_bs_noise.view(b, -1)
        C_ris_bs     = tv.H_ue_ris_noise.view(b, -1)
        C            = torch.concatenate([C_ue_bs, C_ris_bs, C_ue_ris], dim=1)
        C            = torch.concatenate([torch.real(C), torch.imag(C)], dim=1)
        C_encoded    = self.channel_encoder(C)
        x            = torch.concatenate([src_encoded, C_encoded], dim=1)
        out          = self.final_encoder(x)
        out          = out.view(b, self.TpF, 2*self.Nt)
        out          = torch.complex(out[:,:,:self.Nt], out[:,:,self.Nt:])

        return out



class AdvancedChannelAgnosticEncoder(nn.Module):
    def __init__(self, Nt, Nr, N, TpF, chw, hidden_dim=64):
        super(AdvancedChannelAgnosticEncoder, self).__init__()
        self.Nt           = Nt
        self.Nr           = Nr
        self.N            = N
        self.TpF          = TpF
        self.chw          = chw
        #self.hidden_dim   = hidden_dim

        assert np.prod(self.chw) == 3*32*32, 'This class works only for (3,28,28) sized inputs for CIFAR-10'


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32 * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32 * 2)
        self.bn2 = nn.BatchNorm2d(128 * 2)
        self.bn3 = nn.BatchNorm2d(256 * 2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(4096 * 2, 4096 * 2)
        self.fc2 = nn.Linear(4096 * 2, 2048 * 2)
        self.fc3 = nn.Linear(2048 * 2, self.TpF * 2*self.Nt)
        self.relu = nn.ReLU()

    def forward(self, tv: TransmissionVariables):
        b = tv.inputs.shape[0]
        x = tv.inputs

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.dropout2d(x)

        x = self.relu(self.bn3(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.maxpool(x)
        x = self.dropout2d(x)

        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        out = out.view(b, self.TpF, 2 * self.Nt)
        out = torch.complex(out[:, :, :self.Nt], out[:, :, self.Nt:])

        return out



class ChannelAgnosticDecoder(nn.Module):
    def __init__(self, Nt, Nr, N, TpF, hidden_dim=32):
        super(ChannelAgnosticDecoder, self).__init__()
        self.Nt           = Nt
        self.Nr           = Nr
        self.N            = N
        self.TpF          = TpF
        self.hidden_dim   = hidden_dim
        self.n_classes    = 10



        self.combiner = nn.Linear(2*self.Nr, self.hidden_dim)


        self.classifier = nn.Sequential(
            #nn.Linear(2*receive_antennas, encoder_depth * encoder_dim),
            #nn.Tanh(),
            nn.Linear(self.TpF * self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes),
            #nn.Softmax(dim=1),
        )

    def forward(self, transmission_variables: TransmissionVariables):
        y   = transmission_variables.received_signal
        z   = self.combiner(y)
        z   = z.view(-1, self.TpF * self.hidden_dim)
        out = self.classifier(z)
        return out



class ChannelAwareDecoder(nn.Module):
    def __init__(self, Nt, Nr, N, TpF, hidden_dim=32):
        super(ChannelAwareDecoder, self).__init__()
        self.Nt           = Nt
        self.Nr           = Nr
        self.N            = N
        self.TpF          = TpF
        self.hidden_dim   = hidden_dim
        self.n_classes    = 10
        self.received_signal_size = 2 * self.TpF * self.Nr
        self.channel_dim          = 2 * (self.Nt * self.N + self.Nr * self.N + self.Nt * self.Nr)

        self.channel_decoder = nn.Sequential(
            nn.LayerNorm(self.channel_dim),
            nn.Linear(self.channel_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + self.received_signal_size),
            nn.Linear(self.hidden_dim + self.received_signal_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.n_classes),
            #nn.Softmax(dim=1),
        )

    def forward(self, tv: TransmissionVariables):
        b         = tv.inputs.shape[0]
        C_ue_bs   = tv.H_ue_bs_noise[:,0,:,:].view(b, -1)
        C_ue_ris  = tv.H_ris_bs_noise[:,0,:,:].view(b, -1)
        C_ris_bs  = tv.H_ue_ris_noise[:,0,:,:].view(b, -1)
        C         = torch.concatenate([C_ue_bs, C_ris_bs, C_ue_ris], dim=1)
        C         = torch.concatenate([torch.real(C), torch.imag(C)], dim=1)

        C_decoded = self.channel_decoder(C)

        y         = tv.received_signal
        y         = y.view(b, self.received_signal_size)
        x         = torch.concatenate([y, C_decoded], dim=1)
        out       = self.classifier(x)

        return out
