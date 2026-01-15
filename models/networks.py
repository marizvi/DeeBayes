import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    DNet: The Denoising Network part of VDN.
    A 1D U-Net architecture with skip connections for signal reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder: 6 levels of downsampling
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for i in range(6):
            self.encoders.append(nn.Sequential(
                nn.Conv1d(in_channels=prev_channels, out_channels=2**(i+1)*8, kernel_size=16, stride=2, padding=7),
                nn.BatchNorm1d(2**(i+1) * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            prev_channels = 2**(i+1) * 8

        # Decoder: Upsampling and concatenation with encoder features
        self.decoders = nn.ModuleList()
        # First decoder layer (Bottleneck to first upsample)
        self.decoders.append(nn.Sequential(
            nn.ConvTranspose1d(prev_channels, 2**5 * 8, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(2**(5) * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        
        prev_channels = 2**5 * 8
        for i in range(4, -1, -1):
            padding = 7
            kernel_size = 16
            # 2*prev_channels because of skip-connection concatenation
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose1d(2 * prev_channels, 2**i * 8, kernel_size=kernel_size, stride=2, padding=padding),
                nn.BatchNorm1d(2**(i) * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            prev_channels = 2**i * 8

        # Final reconstruction layer
        self.final_conv = nn.ConvTranspose1d(4 * prev_channels, out_channels, kernel_size=16, stride=2, padding=7)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Encoder Path
        encoder_outputs = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            encoder_outputs.append(x)

        x = self.encoders[-1](x)

        # Decoder Path with Skip Connections
        for decoder, encoder_output in zip(self.decoders, reversed(encoder_outputs)):
            x = decoder(x)
            x = torch.cat((x, encoder_output), dim=1)

        x = self.final_conv(x)
        x = self.bnorm(x)
        x = self.leaky_relu(x)
        return x


class DnCNN(nn.Module):
    """
    SNet: The Sigma Network part of VDN.
    A deep CNN designed to estimate the noise variance (sigma).
    """
    def __init__(self, in_channels=1, out_channels=2, dep=10, num_filters=64, slope=0.2):
        super(DnCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(slope, inplace=True)

        mid_layers = []
        for _ in range(1, dep-1):
            mid_layers.append(nn.Conv1d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True))
            mid_layers.append(nn.LeakyReLU(slope, inplace=True))
        self.mid_layers = nn.Sequential(*mid_layers)

        self.conv_last = nn.Conv1d(num_filters, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layers(x)
        out = self.conv_last(x)
        return out


class VDN(nn.Module):
    """
    Variational Denoising Network (VDN).
    Integrates DNet (UNet) for denoising and SNet (DnCNN) for noise estimation.
    """
    def __init__(self):
        super(VDN, self).__init__()
        self.DNet = UNet(in_channels=1, out_channels=2)
        self.SNet = DnCNN(in_channels=1, out_channels=2)

    def forward(self, x, mode='train'):
        mode = mode.lower()
        if mode == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode == 'test':
            return self.DNet(x)
        elif mode == 'sigma':
            return self.SNet(x)
        else:
            raise ValueError("Invalid mode. Choose from ['train', 'test', 'sigma']")