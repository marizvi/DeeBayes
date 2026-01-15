import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset

def convert_numpy_tensor(arr):
    return torch.from_numpy(np.array(arr))

class SignalDataset(Dataset):
    def __init__(self, noisy_data, clean_data):
        self.noisy_data = convert_numpy_tensor(noisy_data)
        self.clean_data = convert_numpy_tensor(clean_data)

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        return self.noisy_data[idx], self.clean_data[idx]

def load_mit_bih_records(path, records, sample_size, sampto=650000):
    data = []
    for patient in records:
        try:
            record = wfdb.rdsamp(f"{path}/{patient}", sampto=sampto)
            signal = record[0][:, 0]
            # Chunking and Min-Max Normalization
            for i in range(0, len(signal), sample_size):
                segment = signal[i:i+sample_size]
                if len(segment) < sample_size: break
                q, p = np.min(segment), np.max(segment)
                normalized = (segment - q) / (p - q) if p != q else segment
                data.append(normalized)
        except Exception as e:
            print(f"Error loading record {patient}: {e}")
    return np.array(data)

def add_noise(clean_data, noise_signal, desired_snr):
    # Ensure noise signal is tiled to match clean_data length
    noise_seg = noise_signal[:clean_data.shape[1]]
    clean_power = np.sum(clean_data**2, axis=1, keepdims=True)
    noise_power = np.sum(noise_seg**2)
    
    scaling_factor = np.sqrt(clean_power / (noise_power * (10**(desired_snr / 10))))
    noisy_data = clean_data + (noise_seg * scaling_factor)
    return noisy_data