import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.networks import VDN
from utils.data_utils import load_mit_bih_records, add_noise, convert_numpy_tensor
from utils.metrics import calculate_snr, calculate_rmse 
import config


def get_patient_inference(patient_id, model, noise_data, device):
    """Processes a specific patient record and returns clean, noisy, and denoised signals."""
    # Load and normalize specific patient data
    clean_signals = load_mit_bih_records(config.DATA_PATH, [patient_id], config.SAMPLE_SIZE)
    
    # Add noise
    noisy_signals = []
    np.random.seed(101)
    for signal in clean_signals:
        index = np.random.randint(0, len(noise_data))
        noise = noise_data[index]
        noisy_signals.append(add_noise(signal.reshape(1, -1), noise, config.SNR_LEVEL).flatten())
    
    # Model Inference
    model.eval()
    with torch.no_grad():
        # VDN expects (Batch, Channels, Length)
        input_tensor = convert_numpy_tensor(noisy_signals).unsqueeze(1).to(device).double()
        # In 'train' mode, VDN returns (phi_Z, phi_sigma)
        phi_Z, _ = model(input_tensor, mode='train')
        
        # VDN logic: denoised = noisy - noise_estimate (phi_Z channel 0)
        noise_estimate = phi_Z[:, :1, :].detach().cpu()
        denoised_signals = (input_tensor.cpu() - noise_estimate).squeeze(1).numpy()

    return clean_signals, noisy_signals, denoised_signals

def run_full_evaluation(model, noise_data, device):
    """Iterates through test records and prints research metrics."""
    results = {}
    for patient in config.TEST_RECORDS:
        print(f"--- Patient: {patient} ---")
        clean, noisy, denoised = get_patient_inference(patient, model, noise_data, device)
        
        # Limit to 200 elements
        elements = min(200, len(clean))
        avg_snr_noisy = np.mean([calculate_snr(clean[i], noisy[i]) for i in range(elements)])
        avg_snr_denoised = np.mean([calculate_snr(clean[i], denoised[i]) for i in range(elements)])
        avg_rmse = np.mean([calculate_rmse(clean[i], denoised[i]) for i in range(elements)])
        
        print(f"SNR Noisy: {avg_snr_noisy:.4f}")
        print(f"SNR Denoised: {avg_snr_denoised:.4f}")
        print(f"RMSE: {avg_rmse:.4f}\n")
        
        results[patient] = (clean, noisy, denoised)
    return results

def plot_and_save_samples(patient_id, clean, noisy, denoised, num_samples=4):
    """Generates the 2x2 plots and saves CSV data for the paper."""
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    plot_len = 512 # Visualizing first 512 samples
    
    for i in range(num_samples):
        r, c = i // 2, i % 2
        x = list(range(plot_len))
        
        axs[r, c].plot(x, clean[i][:plot_len], label='Original', alpha=0.8)
        axs[r, c].plot(x, noisy[i][:plot_len], label='Noisy', alpha=0.5)
        axs[r, c].plot(x, denoised[i][:plot_len], label='Denoised', color='black', linewidth=1)
        
        axs[r, c].set_title(f"Patient {patient_id} - Sample {i}")
        axs[r, c].legend()
        
        # Save to CSV
        df = pd.DataFrame({
            'x': x,
            'original': clean[i][:plot_len],
            'noisy': noisy[i][:plot_len],
            'denoised': denoised[i][:plot_len]
        })
        df.to_csv(f"{patient_id}_sample_{i}_data.csv", index=False)
    
    plt.tight_layout()
    plt.savefig(f"comparison_{patient_id}.png")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = VDN().double().to(device)
    checkpoint = torch.load(config.CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    
    noise_data = np.random.normal(0, 0.05, (1000, config.SAMPLE_SIZE)) 

    # Run Eval
    all_results = run_full_evaluation(model, noise_data, device)
    
    # Generate plots for specific patients mentioned 
    for pid in ['223', '220']:
        if pid in all_results:
            c, n, d = all_results[pid]
            plot_and_save_samples(pid, c, n, d)