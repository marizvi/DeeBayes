import numpy as np

def calculate_snr(clean_signal, noisy_signal):
    noise = np.array(noisy_signal) - np.array(clean_signal)
    snr = 10 * np.log10(np.sum(np.array(clean_signal)**2) / np.sum(noise**2))
    return snr

def calculate_rmse(original_signal, denoised_signal):
    # Convert lists to numpy arrays
    original_signal = np.array(original_signal)
    denoised_signal = np.array(denoised_signal)

    # Calculate the difference between original and denoised signals
    difference = original_signal - denoised_signal

    # Calculate the squared difference
    squared_difference = difference ** 2

    # Calculate the mean of the squared difference
    mean_squared_difference = np.mean(squared_difference)

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_difference)

    return rmse

