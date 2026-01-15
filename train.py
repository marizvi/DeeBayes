import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.networks import VDN
from utils.data_utils import load_mit_bih_records, add_noise, SignalDataset, prepare_noise_data
from utils.losses import vdn_loss_fn, sigma_estimate
import config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model & Optimizer
    model = VDN().double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    start_epoch = 0
    train_losses, val_losses = [], []

    # 2. Load Checkpoint
    if os.path.exists(config.CHECKPOINT_PATH):
        checkpoint = torch.load(config.CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train', [])
        val_losses = checkpoint.get('val', [])
        print(f"Resuming from epoch {start_epoch}")

    # 3. Data Loading (Example setup)
    # Note: Ensure you have downloaded the records to config.DATA_PATH
    train_clean = load_mit_bih_records(config.DATA_PATH, config.TRAIN_RECORDS, config.SAMPLE_SIZE)
    test_clean = load_mit_bih_records(config.DATA_PATH, config.TEST_RECORDS, config.SAMPLE_SIZE)
    
    noise_vector = prepare_noise_data() 
    train_noisy = add_noise(train_clean, noise_vector, config.SNR_LEVEL)
    test_noisy = add_noise(test_clean, noise_vector, config.SNR_LEVEL)

    train_loader = DataLoader(SignalDataset(train_noisy, train_clean), batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SignalDataset(test_noisy, test_clean), batch_size=config.BATCH_SIZE)

    # 4. Training Loop
    for epoch in range(start_epoch, config.N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for noisy, clean in train_loader:
            optimizer.zero_grad()
            
            # Sigma Estimation (on CPU then move to GPU)
            sigma_map = sigma_estimate(noisy.float().numpy(), clean.float().numpy(), config.WIN, config.SIGMA_SPATIAL)
            sigma_map = torch.from_numpy(sigma_map).unsqueeze(1).to(device)

            noisy, clean = noisy.unsqueeze(1).to(device), clean.unsqueeze(1).to(device)

            phi_Z, phi_sigma = model(noisy, mode='train')
            loss = vdn_loss_fn(phi_Z, phi_sigma, noisy, clean, sigma_map, config.EPS2, config.RADIUS)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * noisy.size(0)

        # 5. Validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for noisy, clean in test_loader:
                sigma_map = sigma_estimate(noisy.float().numpy(), clean.float().numpy(), config.WIN, config.SIGMA_SPATIAL)
                sigma_map = torch.from_numpy(sigma_map).unsqueeze(1).to(device)
                
                noisy, clean = noisy.unsqueeze(1).to(device), clean.unsqueeze(1).to(device)
                p_Z, p_sig = model(noisy, mode='train')
                v_loss += vdn_loss_fn(p_Z, p_sig, noisy, clean, sigma_map, config.EPS2, config.RADIUS).item() * noisy.size(0)

        train_losses.append(epoch_loss / len(train_loader.dataset))
        val_losses.append(v_loss / len(test_loader.dataset))

        print(f"Epoch {epoch+1}/{config.N_EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

        # Checkpoint Saving
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train': train_losses,
                'val': val_losses
            }, config.CHECKPOINT_PATH)

if __name__ == "__main__":
    main()