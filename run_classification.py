import numpy as np
from sklearn.preprocessing import LabelEncoder
from models.classifier import ECGClassifier
from evaluate import get_patient_inference
import config
from utils.data_utils import prepare_noise_data 
import torch

# AAMI Standards Mapping
ARRHYTHMIC_BEATS = {'A', 'a', 'J', 'S', 'V', 'E', '!', 'F', 'Q', '/', 'f', '|', 'x'}

def prepare_labeled_data(model, device, noise_data):
    all_clean, all_noisy, all_denoised, all_labels = [], [], [], []
    le = LabelEncoder()
    
    # Process all records (Train + Test) for the final evaluation
    for pid in config.TRAIN_RECORDS + config.TEST_RECORDS:
        clean, noisy, denoised = get_patient_inference(pid, model, noise_data, device)
        
        # Determine labels using your windowing logic
        # (Simplified here: mapping beats to segments)
        for i in range(len(clean)):
            # Labeling logic
            label = "Arrhythmic" if np.random.random() > 0.7 else "Normal" # Placeholder for actual annotation logic
            all_labels.append(label)
            
        all_clean.append(clean)
        all_noisy.append(noisy)
        all_denoised.append(denoised)

    X_clean = np.vstack(all_clean)
    X_noisy = np.vstack(all_noisy)
    X_denoised = np.vstack(all_denoised)
    y = le.fit_transform(all_labels)
    
    return X_clean, X_noisy, X_denoised, y, le

if __name__ == "__main__":
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from models.networks import VDN
    vdn_model = VDN().double().to(device)
    # Load your trained VDN weights here...

    # 2. Get Data
    noise_data = prepare_noise_data()
    X_c, X_n, X_d, y, encoder = prepare_labeled_data(vdn_model, device, noise_data)

    # 3. SVM Experiment
    svm_clf = ECGClassifier()
    
    print("Step 1: Training SVM on Clean Data...")
    svm_clf.train(X_c, y)

    print("Step 2: Evaluating on Noisy Data (Baseline)...")
    svm_clf.evaluate(X_n, y, encoder, "Noisy Signals")

    print("Step 3: Evaluating on VDNet Denoised Data...")
    svm_clf.evaluate(X_d, y, encoder, "Denoised Signals")