import numpy as np
import pywt
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ECGClassifier:
    def __init__(self, wavelet='db4', level=5):
        self.wavelet = wavelet
        self.level = level
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, class_weight='balanced')

    def extract_features(self, ecg_signals):
        """Extracts temporal and Wavelet (DWT) energy features."""
        features_list = []
        for i in range(ecg_signals.shape[0]):
            signal = ecg_signals[i, :]
            
            # 1. Global Temporal Features
            temp_feats = [
                np.mean(signal), np.std(signal), np.min(signal), 
                np.max(signal), np.ptp(signal), np.median(signal),
                np.sqrt(np.mean(signal**2)) # RMS
            ]
            
            # Zero-crossing rate
            zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal) if len(signal) > 1 else 0
            temp_feats.append(zcr)

            # 2. Wavelet Energy Features
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            wave_feats = [np.sum(np.square(c)) for c in coeffs]
            
            features_list.append(temp_feats + wave_feats)
        return np.array(features_list)

    def train(self, X_raw, y):
        features = self.extract_features(X_raw)
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features, y)
        print("SVM Training Complete.")

    def evaluate(self, X_raw, y, label_encoder, data_name="Test"):
        features = self.extract_features(X_raw)
        scaled_features = self.scaler.transform(features)
        preds = self.model.predict(scaled_features)
        
        acc = accuracy_score(y, preds)
        print(f"\n--- SVM Evaluation: {data_name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y, preds, target_names=label_encoder.classes_))
        return acc

    def save(self, path):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']