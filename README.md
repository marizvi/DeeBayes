# VDNet-ECG: Variational Denoising Network for ECG Signal Restoration

[![Publication](https://img.shields.io/badge/Published-ScienceDirect-blue.svg)](https://www.sciencedirect.com/science/article/pii/S0010482525017780?dgcid=author)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official implementation of the research paper: **"A Variational Denoising Network for ECG Signal Enhancement"**. Our approach leverages a dual-network architecture to simultaneously estimate noise variance and reconstruct clean cardiac signals.

---

## 🔬 Model Architecture

Our model, **VDNet**, is composed of two primary sub-networks designed to handle the stochastic nature of ECG noise:

1.  **DNet (UNet):** A deep 1D U-Net with skip connections responsible for signal reconstruction.
2.  **SNet (DnCNN):** A sigma-estimation network that predicts the spatial noise map.

![Model Architecture](deebayes.jpg)



---

## 🌟 Key Features

* **Bayesian Framework:** Unlike standard GANs, VDNet operates on a variational inference principle, modeling noise as a distribution rather than a point estimate.
* **Dynamic Sigma Estimation:** Features an integrated `SNet` that generates a noise map to adjust the denoising strength pixel-by-pixel (sample-by-sample).
* **Physionet Integration:** Fully compatible with MIT-BIH Arrhythmia and NSTDB datasets.
* **Downstream Validation:** Proven to improve SVM classification accuracy from **66.18%** (noisy) to **85.22%** (denoised).

---

## 📂 Project Structure

```text
├── models/
│   ├── networks.py        # VDN, UNet (DNet), and DnCNN (SNet)
├── utils/
│   ├── data_utils.py     # WFDB loaders and normalization logic
│   ├── losses.py         # LogGamma and Variational Loss implementation
│   └── metrics.py        # SNR and RMSE calculation
├── train.py              # Main training script with checkpointing
├── evaluate.py           # Inference, CSV export, and visualization
└── config.py             # Hyperparameters (Batch size: 128, Epochs: 75)