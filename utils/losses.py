import torch
import torch.nn as nn
import numpy as np
import cv2
import math
from math import pi, log
from torch.autograd import Function as autoF

# Clip bounds for stability
log_max = log(1e4)
log_min = log(1e-8)

class LogGamma(autoF):
    """
    Implementation of the logarithm of gamma Function.
    Improved version using np.vectorize for robustness across batch shapes.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_np = input.detach().cpu().numpy()
        v_lgamma = np.vectorize(math.lgamma)
        out_np = v_lgamma(input_np)
        return torch.from_numpy(out_np).to(device=input.device).type(dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return torch.digamma(input) * grad_output

log_gamma = LogGamma.apply

def sigma_estimate(noisy_signal, clean_signal, win, sigma_spatial):
    noise2 = (noisy_signal - clean_signal) ** 2
    # cv2.GaussianBlur handles (Batch, Length) as (Height, Width)
    sigma2_map_est = cv2.GaussianBlur(noise2, (1, win), sigma_spatial)
    sigma2_map_est = np.where(sigma2_map_est < 1e-10, 1e-10, sigma2_map_est).astype(np.float32)
    return sigma2_map_est

def vdn_loss_fn(out_denoise, out_sigma, signal_noisy, clean_signal, sigmaMap, eps2, radius=3):
    C = 1 # Lead channel count
    p2 = (2*radius + 1)**2
    
    alpha0 = 0.5 * torch.tensor([p2-2]).type(sigmaMap.dtype).to(sigmaMap.device)
    beta0 = 0.5 * p2 * sigmaMap

    # Gaussain distribution params (Z)
    out_denoise[:, C:,].clamp_(min=log_min, max=log_max)
    err_mean = out_denoise[:, :C,]
    m2 = torch.exp(out_denoise[:, C:,])

    # Inverse Gamma distribution params (Sigma)
    out_sigma.clamp_(min=log_min, max=log_max)
    log_alpha, log_beta = out_sigma[:, :C,], out_sigma[:, C:,]
    alpha = torch.exp(log_alpha)
    alpha_div_beta = torch.exp(log_alpha - log_beta)

    # KL Divergence calculations
    m2_div_eps = m2 / eps2
    err_mean_gt = signal_noisy - clean_signal
    kl_gauss = 0.5 * torch.mean((err_mean - err_mean_gt)**2/eps2 + (m2_div_eps - 1 - torch.log(m2_div_eps)))

    kl_Igamma = torch.mean((alpha - alpha0) * torch.digamma(alpha) + (log_gamma(alpha0) - log_gamma(alpha))
                           + alpha0 * (log_beta - torch.log(beta0)) + beta0 * alpha_div_beta - alpha)

    # Likelihood
    lh = 0.5 * log(2*pi) + 0.5 * torch.mean((log_beta - torch.digamma(alpha)) + (err_mean**2 + m2) * alpha_div_beta)

    return lh + kl_gauss + kl_Igamma